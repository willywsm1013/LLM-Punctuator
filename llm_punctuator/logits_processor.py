"""Custom logits processors for constrained text generation."""

import torch
from transformers import LogitsProcessor


class CustomLogitsProcessor(LogitsProcessor):
    """Logits processor that constrains generation to original text tokens and punctuation.

    This processor ensures that the model can only generate tokens from the original
    text or specified punctuation marks, maintaining the original text structure while
    adding punctuation.

    Attributes:
        original_text_tokens: List of token IDs from the original text.
        punctuation_tokens: List of token IDs for allowed punctuation marks.
        is_first_token: Whether the next token is the first in the sequence.
        has_prev_input: Whether there is previous input context.
        current_index: Current position in the original text tokens.
    """

    def __init__(
        self, original_text_tokens: list[int], punctuation_tokens: list[int], has_prev_input: bool
    ) -> None:
        """Initialize the custom logits processor.

        Args:
            original_text_tokens: List of token IDs from the original text.
            punctuation_tokens: List of token IDs for allowed punctuation marks.
            has_prev_input: Whether there is previous input context.
        """
        self.original_text_tokens = original_text_tokens
        self.punctuation_tokens = punctuation_tokens
        self.is_first_token = True
        self.has_prev_input = has_prev_input
        self.current_index = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Process logits to constrain token generation.

        Args:
            input_ids: Input token IDs tensor.
            scores: Logits scores tensor.

        Returns:
            Modified logits tensor with only allowed tokens having valid scores.
        """
        # first token only allow original text token
        if self.is_first_token:
            allowed_tokens = {self.original_text_tokens[self.current_index]}
            if self.has_prev_input:
                allowed_tokens.update(self.punctuation_tokens)
            self.is_first_token = False
        else:
            if input_ids[0][-1] == self.original_text_tokens[self.current_index]:
                # if last token is word, increase current index
                allowed_tokens = set(self.punctuation_tokens)
                self.current_index += 1
            else:
                # if last token is punctuation, only allow word
                allowed_tokens = set()

            allowed_tokens.add(self.original_text_tokens[self.current_index])
        logits = torch.full_like(scores, -float("inf"))
        for token in allowed_tokens:
            logits[:, token] = scores[:, token]
        return logits


class BeamSearchCustomLogitsProcessor(LogitsProcessor):
    """Logits processor for beam search with constrained generation.

    This processor maintains separate states for each beam in beam search,
    ensuring that each beam can only generate tokens from the original text
    or specified punctuation marks.

    Attributes:
        original_text_tokens: List of token IDs from the original text.
        punctuation_tokens: List of token IDs for allowed punctuation marks.
        has_prev_input: Whether there is previous input context.
        num_beams: Number of beams for beam search.
        beam_states: List of state dictionaries for each beam.
    """

    def __init__(
        self,
        original_text_tokens: list[int],
        punctuation_tokens: list[int],
        has_prev_input: bool,
        num_beams: int,
    ) -> None:
        """Initialize the beam search custom logits processor.

        Args:
            original_text_tokens: List of token IDs from the original text.
            punctuation_tokens: List of token IDs for allowed punctuation marks.
            has_prev_input: Whether there is previous input context.
            num_beams: Number of beams for beam search.
        """
        self.original_text_tokens = original_text_tokens
        self.punctuation_tokens = punctuation_tokens
        self.has_prev_input = has_prev_input
        self.num_beams = num_beams
        self.beam_states = [{"is_first_token": True, "current_index": 0} for _ in range(num_beams)]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Process logits for beam search with constraints.

        Args:
            input_ids: Input token IDs tensor for all beams.
            scores: Logits scores tensor for all beams.

        Returns:
            Modified logits tensor with only allowed tokens having valid scores.
        """
        batch_size, vocab_size = scores.shape
        new_scores = torch.full_like(scores, float("-inf"))

        for beam_idx in range(self.num_beams):
            beam_state = self.beam_states[beam_idx]
            beam_input_ids = input_ids[beam_idx]

            if beam_state["is_first_token"]:
                allowed_tokens = {self.original_text_tokens[beam_state["current_index"]]}
                if self.has_prev_input:
                    allowed_tokens.update(self.punctuation_tokens)
                beam_state["is_first_token"] = False
            else:
                last_token = beam_input_ids[-1].item()
                if last_token == self.original_text_tokens[beam_state["current_index"]]:
                    allowed_tokens = set(self.punctuation_tokens)
                    beam_state["current_index"] += 1
                else:
                    allowed_tokens = set()

                if beam_state["current_index"] < len(self.original_text_tokens):
                    allowed_tokens.add(self.original_text_tokens[beam_state["current_index"]])

            for token in allowed_tokens:
                new_scores[beam_idx, token] = scores[beam_idx, token]

        return new_scores
