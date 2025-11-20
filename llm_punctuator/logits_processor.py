"""Custom logits processors for constrained text generation."""

import logging

import torch
from transformers import LogitsProcessor

logger = logging.getLogger(__name__)


class CustomLogitsProcessor(LogitsProcessor):
    """Logits processor that constrains generation to original text tokens and punctuation.

    This processor ensures the model generates tokens in a specific pattern:
    1. Original text tokens must appear in order
    2. Punctuation can only be inserted between text tokens
    3. No consecutive punctuation is allowed
    4. The original text content is preserved exactly

    The processor works by:
    - Tracking which original text tokens have been generated
    - Allowing only valid next tokens based on the current state
    - Setting logits to -inf for all invalid tokens

    Example:
        Original tokens: [今天, 天氣, 很, 好]
        Punctuation: [，, 。]
        Valid output: 今天，天氣很好。
        Invalid: 今天天氣，，很好 (consecutive punctuation)
        Invalid: 今天很好 (skipped token)
    """

    def __init__(
        self,
        original_text_tokens: list[int],
        punctuation_tokens: list[int],
        has_prev_input: bool,
    ) -> None:
        """Initialize the custom logits processor.

        Args:
            original_text_tokens: Token IDs from the original text (including EOS). The last token should be the EOS token.
            punctuation_tokens: Token IDs for allowed punctuation marks.
            has_prev_input: If True, allows starting with punctuation (for continuing from a previous chunk). If False, must start with the first text token.
        """
        if not original_text_tokens:
            raise ValueError("original_text_tokens cannot be empty")

        self.original_text_tokens = original_text_tokens
        self.punctuation_tokens = set(punctuation_tokens)
        self.has_prev_input = has_prev_input

        # Store the EOS token separately for clarity
        self.eos_token = original_text_tokens[-1]
        # Text tokens without EOS
        self.text_tokens_only = original_text_tokens[:-1]

        # Track the prompt length to identify generated tokens
        self.prompt_length = None

        logger.debug(
            f"Initialized CustomLogitsProcessor: "
            f"{len(self.text_tokens_only)} text tokens, "
            f"{len(self.punctuation_tokens)} punctuation tokens, "
            f"has_prev_input={has_prev_input}"
        )

    def _count_generated_text_tokens(self, generated_ids: torch.LongTensor) -> int:
        """Count how many original text tokens have been generated.

        Args:
            generated_ids: The generated token IDs (excluding prompt).

        Returns:
            Number of original text tokens that have been generated.
        """
        count = 0
        for token_id in generated_ids:
            token_id_item = token_id.item()
            if count < len(self.text_tokens_only) and token_id_item == self.text_tokens_only[count]:
                count += 1
        return count

    def _get_allowed_tokens(
        self, num_text_tokens_generated: int, last_token_id: int | None
    ) -> set[int]:
        """Determine which tokens are allowed based on current generation state.

        Args:
            num_text_tokens_generated: How many text tokens have been generated so far.
            last_token_id: The last generated token ID, or None if this is the first token.

        Returns:
            Set of allowed token IDs for the next generation step.
        """
        allowed = set()

        # Check if we've generated all text tokens
        if num_text_tokens_generated >= len(self.text_tokens_only):
            # All text tokens generated, only allow EOS or punctuation
            allowed.add(self.eos_token)
            allowed.update(self.punctuation_tokens)
            return allowed

        # Get the next text token that should be generated
        next_text_token = self.text_tokens_only[num_text_tokens_generated]

        if last_token_id is None:
            # First token: must be the first text token
            allowed.add(next_text_token)
            # Special case: if continuing from previous chunk, can start with punctuation
            if self.has_prev_input:
                allowed.update(self.punctuation_tokens)
        elif last_token_id in self.punctuation_tokens:
            # Last token was punctuation: must generate next text token (no consecutive punctuation)
            allowed.add(next_text_token)
        else:
            # Last token was a text token: can add punctuation or continue with next text token
            allowed.add(next_text_token)
            allowed.update(self.punctuation_tokens)

        return allowed

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Process logits to constrain token generation.

        Args:
            input_ids: Input token IDs tensor of shape (batch_size, sequence_length).
                Includes both the prompt and any generated tokens so far.
            scores: Logits scores tensor of shape (batch_size, vocab_size).

        Returns:
            Modified logits tensor where only allowed tokens have valid scores,
            all other tokens have -inf scores.
        """
        # We only support batch_size=1 for now
        if input_ids.shape[0] != 1:
            raise ValueError(f"Only batch_size=1 is supported, got {input_ids.shape[0]}")

        # On first call, record the prompt length
        if self.prompt_length is None:
            self.prompt_length = input_ids.shape[1]
            logger.debug(f"Recorded prompt length: {self.prompt_length}")

        # Extract only the generated tokens (after the prompt)
        generated_ids = input_ids[0, self.prompt_length :]

        # Count how many text tokens have been generated
        num_text_tokens_generated = self._count_generated_text_tokens(generated_ids)

        # Get the last generated token (if any)
        last_token_id = generated_ids[-1].item() if len(generated_ids) > 0 else None

        # Determine allowed tokens
        allowed_tokens = self._get_allowed_tokens(num_text_tokens_generated, last_token_id)

        # Create mask: -inf for disallowed tokens, keep original scores for allowed tokens
        mask = torch.full_like(scores, float("-inf"))
        for token_id in allowed_tokens:
            mask[:, token_id] = scores[:, token_id]

        logger.debug(
            f"Generated {num_text_tokens_generated}/{len(self.text_tokens_only)} text tokens, "
            f"last_token={last_token_id}, allowed_tokens={len(allowed_tokens)}"
        )

        return mask
