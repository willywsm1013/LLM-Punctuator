import torch
from transformers import LogitsProcessor

class CustomLogitsProcessor(LogitsProcessor):
    def __init__(self,
                 original_text_tokens,
                 punctuation_tokens,
                 has_prev_input:bool):
        self.original_text_tokens = original_text_tokens
        self.punctuation_tokens = punctuation_tokens
        self.is_first_token = True
        self.has_prev_input = has_prev_input
        self.current_index = 0

    def __call__(self, input_ids, scores):
        # first token only allow original text token
        if self.is_first_token:
            allowed_tokens = set([self.original_text_tokens[self.current_index]])
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
        logits = torch.full_like(scores, -float('inf'))
        for token in allowed_tokens:
            logits[:, token] = scores[:, token]
        return logits

class BeamSearchCustomLogitsProcessor(LogitsProcessor):
    def __init__(self,
                 original_text_tokens,
                 punctuation_tokens,
                 has_prev_input: bool,
                 num_beams: int):
        self.original_text_tokens = original_text_tokens
        self.punctuation_tokens = punctuation_tokens
        self.has_prev_input = has_prev_input
        self.num_beams = num_beams
        self.beam_states = [{"is_first_token": True, "current_index": 0} for _ in range(num_beams)]

    def __call__(self,
                 input_ids: torch.LongTensor,
                 scores: torch.FloatTensor) -> torch.FloatTensor:
        batch_size, vocab_size = scores.shape
        new_scores = torch.full_like(scores, float('-inf'))

        for beam_idx in range(self.num_beams):
            beam_state = self.beam_states[beam_idx]
            beam_input_ids = input_ids[beam_idx]

            if beam_state["is_first_token"]:
                allowed_tokens = set([self.original_text_tokens[beam_state["current_index"]]])
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