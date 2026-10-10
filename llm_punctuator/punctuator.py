"""Transformers-based LLM punctuator implementations."""

import itertools
import logging
import os
import re

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from llm_punctuator.logits_processor import CustomLogitsProcessor
from llm_punctuator.schema import Message, Role

from .prompt import EN_SYSTEM_PROMPT, ZH_SYSTEM_PROMPT
from .schema import EN_PUNCTUATIONS, ZH_PUNCTUATIONS

# Disable tokenizers parallelism to avoid fork warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)

ZH_UNIT = re.compile(r"\s*(?:[A-Za-z0-9Ａ-Ｚａ-ｚ０-９]+|.)", re.DOTALL)


class TransformersLLMPunctuator:
    """Base class for transformer-based LLM punctuators.

    This class provides common functionality for all transformer-based punctuators,
    including model loading, text chunking, and constrained generation.
    Supports multiple languages through the add_punctuation method.
    """

    def __init__(self, model_name_or_path: str) -> None:
        """Initialize the transformer-based punctuator.

        Args:
            model_name_or_path: HuggingFace model name or local path.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, device_map="auto", dtype="auto"
        )
        self.device = self.model.device
        logger.info(f"Model loaded on device: {self.device}")

        # Extract the assistant closing tag once during initialization
        # This is used to remove the closing tag from prompts to allow continued generation
        self._assistant_closing_tag = self._extract_assistant_closing_tag()

    def _extract_assistant_closing_tag(self) -> str:
        """Extract the assistant closing tag from the tokenizer's chat template.

        This is done once during initialization by creating a dummy conversation
        and extracting the pattern that appears after the assistant content.

        Returns:
            The assistant closing tag string (e.g., "<|im_end|>").
        """
        # Use a unique marker that won't appear in system prompts
        unique_marker = "<<<UNIQUE_MARKER_FOR_CLOSING_TAG>>>"
        dummy_messages = [
            {"role": "user", "content": "test"},
            {"role": "assistant", "content": unique_marker},
        ]
        dummy_prompt = self.tokenizer.apply_chat_template(
            dummy_messages, tokenize=False, add_generation_prompt=False
        )

        # Find the closing tag by looking at what comes after our unique marker
        if unique_marker in dummy_prompt:
            closing_tag = dummy_prompt.split(unique_marker, 1)[1]
            logger.debug(f"Extracted assistant closing tag: {repr(closing_tag)}")
            return closing_tag

        # Fallback: return empty string if we can't extract the tag
        logger.warning("Could not extract assistant closing tag, using empty string")
        return ""

    def apply_chat_template(self, messages: list[Message]) -> str:
        """Apply chat template to messages using the tokenizer's built-in template.

        This method removes the assistant closing tag to allow continued generation,
        since we always provide an assistant message (empty for first chunk, with
        content for subsequent chunks).

        Args:
            messages: List of chat messages. The last message should be from assistant.

        Returns:
            Formatted prompt string for the model without the assistant closing tag.
        """
        # Convert Message objects to dictionaries for tokenizer
        chat_messages = [{"role": msg.role, "content": msg.content} for msg in messages]

        # Generate the prompt without adding generation prompt
        # (since we already have an assistant message)
        prompt = self.tokenizer.apply_chat_template(
            chat_messages, tokenize=False, add_generation_prompt=False
        )

        # Remove the assistant closing tag to allow the model to continue generation
        if self._assistant_closing_tag and prompt.endswith(self._assistant_closing_tag):
            prompt = prompt[: -len(self._assistant_closing_tag)]

        logger.debug(f"Generated prompt:\n{prompt}")
        return prompt

    @torch.no_grad()
    def add_punctuation(
        self,
        text: str,
        punctuations: str | None = None,
        language: str = "zh",
        system_prompt: str | None = None,
        chunk_size: int = 200,
        k: int = -1,
    ) -> str:
        """Add punctuation to text using the LLM.

        Args:
            text: Input text without punctuation.
            punctuations: String of allowed punctuation characters. If None, uses default for language.
            language: Language code ("zh" for Chinese, "en" for English). Default: "zh".
            system_prompt: Custom system prompt. If None, uses default for language.
            chunk_size: Number of tokens per chunk for processing. A chunk that would end inside
                a segment runs on to the end of that segment.
            k: Most positions where a mark may go to decide per forward pass. -1 decides the
                whole rest of the chunk. Default: -1.

        Returns:
            Text with punctuation added.

        Raises:
            ValueError: If the specified language is not supported, or k is 0 or below -1.
        """
        if k == 0 or k < -1:
            raise ValueError(f"k must be -1 or a positive integer, got {k}")

        # Select default punctuations based on language if not provided
        if punctuations is None:
            if language == "zh":
                punctuations = ZH_PUNCTUATIONS
            elif language == "en":
                punctuations = EN_PUNCTUATIONS
            else:
                raise ValueError(
                    f"Language {language} is not supported. Supported languages: zh, en"
                )

        # Select default system prompt based on language if not provided
        if system_prompt is None:
            if language == "zh":
                system_prompt = ZH_SYSTEM_PROMPT
            elif language == "en":
                system_prompt = EN_SYSTEM_PROMPT
            else:
                raise ValueError(
                    f"Language {language} is not supported. Supported languages: zh, en"
                )

        # Format the system prompt with the actual punctuations
        # This allows dynamic insertion of the allowed punctuation marks into the prompt
        try:
            system_prompt = system_prompt.format(punctuations=punctuations)
        except KeyError:
            # If the prompt doesn't have the placeholder, just use it as is
            pass

        segments = self.split_segments(text, language)

        # Encode punctuation tokens individually to ensure correct token IDs
        # Some tokenizers might merge consecutive punctuation or handle them differently
        punctuation_tokens = []
        for p in punctuations:
            # Get the token ID for the punctuation mark
            # We take the last token because some tokenizers might add a start token
            p_tokens = self.encode_text(p)
            if p_tokens:
                punctuation_tokens.append(p_tokens[-1])

        # Remove duplicates while preserving order
        punctuation_tokens = list(dict.fromkeys(punctuation_tokens))

        chunks = self.chunk_segments(segments, chunk_size)
        chunk_nums = len(chunks)
        prev_decode_tokens = []
        prev_chunk = []
        result_tokens = []
        for chunk_idx, chunk_segments in enumerate(tqdm(chunks)):
            chunk = [token for segment in chunk_segments for token in segment]
            chunk_text = self.decode(prev_chunk + chunk)
            assistant_prefix = self.decode(prev_decode_tokens)

            messages = [
                Message(role=Role.System, content=system_prompt),
                Message(role=Role.User, content=chunk_text),
                Message(role=Role.Assistant, content=assistant_prefix),
            ]

            input_prompt = self.apply_chat_template(messages)

            prompt_ids = self.tokenizer(input_prompt)["input_ids"]

            rule = CustomLogitsProcessor(
                chunk + [self.tokenizer.eos_token_id],
                punctuation_tokens,
                has_prev_input=chunk_idx != 0,
                unit_ends=set(itertools.accumulate(len(segment) for segment in chunk_segments)),
            )

            generated_tokens = self.greedy_search(prompt_ids, rule, k)

            # Remove trailing punctuation from non-final chunks to prevent consecutive
            # punctuation at chunk boundaries
            generated_tokens = self.remove_punctuation(
                generated_tokens, punctuation_tokens, chunk_idx == chunk_nums - 1
            )

            generated_result = self.decode(generated_tokens)
            logger.debug(f"chunk result: {generated_result}")
            prev_decode_tokens = generated_tokens
            prev_chunk = chunk
            result_tokens.extend(generated_tokens)

        result = self.decode(result_tokens)
        return result

    def split_segments(self, text: str, language: str) -> list[list[int]]:
        """Encode text and group its tokens into segments a mark may not go inside.

        A mark may go only where a segment ends. For zh, a segment ends where a unit ends and
        the next token does not start inside that unit. A unit is one Chinese character, one
        run of letters or digits, or one other character, with the whitespace before it.
        Full-width letters and digits count as letters and digits. A token that covers two
        units, such as one token for two characters, stays in one segment. For other
        languages, each token is a segment.

        Args:
            text: Text to split.
            language: Language code of the text.

        Returns:
            Token IDs of each segment, in text order. Together they are the tokenizer's
            encoding of the whole text.
        """
        if language != "zh":
            return [[token] for token in self.encode_text(text)]

        encoding = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        tokens, offsets = encoding["input_ids"], encoding["offset_mapping"]
        unit_ends = set(itertools.accumulate(len(unit) for unit in ZH_UNIT.findall(text)))
        cuts = [
            i + 1
            for i in range(len(tokens) - 1)
            if offsets[i][1] in unit_ends and offsets[i + 1][0] >= offsets[i][1]
        ]
        bounds = [0, *cuts, len(tokens)]
        return [tokens[start:end] for start, end in itertools.pairwise(bounds)]

    @staticmethod
    def chunk_segments(segments: list[list[int]], chunk_size: int) -> list[list[list[int]]]:
        """Group consecutive segments into chunks without splitting a segment.

        Args:
            segments: Token IDs of each segment, in text order.
            chunk_size: Number of tokens per chunk. A chunk closes at the first segment end at
                or past this size, so only the last chunk may be shorter.

        Returns:
            The segments of each chunk, in text order.
        """
        chunks = []
        size = chunk_size
        for segment in segments:
            if size >= chunk_size:
                chunks.append([])
                size = 0
            chunks[-1].append(segment)
            size += len(segment)
        return chunks

    def encode_text(self, text: str) -> list[int]:
        """Encode text to token IDs, removing special tokens.

        Args:
            text: Text to encode.

        Returns:
            List of token IDs without BOS/EOS tokens.
        """
        text_tokens = self.tokenizer.encode(text)
        if text_tokens[0] == self.tokenizer.bos_token_id:
            text_tokens = text_tokens[1:]
        if text_tokens[-1] == self.tokenizer.eos_token_id:
            text_tokens = text_tokens[:-1]
        return text_tokens

    def decode(self, tokens: list[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text.

        Args:
            tokens: List of token IDs to decode.
            skip_special_tokens: Whether to skip special tokens in output.

        Returns:
            Decoded text string.
        """
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def remove_punctuation(
        self, generated_tokens: list[int], punctuation_tokens: list[int], is_last_chunk: bool
    ) -> list[int]:
        """Remove trailing punctuation from generated tokens based on chunk position.

        This prevents consecutive punctuation at chunk boundaries. For non-final chunks,
        we remove all trailing punctuation since the next chunk may start with punctuation.
        For the final chunk, we keep the punctuation as it's the end of the text.

        Args:
            generated_tokens: List of generated token IDs.
            punctuation_tokens: List of punctuation token IDs.
            is_last_chunk: Whether this is the last chunk of text.

        Returns:
            List of tokens with trailing punctuation removed (if not last chunk).
        """
        if not generated_tokens:
            return generated_tokens

        punctuation_set = set(punctuation_tokens)

        if is_last_chunk:
            # Last chunk: keep all punctuation
            return generated_tokens
        else:
            # Non-last chunk: remove all trailing punctuation to avoid consecutive punctuation
            # at chunk boundaries (since next chunk may start with punctuation)
            while generated_tokens and generated_tokens[-1] in punctuation_set:
                generated_tokens = generated_tokens[:-1]
            return generated_tokens

    def greedy_search(
        self, prompt_ids: list[int], rule: CustomLogitsProcessor, k: int
    ) -> list[int]:
        """Greedy decoding under the rule, deciding up to k mark positions per forward pass.

        The text is known, so one forward over the tokens not yet cached plus the upcoming
        text gives the scores at every upcoming position. Each position takes the highest
        scoring allowed token, ties going to the lowest token ID as in step-by-step greedy
        decoding. Where a mark wins, the text after it was fed in the wrong place: it is
        cropped from the cache, and the next forward continues after the mark.

        Args:
            prompt_ids: Token IDs of the prompt.
            rule: Which tokens each position allows.
            k: Most positions where a mark may go to decide per forward. -1 for all.

        Returns:
            Generated token IDs, without the EOS token.
        """
        targets = rule.original_text_tokens
        text_len = len(targets) - 1
        cache = DynamicCache()
        emitted: list[int] = []
        pending, start = prompt_ids, 0
        while True:
            upcoming = sorted(p for p in rule.mark_positions if p >= start)
            end = upcoming[k - 1] if 0 < k <= len(upcoming) else text_len
            fed = pending + targets[start:end]
            cached = cache.get_seq_length()
            logits = self.model(
                input_ids=torch.tensor([fed], device=self.device),
                past_key_values=cache,
                use_cache=True,
            ).logits[0, len(pending) - 1 :]
            candidates = sorted(set(targets[start : end + 1]) | rule.punctuation_tokens)
            rows = logits[:, candidates].float().tolist()
            for pos in range(start, end + 1):
                score = dict(zip(candidates, rows[pos - start], strict=True))
                allowed = sorted(rule.allowed_tokens(pos, emitted[-1] if emitted else None))
                choice = max(allowed, key=score.__getitem__)
                emitted.append(choice)
                due = targets[pos]
                if choice != due:
                    cache.crop(cached + len(pending) + pos - start)
                    emitted.append(due)
                    pending = [choice, due]
                    break
            else:
                pending = [due]
            if emitted[-1] == targets[-1]:
                return emitted[:-1]
            start = pos + 1
