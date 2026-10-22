"""Transformers-based LLM punctuator implementations."""

import itertools
import logging
import os
import re
from collections.abc import Generator
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, PreTrainedModel

from llm_punctuator.logits_processor import CustomLogitsProcessor
from llm_punctuator.schema import Message, Role

from .prompt import EN_SYSTEM_PROMPT, ZH_SYSTEM_PROMPT
from .schema import EN_PUNCTUATIONS, ZH_PUNCTUATIONS

# Disable tokenizers parallelism to avoid fork warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)

ChunkJob = tuple[list[int], CustomLogitsProcessor]
ChunkLoop = Generator[ChunkJob, list[int], str]

ALNUM_RUN = r"[A-Za-z0-9Ａ-Ｚａ-ｚ０-９]+"
ZH_UNIT = re.compile(rf"\s*(?:{ALNUM_RUN}(?:[-.]{ALNUM_RUN})*|.)", re.DOTALL)


class ForcedPunctuator:
    """Forced decoding that puts punctuation into a known text.

    Subclasses set self.tokenizer, self.model and self.device, and may override forward.
    """

    tokenizer: Any
    model: PreTrainedModel
    device: torch.device

    @staticmethod
    def default_punctuations(language: str) -> str:
        """Return the default punctuation marks of a language.

        Args:
            language: Language code ("zh" or "en").

        Returns:
            The default punctuation marks of the language.

        Raises:
            ValueError: If the language is not supported.
        """
        if language == "zh":
            return ZH_PUNCTUATIONS
        if language == "en":
            return EN_PUNCTUATIONS
        raise ValueError(f"Language {language} is not supported. Supported languages: zh, en")

    def punctuation_token_ids(self, punctuations: str) -> list[int]:
        """Return the token ID of each punctuation mark, without duplicates.

        Args:
            punctuations: String of allowed punctuation characters.

        Returns:
            Token IDs in the order of first appearance.
        """
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
        return list(dict.fromkeys(punctuation_tokens))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: DynamicCache,
    ) -> torch.Tensor:
        """Run the model once and return its logits.

        Args:
            input_ids: Token IDs to feed, shape (batch, width).
            attention_mask: 2D mask over the cache and the fed tokens.
            position_ids: Position of each fed token.
            cache_position: Cache slot of each fed token.
            past_key_values: Cache that this call extends.

        Returns:
            Logits of each fed token.
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            past_key_values=past_key_values,
            use_cache=True,
        ).logits

    def split_segments(self, text: str, language: str) -> list[list[int]]:
        """Encode text and group its tokens into segments a mark may not go inside.

        A mark may go only where a segment ends. For zh, a segment ends where a unit ends and
        the next token does not start inside that unit. A unit is one Chinese character, one
        run of letters or digits, or one other character, with the whitespace before it.
        Full-width letters and digits count as letters and digits. A half-width "-" or "."
        with a letter or digit on both sides joins the runs on its two sides into one run.
        A token that covers two units, such as one token for two characters, stays in one
        segment. For other languages, each token is a segment.

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

    def greedy_search(self, jobs: list[ChunkJob], k: int) -> list[list[int]]:
        """Greedy decoding of several rows together, deciding up to k mark positions per forward.

        Each forward feeds every unfinished row, see GreedyRow. Rows put marks in different
        places, so the shared cache is never cropped. Text a row fed after its mark stays in
        the cache, masked out of that row's attention, and each row keeps its own positions.
        The mask is 2D, so transformers builds the causal mask each attention backend needs.

        Args:
            jobs: The prompt token IDs and the rule of each row.
            k: Most positions where a mark may go to decide per row per forward. -1 for all.

        Returns:
            Generated token IDs of each row, without the EOS token.
        """
        rows = [GreedyRow(prompt_ids, rule) for prompt_ids, rule in jobs]
        cache = DynamicCache()
        seen = torch.zeros(len(rows), 0, dtype=torch.long)
        positions = [0] * len(rows)
        while not all(row.done for row in rows):
            feeds = [[] if row.done else row.feed(k) for row in rows]
            width = max(map(len, feeds))
            # Padding follows each row's feed: causal attention hides it, and kept drops it from
            # seen. So its token ID does not matter.
            ids = torch.full((len(rows), width), self.tokenizer.eos_token_id)
            position_ids = torch.zeros(len(rows), width, dtype=torch.long)
            for b, fed in enumerate(feeds):
                ids[b, : len(fed)] = torch.tensor(fed, dtype=torch.long)
                position_ids[b, : len(fed)] = torch.arange(positions[b], positions[b] + len(fed))
            past = seen.shape[1]
            seen = torch.cat([seen, torch.ones(len(rows), width, dtype=torch.long)], dim=1)
            logits = self.forward(
                input_ids=ids.to(self.device),
                attention_mask=seen.to(self.device),
                position_ids=position_ids.to(self.device),
                cache_position=torch.arange(past, past + width, device=self.device),
                past_key_values=cache,
            )
            for b, row in enumerate(rows):
                kept = row.take(logits[b]) if feeds[b] else 0
                seen[b, past + kept :] = 0
                positions[b] += kept
        return [row.emitted[:-1] for row in rows]


class TransformersLLMPunctuator(ForcedPunctuator):
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
        texts: list[str],
        punctuations: str | None = None,
        language: str = "zh",
        system_prompt: str | None = None,
        chunk_size: int = 200,
        k: int = -1,
        batch_size: int = 8,
    ) -> list[str]:
        """Add punctuation to texts using the LLM, decoding several texts per forward pass.

        Args:
            texts: Input texts without punctuation.
            punctuations: String of allowed punctuation characters. If None, uses default for language.
            language: Language code ("zh" for Chinese, "en" for English). Default: "zh".
            system_prompt: Custom system prompt. If None, uses default for language.
            chunk_size: Number of tokens per chunk for processing. A chunk that would end inside
                a segment runs on to the end of that segment.
            k: Most positions where a mark may go to decide per forward pass. -1 decides the
                whole rest of the chunk. Default: -1.
            batch_size: Most texts to decode together. Default: 8.

        Returns:
            Each text with punctuation added, in the order of texts.

        Raises:
            TypeError: If texts is a str.
            ValueError: If the specified language is not supported, k is 0 or below -1, or
                batch_size is below 1.
        """
        if isinstance(texts, str):
            raise TypeError("texts must be a list of str, got str. Wrap a single text as [text].")
        if batch_size < 1:
            raise ValueError(f"batch_size must be a positive integer, got {batch_size}")
        if k == 0 or k < -1:
            raise ValueError(f"k must be -1 or a positive integer, got {k}")

        # Select default punctuations based on language if not provided
        if punctuations is None:
            punctuations = self.default_punctuations(language)

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

        punctuation_tokens = self.punctuation_token_ids(punctuations)

        loops = [
            self.punctuate_chunks(text, language, system_prompt, punctuation_tokens, chunk_size)
            for text in texts
        ]
        return self.run_loops(loops, batch_size, k)

    def run_loops(self, loops: list[ChunkLoop], batch_size: int, k: int) -> list[str]:
        """Run chunk loops with up to batch_size of them decoded together.

        Each round decodes the current chunk of every loop in flight. A loop's next chunk waits
        for the round, since its prompt holds the output of the chunk before it. A loop that
        finishes frees its place for the next waiting loop.

        Args:
            loops: Chunk loops from punctuate_chunks.
            batch_size: Most loops in flight.
            k: Most mark positions to decide per row per forward. -1 for all.

        Returns:
            The result of each loop, in the order of loops.
        """
        results = [""] * len(loops)
        waiting = iter(enumerate(loops))
        in_flight: dict[int, tuple[ChunkLoop, ChunkJob]] = {}
        progress = tqdm(total=len(loops))

        def advance(idx: int, loop: ChunkLoop, generated: list[int] | None) -> None:
            try:
                in_flight[idx] = (loop, next(loop) if generated is None else loop.send(generated))
            except StopIteration as stop:
                results[idx] = stop.value
                in_flight.pop(idx, None)
                progress.update()

        while True:
            while len(in_flight) < batch_size and (item := next(waiting, None)):
                advance(*item, None)
            if not in_flight:
                progress.close()
                return results
            jobs = list(in_flight.items())
            outputs = self.greedy_search([job for _, (_, job) in jobs], k)
            for (idx, (loop, _)), generated in zip(jobs, outputs, strict=True):
                advance(idx, loop, generated)

    def punctuate_chunks(
        self,
        text: str,
        language: str,
        system_prompt: str,
        punctuation_tokens: list[int],
        chunk_size: int,
    ) -> ChunkLoop:
        """Walk one text chunk by chunk, handing each chunk out to be decoded.

        Args:
            text: Input text without punctuation.
            language: Language code of the text.
            system_prompt: System prompt with the punctuations filled in.
            punctuation_tokens: Token IDs of the allowed marks.
            chunk_size: Number of tokens per chunk.

        Yields:
            The prompt token IDs and the rule of each chunk, in text order. Send back the
            tokens generated for it.

        Returns:
            Text with punctuation added.
        """
        segments = self.split_segments(text, language)
        chunks = self.chunk_segments(segments, chunk_size)
        chunk_nums = len(chunks)
        prev_decode_tokens = []
        prev_chunk = []
        result_tokens = []
        for chunk_idx, chunk_segments in enumerate(chunks):
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

            generated_tokens = yield prompt_ids, rule

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


class GreedyRow:
    """One row of greedy_search: a chunk decoded under its rule, a few positions per forward.

    The text is known, so one forward over the tokens not yet fed plus the upcoming text gives
    the scores at every upcoming position. Each mark's score gets the rule's mark bonus. Each
    position takes the highest scoring allowed token, ties going to the lowest token ID as in
    step-by-step greedy decoding. Where a mark wins, the text fed after it was in the wrong
    place, and the next forward continues after the mark.
    """

    def __init__(self, prompt_ids: list[int], rule: CustomLogitsProcessor) -> None:
        """Start the row before its prompt is fed.

        Args:
            prompt_ids: Token IDs of the prompt.
            rule: Which tokens each position allows.
        """
        self.rule = rule
        self.targets = rule.original_text_tokens
        self.emitted: list[int] = []
        self.last_mark = 0
        self.pending = prompt_ids
        self.start = 0
        self.end = 0
        self.done = False

    def feed(self, k: int) -> list[int]:
        """Token IDs to feed next, deciding up to k mark positions. -1 for all."""
        upcoming = sorted(p for p in self.rule.mark_positions if p >= self.start)
        self.end = upcoming[k - 1] if 0 < k <= len(upcoming) else len(self.targets) - 1
        return self.pending + self.targets[self.start : self.end]

    def take(self, logits: torch.Tensor) -> int:
        """Decide positions from the logits of the last feed.

        Args:
            logits: Scores at each fed position and any padding after it, shaped
                (sequence length, vocab size).

        Returns:
            How many fed tokens are still valid to attend to.
        """
        start, pending = self.start, self.pending
        candidates = sorted(set(self.targets[start : self.end + 1]) | self.rule.punctuation_tokens)
        fed_len = len(pending) + self.end - start
        rows = logits[len(pending) - 1 : fed_len, candidates].float().tolist()
        for pos in range(start, self.end + 1):
            score = dict(zip(candidates, rows[pos - start], strict=True))
            bonus = self.rule.mark_bonus(pos, self.last_mark)
            for mark in self.rule.punctuation_tokens:
                score[mark] += bonus
            allowed = self.rule.allowed_tokens(pos, self.emitted[-1] if self.emitted else None)
            choice = max(sorted(allowed), key=score.__getitem__)
            self.emitted.append(choice)
            due = self.targets[pos]
            if choice != due:
                self.emitted.append(due)
                self.last_mark = pos
                self.pending = [choice, due]
                break
        else:
            self.pending = [due]
        self.done = self.emitted[-1] == self.targets[-1]
        self.start = pos + 1
        return len(pending) + pos - start
