"""Transformers-based LLM punctuator implementations."""

import abc
import logging
import math
import os

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

# Disable tokenizers parallelism to avoid fork warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from llm_punctuator.items import Message, Role
from llm_punctuator.logits_processor import BeamSearchCustomLogitsProcessor, CustomLogitsProcessor

from .base import LLMPunctuator
from .prompt import ZH_SYSTEM_PROMPT


class TransformersAutoPunctuator:
    """Factory class for creating transformer-based punctuators."""

    @classmethod
    def from_pretrained(
        cls, model_name_or_path: str, language: str = "zh"
    ) -> "TransformersLLMPunctuator":
        """Load a pre-trained punctuator model.

        Args:
            model_name_or_path: HuggingFace model name or local path.
            language: Language code (currently only "zh" is supported).

        Returns:
            An instance of the appropriate punctuator class for the model.
        """
        Punctuator = PATH_TO_TRANSFORMERS_PUNCTUATOR[model_name_or_path]
        return Punctuator(model_name_or_path, language)


class TransformersLLMPunctuator(LLMPunctuator):
    """Base class for transformer-based LLM punctuators.

    This class provides common functionality for all transformer-based punctuators,
    including model loading, text chunking, and constrained generation.
    """

    @abc.abstractmethod
    def apply_chat_template(self, messages: list[Message]) -> str:
        """Apply model-specific chat template to messages.

        Args:
            messages: List of chat messages.

        Returns:
            Formatted prompt string for the model.
        """
        pass

    def __init__(self, model_name_or_path: str, language: str) -> None:
        """Initialize the transformer-based punctuator.

        Args:
            model_name_or_path: HuggingFace model name or local path.
            language: Language code (currently only "zh" is supported).

        Raises:
            ValueError: If the specified language is not supported.
        """
        if language == "zh":
            self.system_prompt = ZH_SYSTEM_PROMPT
        else:
            raise ValueError(f"Language {language} is not supported.")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, device_map="auto", dtype="auto"
        )
        self.device = self.model.device

    @torch.no_grad()
    def add_punctuation(
        self,
        text: str,
        punctuations: str,
        system_prompt: str | None = None,
        chunk_size: int = 200,
        num_beams: int = 1,
    ) -> str:
        """Add punctuation to text using the LLM.

        Args:
            text: Input text without punctuation.
            punctuations: String of allowed punctuation characters.
            system_prompt: Custom system prompt (uses default if None).
            chunk_size: Number of tokens per chunk for processing.
            num_beams: Number of beams for beam search (1 for greedy search).

        Returns:
            Text with punctuation added.
        """
        if system_prompt is None:
            system_prompt = self.system_prompt

        text_tokens = self.encode_text(text)
        punctuation_tokens = self.encode_text(punctuations)

        chunk_nums = math.ceil(len(text_tokens) / chunk_size)
        chunks = [text_tokens[i * chunk_size : (i + 1) * chunk_size] for i in range(chunk_nums)]
        prev_decode_tokens = []
        prev_chunk = []
        result_tokens = []
        for chunk_idx, chunk in enumerate(tqdm(chunks)):
            chunk_text = self.decode(prev_chunk + chunk)
            assistant_prefix = self.decode(prev_decode_tokens)

            messages = [
                Message(role=Role.System, content=system_prompt),
                Message(role=Role.User, content=chunk_text),
                Message(role=Role.Assistant, content=assistant_prefix),
            ]

            input_prompt = self.apply_chat_template(messages)
            inputs = self.tokenizer(input_prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)

            has_prev_input = chunk_idx != 0
            max_length = input_ids.shape[1] + int(len(chunk) * 1.2)
            if num_beams == 1:
                logits_processor = CustomLogitsProcessor(
                    chunk + [self.tokenizer.eos_token_id],
                    punctuation_tokens,
                    has_prev_input=has_prev_input,
                )

                logits_processor_list = LogitsProcessorList([logits_processor])

                generated_tokens = self.greedy_search(
                    input_ids, attention_mask, logits_processor_list, max_length
                )
            else:
                logits_processor = BeamSearchCustomLogitsProcessor(
                    chunk + [self.tokenizer.eos_token_id],
                    punctuation_tokens,
                    has_prev_input=has_prev_input,
                    num_beams=num_beams,
                )

                logits_processor_list = LogitsProcessorList([logits_processor])
                generated_tokens = self.beam_search(
                    input_ids, attention_mask, logits_processor_list, max_length, num_beams
                )
            generated_tokens = generated_tokens.squeeze(0).tolist()
            generated_tokens = self.remove_punctuation(
                generated_tokens, punctuation_tokens, chunk_idx == chunk_nums - 1
            )

            generated_result = self.decode(generated_tokens)
            logging.debug(f"chunk result: {generated_result}")
            prev_decode_tokens = generated_tokens
            prev_chunk = chunk
            result_tokens.extend(generated_tokens)

        result = self.decode(result_tokens)
        return result

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
    ) -> str:
        """Remove excessive punctuation from generated tokens.

        Args:
            generated_tokens: List of generated token IDs.
            punctuation_tokens: List of punctuation token IDs.
            is_last_chunk: Whether this is the last chunk of text.

        Returns:
            List of tokens with excessive punctuation removed.
        """
        # remove too many punctuations in the end
        if is_last_chunk:
            # keep at most 1 punctuation in the end of the last chunk
            while (
                generated_tokens[-1] in punctuation_tokens
                and generated_tokens[-2] in punctuation_tokens
            ):
                generated_tokens = generated_tokens[:-1]
        else:
            while generated_tokens[-1] in punctuation_tokens:
                generated_tokens = generated_tokens[:-1]
        return generated_tokens

    def greedy_search(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        logits_processor_list: LogitsProcessorList,
        max_length: int,
    ) -> torch.Tensor:
        """Perform greedy search generation.

        Args:
            input_ids: Input token IDs tensor.
            attention_mask: Attention mask tensor.
            logits_processor_list: List of logits processors.
            max_length: Maximum generation length.

        Returns:
            Generated token IDs tensor.
        """
        output = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            logits_processor=logits_processor_list,
            max_length=max_length,
            do_sample=False,
            temperature=1.0,
            top_p=1,
        )
        generated_tokens = output[0][input_ids.shape[1] :]
        if generated_tokens[-1] == self.tokenizer.eos_token_id:
            generated_tokens = generated_tokens[:-1]

        return generated_tokens

    def beam_search(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        logits_processor_list: LogitsProcessorList,
        max_length: int,
        num_beams: int,
    ) -> torch.Tensor:
        """Perform beam search generation.

        Args:
            input_ids: Input token IDs tensor.
            attention_mask: Attention mask tensor.
            logits_processor_list: List of logits processors.
            max_length: Maximum generation length.
            num_beams: Number of beams for beam search.

        Returns:
            Generated token IDs tensor.
        """
        output = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            logits_processor=logits_processor_list,
            max_length=max_length,
            num_beams=num_beams,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
        )

        generated_tokens = output[0][input_ids.shape[1] :]
        if generated_tokens[-1] == self.tokenizer.eos_token_id:
            generated_tokens = generated_tokens[:-1]

        return generated_tokens


class Gemma2Punctuator(TransformersLLMPunctuator):
    """Punctuator implementation for Google Gemma2 models."""

    def apply_chat_template(self, messages: list[Message]) -> str:
        """Apply Gemma2-specific chat template.

        Args:
            messages: List of chat messages.

        Returns:
            Formatted prompt string for Gemma2.
        """
        # This is Gemma2 chat template
        messages = [
            Message(
                role=Role.User,
                content=messages[0].content + "以下是你要標的文章：\n" + messages[1].content,
            )
        ] + messages[2:]
        ret = "<|begin_of_text|>"
        for m in messages:
            ret += "<|start_header_id|>" + m.role + "<|end_header_id|>\n\n"
            ret += m.content
            if m.role != Role.Assistant:
                ret += "<|eot_id|>"
        return ret


class Llama3Punctuator(TransformersLLMPunctuator):
    """Punctuator implementation for Meta Llama3 models."""

    def apply_chat_template(self, messages: list[Message]) -> str:
        """Apply Llama3-specific chat template.

        Args:
            messages: List of chat messages.

        Returns:
            Formatted prompt string for Llama3.
        """
        # This is Llama3 and llama3.1 chat template
        ret = "<|begin_of_text|>"
        for m in messages:
            ret += "<|start_header_id|>" + m.role + "<|end_header_id|>\n\n"
            ret += m.content
            if m.role != Role.Assistant:
                ret += "<|eot_id|>"
        return ret


class Qwen2Punctuator(TransformersLLMPunctuator):
    """Punctuator implementation for Qwen2 models."""

    def apply_chat_template(self, messages: list[Message]) -> str:
        """Apply Qwen2-specific chat template.

        Args:
            messages: List of chat messages.

        Returns:
            Formatted prompt string for Qwen2.
        """
        # this is Qwen2 chat template
        ret = ""
        for m in messages:
            ret += "<|im_start|>" + m.role + "\n"
            ret += m.content
            if m.role != Role.Assistant:
                ret += "<|im_end|>\n"
        return ret


class YiPunctuator(LLMPunctuator):
    """Punctuator implementation for Yi models."""

    def apply_chat_template(self, messages: list[Message]) -> str:
        """Apply Yi-specific chat template.

        Args:
            messages: List of chat messages.

        Returns:
            Formatted prompt string for Yi.
        """
        # This is Yi chat template
        ret = ""
        for m in messages:
            if m.role == Role.System:
                ret += m.content
            else:
                ret += "<|im_start|>" + m.role + "\n"
                ret += m.content
                if m.role != Role.Assistant:
                    ret += "<|im_end|>\n"
        return ret


PATH_TO_TRANSFORMERS_PUNCTUATOR = {
    "Qwen/Qwen2-7B-Instruct": Qwen2Punctuator,
    "Qwen/Qwen2-1.5B-Instruct": Qwen2Punctuator,
    "Qwen/Qwen2-0.5B-Instruct": Qwen2Punctuator,
    "taide/Llama3-TAIDE-LX-8B-Chat-Alpha1": Llama3Punctuator,
    "01-ai/Yi-1.5-6B-Chat": YiPunctuator,
    "01-ai/Yi-1.5-9B-Chat": YiPunctuator,
    "meta-llama/Meta-Llama-3.1-8B-Instruct": Llama3Punctuator,
    "google/gemma-2-2b-it": Gemma2Punctuator,
}
