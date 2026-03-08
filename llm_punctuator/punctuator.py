"""Transformers-based LLM punctuator implementations."""

import logging
import math
import os

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

from llm_punctuator.logits_processor import CustomLogitsProcessor
from llm_punctuator.schema import Message, Role

from .prompt import EN_SYSTEM_PROMPT, ZH_SYSTEM_PROMPT
from .schema import EN_PUNCTUATIONS, ZH_PUNCTUATIONS

# Disable tokenizers parallelism to avoid fork warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)


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
    ) -> str:
        """Add punctuation to text using the LLM.

        Args:
            text: Input text without punctuation.
            punctuations: String of allowed punctuation characters. If None, uses default for language.
            language: Language code ("zh" for Chinese, "en" for English). Default: "zh".
            system_prompt: Custom system prompt. If None, uses default for language.
            chunk_size: Number of tokens per chunk for processing.

        Returns:
            Text with punctuation added.

        Raises:
            ValueError: If the specified language is not supported.
        """
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

        text_tokens = self.encode_text(text)

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

            logits_processor = CustomLogitsProcessor(
                chunk + [self.tokenizer.eos_token_id],
                punctuation_tokens,
                has_prev_input=has_prev_input,
            )

            logits_processor_list = LogitsProcessorList([logits_processor])

            generated_tokens = self.greedy_search(
                input_ids, attention_mask, logits_processor_list, max_length
            )

            generated_tokens = generated_tokens.tolist()

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
