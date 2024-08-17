import abc
import math
import torch
import logging
from typing import List, Optional
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LogitsProcessorList
from src.items import Message, Role
from src.logits_processor import (
    CustomLogitsProcessor, BeamSearchCustomLogitsProcessor
)
from .base import LLMPunctuator
from .prompt import ZH_SYSTEM_PROMPT

class TransformerAutoPunctuator():
    @classmethod
    def from_pretrained(cls, model_name_or_path:str, language:str='zh'):
        Punctuator = PATH_TO_TRANSFORMERS_PUNCTUATOR[model_name_or_path]
        return Punctuator(model_name_or_path, language)

class TransformersLLMPunctuator(LLMPunctuator):
    @abc.abstractmethod
    def apply_chat_template(self, messages:List[Message]) -> str:
        pass
 
    def __init__(self, model_name_or_path:str, language:str) -> None:
        if language == 'zh':
            self.system_prompt = ZH_SYSTEM_PROMPT
        else:
            raise ValueError(f'Language {language} is not supported.')

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path) 
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                          device_map='auto',
                                                          torch_dtype='auto')
        self.device= self.model.device
   
    @torch.no_grad()
    def add_punctuation(self,
                        text:str,
                        punctuations:str,
                        system_prompt:Optional[str]=None,
                        chunk_size:int=200,
                        num_beams:int=1) -> str:
        if system_prompt is None:
            system_prompt = self.system_prompt

        text_tokens = self.encode_text(text)
        punctuation_tokens = self.encode_text(punctuations)
        logging.info(f'Punctuation tokens: {punctuation_tokens}')

        chunk_nums = math.ceil(len(text_tokens) / chunk_size)
        chunks = [text_tokens[i*chunk_size:(i+1)*chunk_size] 
                    for i in range(chunk_nums)]
        prev_decode_tokens = []
        prev_chunk = []
        result_tokens = []
        for chunk_idx, chunk in enumerate(tqdm(chunks)):
            chunk_text = self.decode(prev_chunk + chunk)
            assistant_prefix = self.decode(prev_decode_tokens)

            messages = [
                Message(role=Role.System,
                        content=system_prompt),
                Message(role=Role.User,
                        content=chunk_text),
                Message(role=Role.Assistant,
                        content=assistant_prefix)
            ]
            
            input_prompt = self.apply_chat_template(messages)
            print(input_prompt)
            inputs = self.tokenizer(input_prompt, return_tensors='pt')
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)
            
            has_prev_input = chunk_idx != 0
            max_length = input_ids.shape[1] + int(len(chunk)*1.2)
            if num_beams == 1:
                logits_processor = CustomLogitsProcessor(
                                    chunk+[self.tokenizer.eos_token_id],
                                    punctuation_tokens,
                                    has_prev_input=has_prev_input)
                
                logits_processor_list = LogitsProcessorList([logits_processor])

                generated_tokens = self.greedy_search(input_ids,        
                                                      attention_mask,
                                                      logits_processor_list,
                                                      max_length)
            else:
                logits_processor = BeamSearchCustomLogitsProcessor(
                                    chunk+[self.tokenizer.eos_token_id],
                                    punctuation_tokens,
                                    has_prev_input=has_prev_input,
                                    num_beams=num_beams)
                
                logits_processor_list = LogitsProcessorList([logits_processor])
                generated_tokens = self.beam_search(input_ids,        
                                                    attention_mask,
                                                    logits_processor_list,
                                                    max_length,
                                                    num_beams)
            generated_tokens = generated_tokens.squeeze(0).tolist()
            generated_tokens = self.remove_punctuation(
                                        generated_tokens,
                                        punctuation_tokens,
                                        chunk_idx == chunk_nums - 1)
            
            generated_result = self.decode(generated_tokens)
            logging.info(f'chunk result: {generated_result}')
            prev_decode_tokens = generated_tokens
            prev_chunk = chunk
            result_tokens.extend(generated_tokens)

        result = self.decode(result_tokens)
        return result 
    
    def encode_text(self, text:str) -> List[int]: 
        text_tokens = self.tokenizer.encode(text)
        if text_tokens[0] == self.tokenizer.bos_token_id:
            text_tokens = text_tokens[1:]
        if text_tokens[-1] == self.tokenizer.eos_token_id:
            text_tokens = text_tokens[:-1]
        return text_tokens

    def decode(self, tokens:List[int], skip_special_tokens:bool=True) -> str:
        return self.tokenizer.decode(tokens, 
                                     skip_special_tokens=skip_special_tokens)

    def remove_punctuation(self,
                           generated_tokens:List[int],
                           punctuation_tokens:List[int],
                           is_last_chunk:bool) -> str: 
        # remove too many punctuations in the end
        if is_last_chunk:
            # keep at most 1 punctuation in the end of the last chunk
            while generated_tokens[-1] in punctuation_tokens and generated_tokens[-2] in punctuation_tokens:
                generated_tokens = generated_tokens[:-1]
        else: 
            while generated_tokens[-1] in punctuation_tokens:
                generated_tokens = generated_tokens[:-1]
        return generated_tokens

    def greedy_search(self,
                      input_ids:torch.Tensor,
                      attention_mask:torch.Tensor,
                      logits_processor_list:LogitsProcessorList,
                      max_length:int) -> torch.Tensor:
        output = self.model.generate(input_ids,
                                     attention_mask=attention_mask,
                                     logits_processor=logits_processor_list,
                                     max_length=max_length,
                                     do_sample=False,
                                     temperature=0,
                                     top_k=1,
                                     top_p=1,)
        generated_tokens = output[0][input_ids.shape[1]:]
        if generated_tokens[-1] == self.tokenizer.eos_token_id:
            generated_tokens = generated_tokens[:-1]

        return generated_tokens

    def beam_search(self,
                    input_ids:torch.Tensor,
                    attention_mask:torch.Tensor,
                    logits_processor_list:LogitsProcessorList,
                    max_length:int,
                    num_beams:int) -> torch.Tensor:
        output = self.model.generate(input_ids,
                                     attention_mask=attention_mask,
                                     logits_processor=logits_processor_list,
                                     max_length=max_length,
                                     num_beams=num_beams,
                                     do_sample=False)
        
        generated_tokens = output[0][input_ids.shape[1]:]
        if generated_tokens[-1] == self.tokenizer.eos_token_id:
            generated_tokens = generated_tokens[:-1]

        return generated_tokens

class Gemma2Punctuator(TransformersLLMPunctuator):
    def apply_chat_template(self, messages: List[Message]) -> str:
        # This is Gemma2 chat template
        messages = [Message(role=Role.User, content=messages[0].content + "以下是你要標的文章：\n" + messages[1].content)] + messages[2:]
        ret = "<|begin_of_text|>"
        for m in messages:
            ret += "<|start_header_id|>" + m.role + "<|end_header_id|>\n\n"
            ret += m.content
            if m.role != Role.Assistant:
                ret += "<|eot_id|>"
        return ret

class Llama3Punctuator(TransformersLLMPunctuator):
    def apply_chat_template(self, messages: List[Message]) -> str:
        # This is Llama3 and llama3.1 chat template
        ret = "<|begin_of_text|>"
        for m in messages:
            ret += "<|start_header_id|>" + m.role + "<|end_header_id|>\n\n"
            ret += m.content
            if m.role != Role.Assistant:
                ret += "<|eot_id|>"
        return ret

class Qwen2Punctuator(TransformersLLMPunctuator):
    def apply_chat_template(self, messages: List[Message]) -> str:
        # this is Qwen2 chat template
        ret = ''
        for m in messages:
            ret += "<|im_start|>" + m.role + "\n"
            ret += m.content
            if m.role != Role.Assistant:
                ret += "<|im_end|>\n"
        return ret

class YiPunctuator(LLMPunctuator):
    def apply_chat_template(self, messages: List[Message]) -> str:
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
    "google/gemma-2-2b-it": Gemma2Punctuator
}