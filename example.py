import re
import logging
logging.basicConfig(level=logging.INFO)
import argparse
from src.punctuator import PATH_TO_TRANSFORMERS_PUNCTUATOR, TransformerAutoPunctuator

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name_or_path', type=str,
                        choices=PATH_TO_TRANSFORMERS_PUNCTUATOR.keys(),
                        required=True,
                        help="Model name or path to the model")
    text_group = parser.add_mutually_exclusive_group(required=True)
    text_group.add_argument('--file', type=str, help="Path to the file")
    text_group.add_argument('--text', type=str, help="Text to punctuate")
    parser.add_argument('-c', '--chunk_size', type=int,
                        default=50,
                        help="Chunk size for processing. Default: 50")
    parser.add_argument('--num_beams', type=int, default=1,
                        help="Number of beams for beam search")
    parser.add_argument('-l', '--language', type=str,
                        choices=['zh'],
                        default='zh')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()

    punctuator = TransformerAutoPunctuator.from_pretrained(args.model_name_or_path, args.language) 
    if args.language == 'zh':
        punctuations = "，。？！、；"

    if args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        text = args.text
    logging.info(f'Original text: {text}')

    # remove space between chinese characters
    clean_text = re.sub(r'(?<=[\u4e00-\u9fa5])\s+(?=[\u4e00-\u9fa5])', 
                        '', 
                        text)
    
    logging.info(f'Clean text: {clean_text}')
    result = punctuator.add_punctuation(clean_text,
                                        punctuations, 
                                        chunk_size=args.chunk_size,
                                        num_beams=args.num_beams)

    print(result)