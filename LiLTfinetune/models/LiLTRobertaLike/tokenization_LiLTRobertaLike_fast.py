# coding=utf-8
from transformers import RobertaTokenizerFast, XLMRobertaTokenizerFast
from transformers.file_utils import is_sentencepiece_available
from transformers.utils import logging
import os


if is_sentencepiece_available():
    from .tokenization_LiLTRobertaLike import LiLTRobertaLikeTokenizer
else:
    LiLTRobertaLikeTokenizer = None


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.bpe.model", "tokenizer_file": "tokenizer.json"}

if os.path.exists('tag.txt'):
    with open('tag.txt', 'r') as tagf:
        TAG = tagf.read().lower()
else:
  print("tag.txt not detected, defaulting to monolingual LilT")
  TAG = 'monolingual'
assert TAG == 'monolingual' or TAG == 'multilingual', 'TAG is wrong. It should be monolingual or multilingual.'

if TAG == 'monolingual':
    class LiLTRobertaLikeTokenizerFast(RobertaTokenizerFast):

        vocab_files_names = VOCAB_FILES_NAMES
        max_model_input_sizes = {"lilt-roberta-base": 512,}
        model_input_names = ["input_ids", "attention_mask"]
        slow_tokenizer_class = LiLTRobertaLikeTokenizer

        def __init__(self, model_max_length=512, **kwargs):
            super().__init__(model_max_length=model_max_length, **kwargs)
            
elif TAG == 'multilingual':
    class LiLTRobertaLikeTokenizerFast(XLMRobertaTokenizerFast):

        vocab_files_names = VOCAB_FILES_NAMES
        max_model_input_sizes = {"lilt-infoxlm-base": 512,}
        model_input_names = ["input_ids", "attention_mask"]
        slow_tokenizer_class = LiLTRobertaLikeTokenizer

        def __init__(self, model_max_length=512, **kwargs):
            super().__init__(model_max_length=model_max_length, **kwargs)
