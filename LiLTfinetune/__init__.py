from collections import OrderedDict
import os
import types

from transformers import CONFIG_MAPPING, MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING, MODEL_NAMES_MAPPING, TOKENIZER_MAPPING, AutoConfig
from transformers.convert_slow_tokenizer import SLOW_TO_FAST_CONVERTERS, BertConverter, RobertaConverter, XLMRobertaConverter
from transformers.models.auto.modeling_auto import _BaseAutoModelClass, auto_class_update

from .models.LiLTRobertaLike import (
    LiLTRobertaLikeConfig,
    LiLTRobertaLikeForRelationExtraction,
    LiLTRobertaLikeForTokenClassification,
    LiLTRobertaLikeTokenizer,
    LiLTRobertaLikeTokenizerFast,
)

#CONFIG_MAPPING.update([("liltrobertalike", LiLTRobertaLikeConfig),])
AutoConfig.register("liltrobertalike", LiLTRobertaLikeConfig)

MODEL_NAMES_MAPPING.update([("liltrobertalike", "LiLTRobertaLike"),])
TOKENIZER_MAPPING.update(
    [
        (LiLTRobertaLikeConfig, (LiLTRobertaLikeTokenizer, LiLTRobertaLikeTokenizerFast)),
    ]
)

if os.path.exists('tag.txt'):
    with open('tag.txt', 'r') as tagf:
        TAG = tagf.read().lower()
else:
  print("tag.txt not detected, defaulting to monolingual LilT")
  TAG = 'monolingual'
assert TAG == 'monolingual' or TAG == 'multilingual', 'TAG is wrong. It should be monolingual or multilingual.'


if TAG == 'monolingual':
    SLOW_TO_FAST_CONVERTERS.update({"LiLTRobertaLikeTokenizer": RobertaConverter,})
elif TAG == 'multilingual':
    SLOW_TO_FAST_CONVERTERS.update({"LiLTRobertaLikeTokenizer": XLMRobertaConverter,})

MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING.update(
    [(LiLTRobertaLikeConfig, LiLTRobertaLikeForTokenClassification),]
)

#MODEL_FOR_RELATION_EXTRACTION_MAPPING = OrderedDict(
#    [(LiLTRobertaLikeConfig, LiLTRobertaLikeForRelationExtraction),]
#)

class AutoModelForTokenClassification(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING
    _model_mapping.register(
        LiLTRobertaLikeConfig, LiLTRobertaLikeForTokenClassification
    )


AutoModelForTokenClassification = auto_class_update(AutoModelForTokenClassification, head_doc="token classification")

#cls = types.new_class("AutoModelForTokenClassification", (_BaseAutoModelClass,))
#cls._model_mapping = MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING
#cls.__name__ = "AutoModelForTokenClassification"

#AutoModelForTokenClassification = auto_class_update(cls, head_doc="token classification")

#AutoModelForTokenClassification = auto_class_factory(
#    "AutoModelForTokenClassification", MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING, head_doc="token classification"
#)

#AutoModelForRelationExtraction = auto_class_factory(
#    "AutoModelForRelationExtraction", MODEL_FOR_RELATION_EXTRACTION_MAPPING, head_doc="relation extraction"
#)
