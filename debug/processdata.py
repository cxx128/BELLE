
from typing import Any, Callable, Dict, List
import copy
from transformers import PreTrainedTokenizer
import json
from transformers import (
    AutoTokenizer,
)

IGNORE_INDEX = -100


def sft_sample_to_ids(conversations: Dict[str, Any], tokenizer: PreTrainedTokenizer):
    input_ids = []
    labels = []
    for sentence in conversations:
        sentence_from = sentence["from"].lower()
        sentence_value = (
            "Human: \n" + sentence["value"] + "\n\nAssistant: \n"
            if sentence_from == "human"
            else sentence["value"]
        )  # https://github.com/LianjiaTech/BELLE/issues/337
        # conversation += sentence_value
        sentence_ids = tokenizer.encode(
            sentence_value, add_special_tokens=False
        )  # do not add bos_token_id
        label = (
            copy.deepcopy(sentence_ids)
            if sentence_from != "human"
            else [IGNORE_INDEX] * len(sentence_ids)
        )
        input_ids += sentence_ids
        labels += label
        # add eos at every end of assistant sentence
        if sentence_from != "human":
            input_ids += [tokenizer.eos_token_id]  # make sure eos_token_id is correct
            labels += [tokenizer.eos_token_id]
    attention_mask=[[1]*len(input_ids)]
    return input_ids, labels,attention_mask



all=[{"conversations": [{"from": "human", "value": " 生成一个描述现在天气的句子。"}, {"from": "assistant", "value": " 现在的天气晴朗，温度适中，适合外出活动。"}],"history":[]},
{"conversations": [{"from": "human", "value": " 生成一个描述现在天气的句子。"}, {"from": "assistant", "value": " 现在的天气晴朗，温度适中，适合外出活动。"}],"history":[]},
{"conversations": [{"from": "human", "value": " 生成一个描述现在天气的句子。"}, {"from": "assistant", "value": " 现在的天气晴朗，温度适中，适合外出活动。"}],"history":[]},
{"conversations": [{"from": "human", "value": " 生成一个描述现在天气的句子。"}, {"from": "assistant", "value": " 现在的天气晴朗，温度适中，适合外出活动。"}],"history":[]},
{"conversations": [{"from": "human", "value": " 生成一个描述现在天气的句子。"}, {"from": "assistant", "value": " 现在的天气晴朗，温度适中，适合外出活动。"}],"history":[]},
{"conversations": [{"from": "human", "value": " 生成一个描述现在天气的句子。"}, {"from": "assistant", "value": " 现在的天气晴朗，温度适中，适合外出活动。"}],"history":[]},
{"conversations": [{"from": "human", "value": " 生成一个描述现在天气的句子。"}, {"from": "assistant", "value": " 现在的天气晴朗，温度适中，适合外出活动。"}],"history":[]},
{"conversations": [{"from": "human", "value": " 生成一个描述现在天气的句子。"}, {"from": "assistant", "value": " 现在的天气晴朗，温度适中，适合外出活动。"}],"history":[]},
{"conversations": [{"from": "human", "value": " 生成一个描述现在天气的句子。"}, {"from": "assistant", "value": " 现在的天气晴朗，温度适中，适合外出活动。"}],"history":[]},
{"conversations": [{"from": "human", "value": " 生成一个描述现在天气的句子。"}, {"from": "assistant", "value": " 现在的天气晴朗，温度适中，适合外出活动。"}],"history":[]},
]
model_name_or_path='/mnt/afs/chenxiaoxuan/hf_home/Qwen2.5-7B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,padding_side="right",use_fast=False,trust_remote_code=True)
for conversations in all:
    input_ids,labels=sft_sample_to_ids(conversations["conversations"],tokenizer)
    print('xx')
print('xx')