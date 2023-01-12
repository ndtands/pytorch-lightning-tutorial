
import torch
from model import NERModelModule
from transformers import AutoTokenizer
import typing as t
import numpy as np
import re
from utils.preprocess import preprocess_text
from utils.postprocess import concat_tag
import time

def inference(model: NERModelModule, 
        tokenizer: AutoTokenizer,
        text: str, 
        tags: t.List, 
        device: str) -> t.List:
    normed_text = preprocess_text(text)
    words_list = normed_text.split(' ')
    # tokenize text
    tokenized_text = tokenizer.encode_plus(
        normed_text,
        add_special_tokens=True,
        max_length=256,
        padding=False,
        truncation=True,
        return_attention_mask=True,
        return_tensors='np',
    )
    encoding = tokenized_text.data
    if device.type == 'cpu':
        encoding['input_ids'] = torch.LongTensor(encoding['input_ids'], device=device)
        encoding['attention_mask'] = torch.LongTensor(encoding['attention_mask'], device=device)
    elif device.type == 'cuda':
        encoding['input_ids'] = torch.cuda.LongTensor(encoding['input_ids'], device=device)
        encoding['attention_mask'] = torch.cuda.LongTensor(encoding['attention_mask'], device=device)

    word_ids = tokenized_text.word_ids()
    dropout = torch.nn.Dropout()
    dropout.train()
    start = time.time()
    results = dropout(model(**encoding)['logits'])
    print(time.time() - start)
    logit = results.detach().cpu().numpy()
    prediction = np.argmax(logit, axis=-1).squeeze()
    tag_prediction = []

    pre_word_index = None
    for i in range(len(prediction)):
        origin_index = word_ids[i]
        id_tag = prediction[i]
        tag = tags[id_tag]

        if origin_index not in [pre_word_index, None]:
            tag_prediction.append((words_list[origin_index], tag))
            pre_word_index = origin_index

    words_list, entities_list = concat_tag(iob_format=tag_prediction)
    return list(zip(words_list,entities_list))

def extract_level(infer: t.List, json_out: t.Dict)-> t.Dict:
    level = None
    for word, tag in infer:
        if tag == 'SKILL_LEVEL':
            level = word
        if tag == 'SKILL_TECH':
            if level is None:
                json_out['output_ner'][tag].append(("NONE",word))
            else:
                json_out['output_ner'][tag].append((level,word))
        if tag == 'SOFT_SKILL':
            json_out['output_ner'][tag].append(word)
        if tag == 'JOB_TITLE':
            json_out['job_title'].append(word)
        if tag == 'DEGREE':
            json_out['output_ner'][tag].append(word)
        if tag == 'MAJOR':
            json_out['output_ner'][tag].append(word)

def get_idx(value: t.List) -> int:
    arr = ['year','good experience', 'experience', 'good knowledge', 'knowledge', 'understand']
    for idx,level in enumerate(arr):
        if level in value:
            return idx
    if value == 'none':
        return 1000
    else:
        return 100 

def sub_condition(value_1: str, value_2: str) -> str:
    idx_value_1 = get_idx(value_1.lower())
    idx_value_2 = get_idx(value_2.lower())
    if idx_value_1 == 0 and idx_value_2 == 0:
        if int(re.findall(r'\d+',value_1)[0]) < int(re.findall(r'\d+',value_2)[0]):
            return True
    if idx_value_2 < idx_value_1:
        return True
    return False

def sort_skill(temp: t.List) -> t.List:
    for index in range(1, len(temp)):
        curr_value = temp[index]
        position = index
        while position > 0 and sub_condition(temp[position-1][0],curr_value[0]) :
            temp[position]=temp[position-1]
            position = position-1
        temp[position] = curr_value
    return temp

