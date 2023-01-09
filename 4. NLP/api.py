from utils.preprocess import preprocess_text
from utils.postprocess import concat_tag
from model import NERModelModule
from pydantic import BaseModel
from transformers import AutoTokenizer
from configs import TAGS, BEST_CHECKPOINT
import warnings
import torch
import os
import numpy as np
from fastapi import FastAPI
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


api = FastAPI(title="NER", version='0.1.0')
class TextInput(BaseModel):
    text: str

model = NERModelModule(
    model_name_or_path='xlm-roberta-base',
    num_labels=len(TAGS),
    tags_list=TAGS
).load_from_checkpoint(BEST_CHECKPOINT)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base', use_fast=True)

@api.post("/api/predict")
def predict(text_input: TextInput):
    text = text_input.text
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
    results = model(**encoding)['logits']
    logit = results.detach().cpu().numpy()
    prediction = np.argmax(logit, axis=-1).squeeze()
    tag_prediction = []

    pre_word_index = None
    for i in range(len(prediction)):
        origin_index = word_ids[i]
        id_tag = prediction[i]
        tag = TAGS[id_tag]

        if origin_index not in [pre_word_index, None]:
            tag_prediction.append((words_list[origin_index], tag))
            pre_word_index = origin_index

    words_list, entities_list = concat_tag(iob_format=tag_prediction)

    return {'words': words_list, 'tags': entities_list}