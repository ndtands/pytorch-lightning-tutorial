from utils.preprocess import preprocess_JD
from inference import (
    inference,
    extract_level,
    sort_skill
)
from model import NERModelModule
from pydantic import BaseModel
from transformers import AutoTokenizer
from configs import *
import warnings
import torch
import os
from fastapi import FastAPI
warnings.filterwarnings("ignore")

api = FastAPI(title="JD Extraction", version='0.1.0')
class TextInput(BaseModel):
    text: str

model = NERModelModule(
    model_name_or_path=BASE_MODEL_NAME,
    num_labels=len(TAGS),
    tags_list=TAGS
).load_from_checkpoint(BEST_CHECKPOINT)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# Load tokenizer
tokenizer = model.tokenizer

@api.post("/jd_extraction")
def predict(text_input: TextInput):
    json_out = {
        'job_title': [],
        'output_ner': {
            'DEGREE':[],
            'MAJOR':[],
            'SOFT_SKILL':[],
            'SKILL_TECH':[],
        },
    }
    JD = text_input.text
    JD = preprocess_JD(JD)
    for line in JD:
        if line != '':
            out=inference(model, tokenizer, line, TAGS, device)
            extract_level(out, json_out)   
    json_out["output_ner"]["SKILL_TECH"] = sort_skill(json_out["output_ner"]["SKILL_TECH"])  
    return json_out