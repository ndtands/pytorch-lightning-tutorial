import os
from pathlib import Path

# Configs GPU ID
os.environ['CUDA_VISIBLE_DEVICES']='1'

# Create path
BASE_DIR = Path(os.getcwd())
PATH_DATASET = Path(BASE_DIR, "artifact/data") 
PATH_CHECKPOINT = Path(BASE_DIR,"artifact/weight")

# Configuration for training
TAGS = [
        "O",
        "B-SKILL_LEVEL", "I-SKILL_LEVEL",
        "B-SKILL_TECH", "I-SKILL_TECH",
        "B-SOFT_SKILL", "I-SOFT_SKILL",
        "B-JOB_TITLE", "I-JOB_TITLE",
        "B-DEGREE", "I-DEGREE",
        "B-MAJOR", "I-MAJOR",
    ]
BASE_MODEL_NAME = "xlm-roberta-base"
# Configuration for deployment
RUN_ID = "jd_extract_1673449211.8823872"
WEIGHT_NAME = "epoch=20--val_overall_f1=0.85.ckpt"
BEST_CHECKPOINT = Path(PATH_CHECKPOINT, RUN_ID, WEIGHT_NAME)

# Visualize
COLORS ={
    'SKILL_LEVEL':'#FDEE00',
    'SKILL_TECH':'#C32148',
    'DEGREE':'#FE6F5E',
    'MAJOR': '#9F8170',
    'SKILL_SOFT':'#007BA7',
    'JOB_TITLE':'#D891EF',

}
NER = list(COLORS.keys())

OPTIONS = {'ents': NER, 'colors': COLORS}