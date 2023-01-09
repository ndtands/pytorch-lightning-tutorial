import os
from pathlib import Path

# Create path
BASE_DIR = Path(os.getcwd())
PATH_DATASET = Path(BASE_DIR, "artifact/data") 
PATH_CHECKPOINT = Path(BASE_DIR,"artifact/weight")
BEST_CHECKPOINT = '/media/Z/TanND22/JD_EXTRACT/INFERENCE/model_1500JD/epoch=20--val_overall_f1=0.85.ckpt'
TAGS = [
                "O",
                "B-SKILL_LEVEL", "I-SKILL_LEVEL",
                "B-SKILL_TECH", "I-SKILL_TECH",
                "B-SOFT_SKILL", "I-SOFT_SKILL",
                "B-JOB_TITLE", "I-JOB_TITLE",
                "B-DEGREE", "I-DEGREE",
                "B-MAJOR", "I-MAJOR",
        ]