import os
from pathlib import Path

# Create path
BASE_DIR = Path(os.getcwd())
PATH_DATASET = Path(BASE_DIR, "artifact/data") 
PATH_CHECKPOINT = Path(BASE_DIR,"artifact/weight")
TAGS = [
        "O",
        "B-ORG", "I-ORG",
        "B-TIME", "I-TIME",
        "B-POSITION", "I-POSITION",
        "B-DEGREE", "I-DEGREE",
        "B-MAJOR", "I-MAJOR",
        "B-SCORE", "I-SCORE",
        ]