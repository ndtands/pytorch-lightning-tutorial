import os
from decouple import config
from dotenv import load_dotenv
import torch
import torch.nn as nn

load_dotenv()
DATASET_PATH = config('PATH_DATASETS')
PATH_CONVNETS = config('PATH_CONVNETS')
os.makedirs(PATH_CONVNETS, exist_ok=True)
device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")

act_fn_by_name = {"tanh": nn.Tanh, "relu": nn.ReLU, "leakyrelu": nn.LeakyReLU, "gelu": nn.GELU}