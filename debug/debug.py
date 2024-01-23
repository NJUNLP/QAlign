from datasets import load_dataset
import os
from utils import load_datasets, Dataset
import os
import json
import datasets
import pandas as pd
import transformers

# d = load_datasets("/cpfs01/user/zhuwenhao/open-source/QAlign/code/train/data/metamath_all", split="train")
d = load_datasets("/cpfs01/user/zhuwenhao/open-source/QAlign/code/data/gsmtrans_gsm8kinstruct_answer_all-en", split="train")
_d = list(d)
print(len(_d))
print(_d[:5])