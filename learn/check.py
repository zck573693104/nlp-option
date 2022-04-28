import os
import argparse
from functools import partial
import paddle
import paddle.nn.functional as F
from paddlenlp.metrics import ChunkEvaluator
from paddlenlp.datasets import load_dataset
from paddlenlp.data import Pad, Stack, Tuple
from paddlenlp.transformers import SkepTokenizer, SkepModel, LinearDecayWithWarmup, SkepForTokenClassification

import data_cls
import data_ext
from utils import set_seed
# 数据加载
train_path = "../data/data121190/train_ext.txt"
dev_path = "../data/data121190/dev_ext.txt"
test_path = "../data/data121190/test_ext.txt"
label_path = "../data/data121190/label_ext.dict"

model_name = "skep_ernie_1.0_large_ch"
