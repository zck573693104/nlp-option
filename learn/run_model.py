import os
import argparse
from functools import partial
import paddle
import paddle.nn.functional as F
from paddlenlp.metrics import ChunkEvaluator
from paddlenlp.datasets import load_dataset
from paddlenlp.data import Pad, Stack, Tuple
from paddlenlp.transformers import SkepTokenizer, SkepModel, LinearDecayWithWarmup, SkepForTokenClassification, \
    SkepForSequenceClassification

import data_cls
import data_ext
from utils import set_seed, decoding, concate_aspect_and_opinion, format_print

label_ext_path = "../data/data121190/label_ext.dict"
label_cls_path = "../data/data121242/label_cls.dict"
ext_model_path = "/opt/zck/all_model/best_ext.pdparams"
cls_model_path = "/opt/zck/all_model/best_cls.pdparams"

# load dict
model_name = "skep_ernie_1.0_large_ch"
ext_label2id, ext_id2label = data_ext.load_dict(label_ext_path)
cls_label2id, cls_id2label = data_cls.load_dict(label_cls_path)
tokenizer = SkepTokenizer.from_pretrained(model_name)
print("label dict loaded.")

# load ext model
ext_state_dict = paddle.load(ext_model_path)
ext_skep = SkepModel.from_pretrained(model_name)
ext_model = SkepForTokenClassification(ext_skep, num_classes=len(ext_label2id))
ext_model.load_dict(ext_state_dict)
print("extraction model loaded.")

# load cls model
cls_state_dict = paddle.load(cls_model_path)
cls_skep = SkepModel.from_pretrained(model_name)
cls_model = SkepForSequenceClassification(cls_skep, num_classes=len(cls_label2id))
cls_model.load_dict(cls_state_dict)
print("classification model loaded.")


def predict(input_text, ext_model, cls_model, tokenizer, ext_id2label, cls_id2label, max_seq_len=512):
    ext_model.eval()
    cls_model.eval()

    # processing input text
    encoded_inputs = tokenizer(list(input_text), is_split_into_words=True, max_seq_len=max_seq_len, )
    input_ids = paddle.to_tensor([encoded_inputs["input_ids"]])
    token_type_ids = paddle.to_tensor([encoded_inputs["token_type_ids"]])

    # extract aspect and opinion words
    logits = ext_model(input_ids, token_type_ids=token_type_ids)
    predictions = logits.argmax(axis=2).numpy()[0]
    tag_seq = [ext_id2label[idx] for idx in predictions][1:-1]
    aps = decoding(input_text, tag_seq)

    # predict sentiment for aspect with cls_model
    results = []
    for ap in aps:
        aspect = ap[0]
        opinion_words = list(set(ap[1:]))
        aspect_text = concate_aspect_and_opinion(input_text, aspect, opinion_words)

        encoded_inputs = tokenizer(aspect_text, text_pair=input_text, max_seq_len=max_seq_len, return_length=True)
        input_ids = paddle.to_tensor([encoded_inputs["input_ids"]])
        token_type_ids = paddle.to_tensor([encoded_inputs["token_type_ids"]])

        logits = cls_model(input_ids, token_type_ids=token_type_ids)
        prediction = logits.argmax(axis=1).numpy()[0]

        result = {"aspect": aspect, "opinions": opinion_words, "sentiment": cls_id2label[prediction]}
        results.append(result)

    # print results
    format_print(results)


max_seq_len = 512
input_text = "环境装修不错，也很干净，前台服务非常好"
predict(input_text, ext_model, cls_model, tokenizer, ext_id2label, cls_id2label, max_seq_len=max_seq_len)

input_text = "蛋糕味道不错，很好吃，店家很耐心，服务也很好，很棒"
predict(input_text, ext_model, cls_model, tokenizer, ext_id2label, cls_id2label, max_seq_len=max_seq_len)