import argparse
import os
import re

import pandas as pd

# Aspect Term Extraction
# schema =  ["评价维度"]
# Aspect - Opinion Extraction
# schema =  [{"评价维度":["观点词"]}]
# Aspect - Sentiment Extraction
# schema =  [{"评价维度":["情感倾向[正向,负向,未提及]"]}]
# Aspect - Sentiment - Opinion Extraction


import paddle
from paddlenlp import Taskflow
from paddlenlp.transformers import SkepTokenizer, SkepForTokenClassification, SkepForSequenceClassification
from seqeval.metrics.sequence_labeling import get_entities

from data_base_utils import DataBaseUtils

schema = [{"评价维度": ["观点词", "情感倾向[正向,负向,未提及]"]}]

senta = Taskflow("sentiment_analysis", model="uie-senta-base", schema=schema)
db = DataBaseUtils()
with open('final_res.txt','r') as f:
    datas = f.readlines()
# for row in datas:
#     aspect = senta(row)
#     if '评价维度' in aspect[0]:
#         print(row[0])
#         for i in aspect[0]['评价维度']:
#             text = i['text']
#             relations = i['relations']
#             options = relations['观点词']
#             aspect = relations['情感倾向[正向,负向,未提及]']
#             k = 0
#             for j in options:
#                 probability = j['probability']
#                 if probability > 0.6:
#                     op = j['text']
#                 k+=1
def load_dict(dict_path):
    with open(dict_path, "r", encoding="utf-8") as f:
        words = [word.strip() for word in f.readlines()]
        word2id = dict(zip(words, range(len(words))))
        id2word = dict((v, k) for k, v in word2id.items())

        return word2id, id2word
def decoding(text, tag_seq):
    assert len(text) == len(
        tag_seq), f"text len: {len(text)}, tag_seq len: {len(tag_seq)}"

    puncs = list(",.?;!，。？；！")
    splits = [idx for idx in range(len(text)) if text[idx] in puncs]

    prev = 0
    sub_texts, sub_tag_seqs = [], []
    for i, split in enumerate(splits):
        sub_tag_seqs.append(tag_seq[prev:split])
        sub_texts.append(text[prev:split])
        prev = split
    sub_tag_seqs.append(tag_seq[prev:])
    sub_texts.append((text[prev:]))

    ents_list = []
    for sub_text, sub_tag_seq in zip(sub_texts, sub_tag_seqs):
        ents = get_entities(sub_tag_seq, suffix=False)
        ents_list.append((sub_text, ents))

    aps = []
    no_a_words = []
    for sub_tag_seq, ent_list in ents_list:
        sub_aps = []
        sub_no_a_words = []
        for ent in ent_list:
            ent_name, start, end = ent
            if ent_name == "Aspect":
                aspect = sub_tag_seq[start:end + 1]
                sub_aps.append([aspect])
                if len(sub_no_a_words) > 0:
                    sub_aps[-1].extend(sub_no_a_words)
                    sub_no_a_words.clear()
            else:
                ent_name == "Opinion"
                opinion = sub_tag_seq[start:end + 1]
                if len(sub_aps) > 0:
                    sub_aps[-1].append(opinion)
                else:
                    sub_no_a_words.append(opinion)

        if sub_aps:
            aps.extend(sub_aps)
            if len(no_a_words) > 0:
                aps[-1].extend(no_a_words)
                no_a_words.clear()
        elif sub_no_a_words:
            if len(aps) > 0:
                aps[-1].extend(sub_no_a_words)
            else:
                no_a_words.extend(sub_no_a_words)

    if no_a_words:
        no_a_words.insert(0, "None")
        aps.append(no_a_words)

    return aps


def concate_aspect_and_opinion(text, aspect, opinions):
    aspect_text = ""
    for opinion in opinions:
        if text.find(aspect) <= text.find(opinion):
            aspect_text += aspect + opinion + "，"
        else:
            aspect_text += opinion + aspect + "，"
    aspect_text = aspect_text[:-1]

    return aspect_text

def predict(args, ext_model, cls_model, tokenizer, ext_id2label, cls_id2label):
    ext_model.eval()
    cls_model.eval()
    points = []
    with open('final_res.txt', 'r') as f:
        datas = f.readlines()
    for input_text in datas:
        input_text = re.sub('([^\u4e00-\u9fa5a-zA-Z0-9，。！？])', '', input_text)
        print(input_text)

        if len(input_text) > 0:
            encoded_inputs = tokenizer(
                list(input_text),
                is_split_into_words=True,
                max_seq_len=args.ext_max_seq_len)
            input_ids = paddle.to_tensor([encoded_inputs["input_ids"]])
            token_type_ids = paddle.to_tensor([encoded_inputs["token_type_ids"]])

            # extract aspect and opinion words
            logits = ext_model(input_ids, token_type_ids=token_type_ids)
            predictions = logits.argmax(axis=2).numpy()[0]
            tag_seq = [ext_id2label[idx] for idx in predictions][1:-1]

            aps = decoding(input_text[:args.ext_max_seq_len - 2], tag_seq)

            # predict sentiment for aspect with cls_model
            results = []
            for ap in aps:
                aspect = ap[0]
                opinion_words = list(set(ap[1:]))
                if len(opinion_words) == 0:
                    continue
                aspect_text = concate_aspect_and_opinion(input_text, aspect,
                                                         opinion_words)

                encoded_inputs = tokenizer(
                    aspect_text,
                    text_pair=input_text,
                    max_seq_len=args.cls_max_seq_len,
                    return_length=True)
                input_ids = paddle.to_tensor([encoded_inputs["input_ids"]])
                token_type_ids = paddle.to_tensor(
                    [encoded_inputs["token_type_ids"]])

                logits = cls_model(input_ids, token_type_ids=token_type_ids)
                prediction = logits.argmax(axis=1).numpy()[0]

                result = {
                    "aspect": aspect,
                    "opinions": opinion_words,
                    "sentiment_polarity": cls_id2label[prediction]
                }
                for word in opinion_words:
                    points.append((aspect,word, result['sentiment_polarity'], input_text))
    db.insertUserViewPoint(points)

if __name__ == "__main__":
    # yapf: disable
    parser = argparse.ArgumentParser()
    parser.add_argument("--ext_model_path", type=str, default='/home/hycan/HDD1/opt/zck/PaddleNLP/applications/sentiment_analysis/checkpoints/ext_checkpoints/best.pdparams',
                        help="The path of extraction model path that you want to load.")
    parser.add_argument("--cls_model_path", type=str, default='/home/hycan/HDD1/opt/zck/PaddleNLP/applications/sentiment_analysis/checkpoints/cls_checkpoints/best.pdparams',
                        help="The path of classification model path that you want to load.")
    parser.add_argument("--ext_label_path", type=str, default='/home/hycan/HDD1/opt/zck/PaddleNLP/applications/sentiment_analysis/data/ext_data/label.dict', help="The path of extraction label dict.")
    parser.add_argument("--cls_label_path", type=str, default='/home/hycan/HDD1/opt/zck/PaddleNLP/applications/sentiment_analysis/data/cls_data/label.dict', help="The path of classification label dict.")
    parser.add_argument("--ext_max_seq_len", type=int, default=512,
                        help="The maximum total input sequence length after tokenization for extraction model.")
    parser.add_argument("--cls_max_seq_len", type=int, default=512,
                        help="The maximum total input sequence length after tokenization for classification model.")
    args = parser.parse_args()
    # yapf: enbale

    # load dict
    model_name = "skep_ernie_1.0_large_ch"
    ext_label2id, ext_id2label = load_dict(args.ext_label_path)
    cls_label2id, cls_id2label = load_dict(args.cls_label_path)
    tokenizer = SkepTokenizer.from_pretrained(model_name)
    print("label dict loaded.")

    # load ext model
    ext_state_dict = paddle.load(args.ext_model_path)
    ext_model = SkepForTokenClassification.from_pretrained(model_name, num_classes=len(ext_label2id))
    ext_model.load_dict(ext_state_dict)
    print("extraction model loaded.")

    # load cls model
    cls_state_dict = paddle.load(args.cls_model_path)
    cls_model = SkepForSequenceClassification.from_pretrained(model_name, num_classes=len(cls_label2id))
    cls_model.load_dict(cls_state_dict)
    print("classification model loaded.")
    predict(args, ext_model, cls_model, tokenizer, ext_id2label, cls_id2label)