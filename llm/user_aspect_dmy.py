# 达摩院模型
import json
import re

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import os

from data_base_utils import DataBaseUtils




class AOE:
    def __init__(self, aspect, option, sentiment, content):
        self.aspect = aspect
        self.option = option
        self.sentiment = sentiment
        self.content = content


semantic_cls = pipeline(Tasks.siamese_uie, 'damo/nlp_structbert_siamese-aoe_chinese-base', model_revision='v1.0')
aoes = []


def deal_user_content():
    with open('final_res.txt', 'r') as f:
        datas = f.readlines()
    i = len(datas)
    for input_text in datas:
        print(i)
        i -= 1
        input_text = re.sub('([^\u4e00-\u9fa5a-zA-Z0-9，。！？])', '', input_text)
        # 属性情感抽取 {属性词: {情感词: None}}
        res = semantic_cls(
            input=input_text,
            schema={
                '属性词': {
                    "正向": None,
                    "负向": None,
                    "中性": None
                }
            }
        )
        # res = json.dumps(res, ensure_ascii=False)
        for items in res['output']:
            # aspect, option, sentiment, content
            aoes.append((items[0]['span'], items[1]['span'], items[1]['type'], input_text))


deal_user_content()
db = DataBaseUtils()
db.insertUserViewPoint(aoes)