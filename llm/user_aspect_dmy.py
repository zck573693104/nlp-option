# 达摩院模型
import json

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import os




class AOE:
    def __init__(self, option, aspect):
        self.option = option
        self.aspect = aspect


semantic_cls = pipeline(Tasks.siamese_uie, 'damo/nlp_structbert_siamese-aoe_chinese-base', model_revision='v1.0')

# 属性情感抽取 {属性词: {情感词: None}}
res = semantic_cls(
    input='车上的高德地图是什么版本，能更新吗，太简单了，一不能选语音，二经常变路就容易出错，走的不是最佳路径，重新导航后又正常！定位更新太慢。用户小管家',
    schema={
        '属性词': {
            '情感词': None,
        }
    }
)
aoes = []
# res = json.dumps(res, ensure_ascii=False)
for items in res['output']:
    temps = []
    for item in items:
        type = item['type']
        span = item['span']
        temps.append(span)
        if len(temps) == 2:
            aoes.append(AOE(temps[0], temps[1]).__dict__)
            temps.clear()
print(json.dumps(aoes,ensure_ascii=False,indent=2))
