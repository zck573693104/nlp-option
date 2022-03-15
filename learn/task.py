from paddlenlp import Taskflow

seg = Taskflow("word_segmentation", device='cpu')
print(seg("第十四届全运会在西安举办"))
# 情感分析
senta = Taskflow("sentiment_analysis")
print(senta("这个产品用起来真的很流畅，我非常喜欢"))
