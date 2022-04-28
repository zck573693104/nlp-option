import time
from collections import defaultdict

from paddlenlp.datasets import load_dataset
from paddlenlp import Taskflow

seg_fast = Taskflow("word_segmentation", mode="fast")

# 加载ChnSentiCorp数据集
train_ds, dev_ds = load_dataset("chnsenticorp", splits=["train", "dev"])
texts = []
for data in train_ds:
    texts.append(data["text"])
for data in dev_ds:
    texts.append(data["text"])
inputs_length = len(texts)

print("1. 句子数量：", inputs_length)

tic_seg = time.time()

# 快速分词
results = seg_fast(texts)

time_diff = time.time() - tic_seg

print("2. 平均速率：%.2f句/s" % (inputs_length/time_diff))

# 词频统计
word_counts = defaultdict(int)
for result in results:
    for word in result:
        word_counts[word] += 1

# 打印频次最高的前20个单词及其对应词频
print("3. Top 20 Words：", sorted(word_counts.items(), key=lambda d: d[1], reverse=True)[:20])

# 使用BiLSTM作为编码器，速度最快
ddp = Taskflow("dependency_parsing")

print(ddp("2月8日谷爱凌夺得北京冬奥会第三金"))
senta = Taskflow("sentiment_analysis")

print(senta("这个产品用起来真的很流畅，我非常喜欢"))


senta = Taskflow("sentiment_analysis", model="skep_ernie_1.0_large_ch")

print(senta("作为老的四星酒店，房间依然很整洁，相当不错。机场接机服务很好，可以在车上办理入住手续，节省时间。"))

# 编码器部分将BiLSTM替换为ERNIE，模型准确率更高！
ddp = Taskflow("dependency_parsing", model="ddparser-ernie-1.0")

print(ddp("2月8日谷爱凌夺得北京冬奥会第三金"))


senta = Taskflow("sentiment_analysis", model="skep_ernie_1.0_large_ch")

print(senta("作为老的四星酒店，房间依然很整洁，相当不错。机场接机服务很好，可以在车上办理入住手续，节省时间。"))


similarity = Taskflow("text_similarity")
print(similarity([["春天适合种什么花？", "春天适合种什么菜？"], ["小蝌蚪找妈妈怎么样", "小蝌蚪找妈妈是谁画的"]]))