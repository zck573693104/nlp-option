import os
import argparse
from functools import partial
import paddle
import paddle.nn.functional as F
from paddlenlp.metrics import AccuracyAndF1
from paddlenlp.datasets import load_dataset
from paddlenlp.data import Pad, Stack, Tuple
from paddlenlp.transformers import SkepTokenizer, SkepModel, LinearDecayWithWarmup
from utils import set_seed, decoding, is_aspect_first, concate_aspect_and_opinion, format_print
import data_ext, data_cls

train_path = "../data/data121242/train_cls.txt"
dev_path = "../data/data121242/dev_cls.txt"
test_path = "../data/data121242/test_cls.txt"
label_path = "../data/data121242/label_cls.dict"

# load and process data
label2id, id2label = data_cls.load_dict(label_path)
train_ds = load_dataset(data_cls.read, data_path=train_path, lazy=False)
dev_ds =  load_dataset(data_cls.read, data_path=dev_path, lazy=False)
test_ds =  load_dataset(data_cls.read, data_path=test_path, lazy=False)

# print examples
for example in train_ds[:2]:
    print(example)


model_name = "skep_ernie_1.0_large_ch"
batch_size = 8
max_seq_len = 512

tokenizer = SkepTokenizer.from_pretrained(model_name)
trans_func = partial(data_cls.convert_example_to_feature, tokenizer=tokenizer, label2id=label2id, max_seq_len=max_seq_len)
train_ds = train_ds.map(trans_func, lazy=False)
dev_ds = dev_ds.map(trans_func, lazy=False)
test_ds = test_ds.map(trans_func, lazy=False)

# print examples
# print examples
for example in train_ds[:2]:
    print("input_ids: ", example[0])
    print("token_type_ids: ", example[1])
    print("seq_len: ", example[2])
    print("label: ", example[3])
    print()

batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),
        Stack(dtype="int64"),
        Stack(dtype="int64")
    ): fn(samples)

train_batch_sampler = paddle.io.BatchSampler(train_ds, batch_size=batch_size, shuffle=True)
dev_batch_sampler = paddle.io.BatchSampler(dev_ds, batch_size=batch_size, shuffle=False)
test_batch_sampler = paddle.io.BatchSampler(test_ds, batch_size=batch_size, shuffle=False)

train_loader = paddle.io.DataLoader(train_ds, batch_sampler=train_batch_sampler, collate_fn=batchify_fn)
dev_loader = paddle.io.DataLoader(dev_ds, batch_sampler=dev_batch_sampler, collate_fn=batchify_fn)
test_loader = paddle.io.DataLoader(test_ds, batch_sampler=test_batch_sampler, collate_fn=batchify_fn)

class SkepForSequenceClassification(paddle.nn.Layer):
    def __init__(self, skep, num_classes=2, dropout=None):
        super(SkepForSequenceClassification, self).__init__()
        self.num_classes = num_classes
        self.skep = skep
        self.dropout = paddle.nn.Dropout(dropout if dropout is not None else self.skep.config["hidden_dropout_prob"])
        self.classifier = paddle.nn.Linear(self.skep.config["hidden_size"], num_classes)

    def forward(self, input_ids, token_type_ids=None, position_ids=None, attention_mask=None):
        _, pooled_output = self.skep(input_ids, token_type_ids=token_type_ids, position_ids=position_ids, attention_mask=attention_mask)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

    # model hyperparameter  setting


num_epoch = 3
learning_rate = 3e-5
weight_decay = 0.01
warmup_proportion = 0.1
max_grad_norm = 1.0
log_step = 20
eval_step = 100
seed = 1000
checkpoint = "./checkpoint/"

set_seed(seed)
use_gpu = True if paddle.get_device().startswith("gpu") else False
if use_gpu:
    paddle.set_device("gpu:0")
if not os.path.exists(checkpoint):
    os.mkdir(checkpoint)

skep = SkepModel.from_pretrained(model_name)
model = SkepForSequenceClassification(skep, num_classes=len(label2id))

num_training_steps = len(train_loader) * num_epoch
lr_scheduler = LinearDecayWithWarmup(learning_rate=learning_rate, total_steps=num_training_steps,
                                     warmup=warmup_proportion)
decay_params = [p.name for n, p in model.named_parameters() if not any(nd in n for nd in ["bias", "norm"])]
grad_clip = paddle.nn.ClipGradByGlobalNorm(max_grad_norm)
optimizer = paddle.optimizer.AdamW(learning_rate=lr_scheduler, parameters=model.parameters(), weight_decay=weight_decay,
                                   apply_decay_param_fun=lambda x: x in decay_params, grad_clip=grad_clip)

metric = AccuracyAndF1()

def evaluate(model, data_loader, metric):

    model.eval()
    metric.reset()
    for batch_data in data_loader:
        input_ids, token_type_ids, _, labels = batch_data
        logits = model(input_ids, token_type_ids=token_type_ids)
        correct = metric.compute(logits, labels)
        metric.update(correct)

    accuracy, precision, recall, f1, _ = metric.accumulate()

    return accuracy, precision, recall, f1

def train():
    # start to train model
    global_step, best_f1 = 1, 0.
    model.train()
    for epoch in range(1, num_epoch+1):
        for batch_data in train_loader():
            input_ids, token_type_ids, _, labels = batch_data
            # logits: batch_size, seql_len, num_tags
            logits = model(input_ids, token_type_ids=token_type_ids)
            loss = F.cross_entropy(logits, labels)

            loss.backward()
            lr_scheduler.step()
            optimizer.step()
            optimizer.clear_grad()

            if global_step > 0 and global_step % log_step == 0:
                print(f"epoch: {epoch} - global_step: {global_step}/{num_training_steps} - loss:{loss.numpy().item():.6f}")
            if (global_step > 0 and global_step % eval_step == 0) or global_step == num_training_steps:
                accuracy, precision, recall, f1  = evaluate(model, dev_loader,  metric)
                model.train()
                if f1 > best_f1:
                    print(f"best F1 performence has been updated: {best_f1:.5f} --> {f1:.5f}")
                    best_f1 = f1
                    paddle.save(model.state_dict(), f"{checkpoint}/best_cls.pdparams")
                print(f'evalution result: accuracy:{accuracy:.5f} precision: {precision:.5f}, recall: {recall:.5f},  F1: {f1:.5f}')

            global_step += 1

    paddle.save(model.state_dict(), f"{checkpoint}/final_cls.pdparams")


def model_evaluate():
    global skep, model
    # load model
    model_path = "./checkpoint/best_cls.pdparams"
    loaded_state_dict = paddle.load(model_path)
    skep = SkepModel.from_pretrained(model_name)
    model = SkepForSequenceClassification(skep, num_classes=len(label2id))
    model.load_dict(loaded_state_dict)
    accuracy, precision, recall, f1 = evaluate(model, test_loader, metric)
    print(f'evalution result: accuracy:{accuracy:.5f} precision: {precision:.5f}, recall: {recall:.5f},  F1: {f1:.5f}')


# 属性情感分类
if __name__ == '__main__':
    train()
    model_evaluate()