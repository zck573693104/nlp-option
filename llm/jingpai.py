from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

model_id = "damo/nlp_corom_sentence-embedding_chinese-base"
pipeline_se = pipeline(Tasks.sentence_embedding,
                       model=model_id)


# 当输入包含“soure_sentence”与“sentences_to_compare”时，会输出source_sentence中首个句子与sentences_to_compare中每个句子的向量表示，以及source_sentence中首个句子与sentences_to_compare中每个句子的相似度。
def score(source_sentence, sentences_to_compare):
    print(source_sentence)
    print(sentences_to_compare)
    inputs = {
        "source_sentence": source_sentence,
        "sentences_to_compare": sentences_to_compare
    }
    return pipeline_se(input=inputs)



