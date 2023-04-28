import argparse

import numpy as np
from elasticsearch import Elasticsearch
from elasticsearch.helpers import streaming_bulk, parallel_bulk, bulk, scan

import json

from langchain.embeddings import HuggingFaceEmbeddings


parser = argparse.ArgumentParser()
parser.add_argument("--index", default="pdf_index", type=str, help="The index name of the ANN search engine")
parser.add_argument("--data_path", default="./txt/demo_ocr_res.txt", type=str,
                    help="The index name of the ANN search engine")

args = parser.parse_args()
es = Elasticsearch([
    {"host": "10.10.8.72", "port": 9200},

])


def init_mapping(index):
    es.indices.delete(index=index, ignore=[404])
    es.indices.create(index=index)
    # 配置 Elasticsearch 向量映射
    mapping = {
        "properties": {
            "embedding": {"type": "dense_vector", "dims": 768},
            # "question": {"type": "keyword"},
            # "answer": {"type": "keyword"},
        }
    }
    es.indices.put_mapping(index=index, body=mapping)


def insert(index, data_path):
    with open(data_path, 'r') as f:
        items = f.readlines()

    # embeddings = HuggingFaceEmbeddings(model_name='model/simcse-chinese-roberta-wwm-ext')
    # embeddings = embeddings.embed_documents(items)
    from text2vec import SentenceModel


    model = SentenceModel('text2vec-base-chinese')
    embeddings = model.encode(items)
    print(embeddings)
    # 构造问答对 JSON 对象
    docs = []
    i = 0
    for item in items:
        item = item.replace('\n','')
        doc = {
            '_index': index,
            '_id': i,
            'item': item,
            'embedding': embeddings[i]
        }
        i += 1
        docs.append(doc)
    # 将问答对数据批量插入 Elasticsearch 中
    for success, info in parallel_bulk(es, docs, index=index):
        if not success:
            print('A document failed:', info)


def query(index, embedding):
    # Elasticsearch 向量搜索
    script_query = {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.query_vector, 'embedding')+1.0",
                "params": {
                    "query_vector": embedding}
            }
        }
    }
    result = es.search(index=index, body={"size": 10, "query": script_query})
    rs = []
    for hit in result['hits']['hits']:
        score = hit['_score']
        id = hit['_id']
        item = hit['_source']['item']
        rs.append({'score': score, 'id': id, 'content': item,'title':'todo'})
    print(json.dumps(rs, indent=2, ensure_ascii=False))
    return rs


def query_id(id):
    result = es.get(
        index='pdf_index',
        id=id,
        _source_includes=['item']
    )
    print('------------es-------------------')
    _source = result['_source']
    print(_source['item'])


if __name__ == "__main__":
    init_mapping(args.index)
    insert(args.index, args.data_path)
    # text = '智能钥匙解锁方法,要如何解决这个问题'
    # embedding = HuggingFaceEmbeddings(model_name='model/simcse-chinese-roberta-wwm-ext').embed_query(text)
    # query(args.index, embedding)
