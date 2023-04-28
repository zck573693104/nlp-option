# 1.向量插入到ES里面
# 2.查询转向量
# 3.通过es相似度查找
# 4.jingpai查找
def search_txt(param):
    from text2vec import SentenceModel

    model = SentenceModel('text2vec-base-chinese')
    embeddings = model.encode([param])
    response_d = es.query_txt('pdf_index',embeddings[0])
    content = [i['content'] for i in response_d]
    return content