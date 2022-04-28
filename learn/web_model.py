'''
Created on 2022 年 03 月 20 日
部署在 Linux,直接来访问 paddlenlp 训练出来的推理模型.
多标签.
@author: SHGUAN
'''

from get_ip import get_host_ip

from sanic import Sanic, response

import os

import time


# import paddle.inference as paddle_infer


class Predictor(object):

    def __init__(self, model_dir, device='cpu', max_seq_length=128):
        import paddle
        import paddlenlp as ppnlp
        # from paddlenlp.transformers import ErnieTokenizer
        self.max_seq_length = max_seq_length
        model_file = model_dir + "/inference.pdmodel"
        params_file = model_dir + "/inference.pdiparams"
        if not os.path.exists(model_file):
            raise ValueError("not find model file path {}".format(model_file))
        if not os.path.exists(params_file):
            raise ValueError("not find params file path {}".format(model_file))

        config = paddle.inference.Config(model_file, params_file)

        if device == "gpu":
            # set GPU configs accordingly
            config.enable_use_gpu(100, 0)  # GPU 显存 100M,Device_ID 为 0

        elif device == "cpu":
            # set CPU configs accordingly,
            # such as enable_mkldnn, set_cpu_math_library_num_threads
            config.disable_gpu()
            config.set_mkldnn_cache_capacity(10)
            config.enable_mkldnn()
            config.set_cpu_math_library_num_threads(10)

        elif device == "xpu":
            # set XPU configs accordingly
            config.enable_xpu(100)

        config.switch_use_feed_fetch_ops(False)

        # 创建预测器
        self.predictor = paddle.inference.create_predictor(config)

        self.input_handles = [self.predictor.get_input_handle(name) for name in self.predictor.get_input_names()]
        self.output_handle = self.predictor.get_output_handle(self.predictor.get_output_names()[0])

        self.tokenizer = ppnlp.transformers.ErnieTokenizer.from_pretrained("ernie-1.0")

        # self.predictor_poool = paddle_infer.PredictorPool(config,4)
        '''
        https://paddleinference.paddlepaddle.org.cn/api_reference/python_api_doc/PredictorPool.html

        预测对象线程池
        self.pred_pool = paddle_infer.PredictorPool(config,4)  # predictor 对象数量

        predictor1 = self.pred_pool.retrive(2)
        '''

    def predict(self, data, batch_size=1, threshold=0.5):
        """
        Predicts the data labels.

        Args:
            data (obj:`List(Example)`): The processed data whose each element is a Example (numedtuple) object.
                A Example object contains `text`(word_ids) and `se_len`(sequence length).
            tokenizer(obj:`PretrainedTokenizer`): This tokenizer inherits from :class:`~paddlenlp.transformers.PretrainedTokenizer`
                which contains most of the methods. Users should refer to the superclass for more information regarding methods.
            batch_size(obj:`int`, defaults to 1): The number of batch.
            threshold(obj:`int`, defaults to 0.5): The threshold for converting probabilities to labels.

        Returns:
            results(obj:`dict`): All the predictions labels.
        """
        from paddlenlp.data import Tuple, Pad
        import paddle.nn.functional as F
        import paddle
        examples = []
        for text in data:
            example = {"text": text}
            input_ids, segment_ids = self.convert_example(example, self.tokenizer, max_seq_length=self.max_seq_length)
            examples.append((input_ids, segment_ids))

        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=self.tokenizer.pad_token_id),  # input
            Pad(axis=0, pad_val=self.tokenizer.pad_token_id),  # segment
        ): fn(samples)

        # Seperates data into some batches.
        batches = [
            examples[idx:idx + batch_size]
            for idx in range(0, len(examples), batch_size)
        ]

        results = []
        for batch in batches:
            input_ids, segment_ids = batchify_fn(batch)
            # 输入运行参数
            self.input_handles[0].copy_from_cpu(input_ids)
            self.input_handles[1].copy_from_cpu(segment_ids)
            self.predictor.run()  # 执行预测
            # 获取输出的 Tensor
            logits = paddle.to_tensor(self.output_handle.copy_to_cpu())

            probs = F.sigmoid(logits)
            preds = (probs.numpy() > threshold).astype(int)
            results.extend(preds)
        return results

    def convert_example(self, example, tokenizer, max_seq_length=128):
        encoded_inputs = tokenizer(text=example["text"], max_seq_len=max_seq_length)
        input_ids = encoded_inputs["input_ids"]
        token_type_ids = encoded_inputs["token_type_ids"]

        return input_ids, token_type_ids


predictor_wenhao = Predictor('/home/models/output_wenhao_zhengce_banshi', 'gpu', 128)

# tokenizer = ErnieTokenizer.from_pretrained("ernie-1.0")

# tokenizer = ppnlp.transformers.ErnieTokenizer.from_pretrained("ernie-1.0")


app = Sanic(__name__)


@app.route('/', methods=['GET'])
async def hello(request):
    return response.json(
        {'message': 'Hello world'}, status=200
    )


@app.route('/wzb', methods=['POST'])
async def postWenHao(request):
    query_text = str(request.form.get("query"))
    query_text = query_text.strip()
    if query_text.lower() == 'none':
        return response.json(
            {'message': 'Please input the query text.'}, status=200
        )
    else:
        print("query:" + query_text)

    data = []
    for d in query_text.split(","):
        data.append(d)

    starttime = time.time() * 1000
    results = predictor_wenhao.predict(data, 8, 0.5)
    endtime = time.time() * 1000
    mytime = str(round(endtime - starttime, 2))
    print(results[0])
    print(mytime)
    myresult = [str(label) for label in results[0]]
    # return response.json({"query": query_text,"title":title_text,"cosine":res_v,"duration":duration},status=200)
    return response.json({'query': query_text, "label": " ".join(myresult), "label_name": "政策 文号 办事 提问",
                          "consuming time": mytime + " ms"}, status=200)


@app.route('/wzb', methods=['GET'])
async def getWenHao(request):
    query_text = str(request.args.get("query"))
    query_text = query_text.strip()
    if query_text.lower() == 'none':
        return response.json(
            {'message': 'Please input the query text.'}, status=200
        )
    else:
        print("query:" + query_text)

    data = []
    for d in query_text.split(","):
        data.append(d)

    starttime = time.time() * 1000
    results = predictor_wenhao.predict(data, 8, 0.5)
    endtime = time.time() * 1000
    mytime = str(round(endtime - starttime, 2))
    print(results[0])
    print(mytime)
    myresult = [str(label) for label in results[0]]
    # return response.json({"query": query_text,"title":title_text,"cosine":res_v,"duration":duration},status=200)
    return response.json({'query': query_text, "label": " ".join(myresult), "label_name": "政策 文号 办事 提问",
                          "consuming time": mytime + " ms"}, status=200)


'''
#pip3 install sanic
'''

if __name__ == "__main__":
    this_ip = get_host_ip()
    port = 9200

    app.run(host=this_ip, port=port, workers=2, debug=False, access_log=False)



