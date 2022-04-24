#! -*- coding: utf-8 -*-
# 法研杯2020 司法摘要
# 最终模型：所有步骤串起来
# 使用方式：
# from final import *
# summary = predict(text, topk=3)
# print(summary)
# 科学空间：https://kexue.fm

import os
os.system('pip install -r requirements.txt')

import numpy as np
import extract_convert as convert
import extract_vectorize as vectorize
import extract_model as extract
import seq2seq_model as seq2seq
from snippets import *

# if len(sys.argv) == 1:
#     fold = 0
# else:
#     fold = int(sys.argv[1])
fold = 15


def predict(text, topk=3):
    # 抽取
    texts = convert.text_split(text)
    vecs = vectorize.predict(texts)
    preds = extract.model.predict(vecs[None])[0, :, 0]
    preds = np.where(preds > extract.threshold)[0]
    summary = ''.join([texts[i] for i in preds])
    # 生成
    summary = seq2seq.autosummary.generate(summary, topk=topk)
    # 返回
    return summary


if __name__ == '__main__':

    # train
    from tqdm import tqdm

    data = extract.load_data(extract.data_extract_json)
    valid_data = data_split(data, fold, num_folds, 'valid')
    total_metrics = {k: 0.0 for k in metric_keys}
    for d in tqdm(valid_data):
        text = '\n'.join(d[0])
        summary = predict(text)
        metrics = compute_metrics(summary, d[2])
        for k, v in metrics.items():
            total_metrics[k] += v

    metrics = {k: v / len(valid_data) for k, v in total_metrics.items()}
    print(metrics)
    
    # predict
    import json
#     fin = open('datasets/train.json', 'r')
    fin = open('/opt/ml/input/data/training/train.json', 'r')
    for line in fin:
        data = json.loads(line)
        text = ''
        for texti in data['text']:
            text += texti['sentence']+'\n'
        summary = predict(text, topk=3)
        print(summary)
    fin.close()
