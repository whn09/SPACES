import json
import pandas as pd


texts = []
summaries = []
fin = open('datasets/train.json', 'r')
for line in fin:
    data = json.loads(line)
    text = ''
    for texti in data['text']:
        text += texti['sentence']+'\n'
    print(text)
    summary = 'beginbegin'+data['summary']+'endend'
    print(summary)
    texts.append(text)
    summaries.append(summary)
fin.close()

data = pd.DataFrame({'正文': texts, '摘要': summaries})
data.to_excel('customer.xlsx', index=False)