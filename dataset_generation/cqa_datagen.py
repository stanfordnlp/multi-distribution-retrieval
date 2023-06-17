import json
import pandas as pd
import os

os.makedirs('concurrentqa', exist_ok=True)

df1 = pd.read_json('data/CQA_train_all.json', lines=True)
for i, dataset in enumerate(['enron', 'wikipedia']):
    train_f1 = open(f'concurrentqa/train_{dataset}.json', 'w')
    for _, e in df1.iterrows():
        if e['domain'][0] == e['domain'][1] and e['domain'][0] == i:
            ep = {
                '_id': e['_id'],
                'question': e['question'],
                'answers': [e['answer']],
                'pos_paras': [{'title': e['sp'][0]['title'],
                            'text': ' '.join(e['sp'][0]['sents'])}],
                'neg_paras': [],
                'type': e['type'],
                'bridge': e['sp'][0]['title'],
            }
            train_f1.write(json.dumps(ep)+'\n')


df2 = pd.read_json('data/CQA_dev_all.json', lines=True)
val_f = open(f'concurrentqa/val.json', 'w')
for _, e in df2.iterrows():
    if e['domain'][0] != e['domain'][1] and e['type'] == 'comparison':
        ep = {
            '_id': e['_id'],
            'question': e['question'],
            'answers': [e['answer']],
            'pos_paras': [{'title': e['sp'][i]['title'],
                        'text': ' '.join(e['sp'][i]['sents'])} for i in range(2)],
            'neg_paras': [],
            'type': e['type'],
            'bridge': e['sp'][0]['title'],
        }
        val_f.write(json.dumps(ep)+'\n')


df3 = pd.read_json('data/CQA_test_all.json', lines=True)
test_f = open(f'concurrentqa/test.json', 'w')
for _, e in df3.iterrows():
    if e['domain'][0] != e['domain'][1] and e['type'] == 'comparison':
        ep = {
            '_id': e['_id'],
            'question': e['question'],
            'answers': [e['answer']],
            'pos_paras': [{'title': e['sp'][i]['title'],
                        'text': ' '.join(e['sp'][i]['sents'])} for i in range(2)],
            'neg_paras': [],
            'type': e['type'],
            'bridge': e['sp'][0]['title'],
        }
        test_f.write(json.dumps(ep)+'\n')


dom1_file = open(f'concurrentqa/enron_corpus.json', 'w')
for k, e in json.load(open('data/enron_only_corpus.json')).items():
    dom1_file.write(json.dumps(
        {'id': e['id'],
         'title': e['title'],
         'text': e['text']})+'\n')


dom2_file = open(f'concurrentqa/wikipedia_corpus.json', 'w')
for k, e in json.load(open('data/wiki_only_corpus.json')).items():
    dom2_file.write(json.dumps(
        {'id': e['id'],
         'title': e['title'],
         'text': e['text']})+'\n')
