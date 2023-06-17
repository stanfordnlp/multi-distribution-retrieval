#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import torch
import os
import json
import logging
import sys


# dataset = 'Amazon-Google'
# dataset = 'Walmart-Amazon'
dataset = sys.argv[1]
logging.info('Dataset: ', dataset)

if dataset == 'Walmart-Amazon':
    DOMAIN1_DESC_STR = 'longdescr'
    DOMAIN2_DESC_STR = 'proddescrlong'
    DOMAIN1_TITLE_STR = 'title'
    DOMAIN2_TITLE_STR = 'title'
    ID_COL = 'custom_id'
    DOMAIN1_ID = 'id1'
    DOMAIN2_ID = 'id2'
    domain1_csv = 'data/products/walmart.csv'
    domain2_csv = 'data/products/amazon.csv'
    matches_csv = 'data/products/matches_walmart_amazon.csv'
elif dataset == 'Amazon-Google':
    DOMAIN1_DESC_STR = 'description'
    DOMAIN2_DESC_STR = 'description'
    DOMAIN1_TITLE_STR = 'title'
    DOMAIN2_TITLE_STR = 'name'
    ID_COL = 'id'
    DOMAIN1_ID = 'idAmazon'
    DOMAIN2_ID = 'idGoogleBase'
    domain1_csv = 'data/Amazon.csv'
    domain2_csv = 'data/GoogleProducts.csv'
    matches_csv = 'data/Amzon_GoogleProducts_perfectMapping.csv'


df1 = pd.read_csv(domain1_csv, index_col=ID_COL, encoding='unicode_escape' if dataset=='Amazon-Google' else None)
df2 = pd.read_csv(domain2_csv, index_col=ID_COL, encoding='unicode_escape' if dataset=='Amazon-Google' else None)
matches_df = pd.read_csv(matches_csv)

logging.info('Raw Dataset size: Domain 1: %d, Domain 2: %d, Matches: %d', len(df1), len(df2), len(matches_df))


df1_drop_ids = df1.index[df1[DOMAIN1_DESC_STR].isna()].tolist()
df2_drop_ids = df2.index[df2[DOMAIN2_DESC_STR].isna()].tolist()

df1 = df1.dropna(subset=[DOMAIN1_TITLE_STR, DOMAIN1_DESC_STR])
df2 = df2.dropna(subset=[DOMAIN2_TITLE_STR, DOMAIN2_DESC_STR])
matches_df = matches_df[~(matches_df[DOMAIN1_ID].isin(df1_drop_ids) | matches_df[DOMAIN2_ID].isin(df2_drop_ids))]

df1[DOMAIN1_TITLE_STR] = df1[DOMAIN1_TITLE_STR].apply(lambda x: x.replace('\n', ' '))
df1[DOMAIN1_DESC_STR] = df1[DOMAIN1_DESC_STR].apply(lambda x: x.replace('\n', ' '))
df2[DOMAIN2_TITLE_STR] = df2[DOMAIN2_TITLE_STR].apply(lambda x: x.replace('\n', ' '))
df2[DOMAIN2_DESC_STR] = df2[DOMAIN2_DESC_STR].apply(lambda x: x.replace('\n', ' '))

logging.info('Filtered Dataset size: Domain 1: %d, Domain 2: %d, Matches: %d', len(df1), len(df2), len(matches_df))


# Dump Dataset

# Dump Corpus
os.makedirs(dataset, exist_ok=True)
dom1_file = open(f'{dataset}/{dataset.split("-")[0]}_corpus.json', 'w')
for _, e in df1.iterrows():
    dom1_file.write(json.dumps(
        {'id': f'{dataset.split("-")[0]}:{e[DOMAIN1_TITLE_STR]}',
         'title': ' ',
         'text': e[DOMAIN1_DESC_STR]})+'\n')

dom2_file = open(f'{dataset}/{dataset.split("-")[1]}_corpus.json', 'w')
for _, e in df2.iterrows():
    dom2_file.write(json.dumps(
        {'id': f'{dataset.split("-")[1]}:{e[DOMAIN2_TITLE_STR]}',
         'title': ' ',
         'text': e[DOMAIN2_DESC_STR]})+'\n')

# Dump Queries
os.makedirs(dataset, exist_ok=True)

i = 40001
train_df1 = df1[~df1.index.isin(matches_df[DOMAIN1_ID])][[DOMAIN1_TITLE_STR, DOMAIN1_DESC_STR]]
train_f1 = open(f'{dataset}/train_{dataset.split("-")[0]}.json', 'w')
for _, e in train_df1.iterrows():
    ep = {
        '_id': f'{dataset}:{i}',
        'question': e[DOMAIN1_TITLE_STR],
        'answers': [e[DOMAIN1_DESC_STR]],
        'pos_paras': [{'title': ' ', 'text': e[DOMAIN1_DESC_STR]}],
        'neg_paras': [],
        'type': 'comparison',
        'bridge': e[DOMAIN1_DESC_STR],
    }
    train_f1.write(json.dumps(ep)+'\n')
    i += 1

train_df2 = df2[~df2.index.isin(matches_df[DOMAIN2_ID])][[DOMAIN2_TITLE_STR, DOMAIN2_DESC_STR]]
train_f2 = open(f'{dataset}/train_{dataset.split("-")[1]}.json', 'w')
for _, e in train_df2.iterrows():
    ep = {
        '_id': f'{dataset}:{i}',
        'question': e[DOMAIN2_TITLE_STR],
        'answers': [e[DOMAIN2_DESC_STR]],
        'pos_paras': [{'title': ' ', 'text': e[DOMAIN2_DESC_STR]}],
        'neg_paras': [],
        'type': 'comparison',
        'bridge': e[DOMAIN2_DESC_STR],
    }
    train_f2.write(json.dumps(ep)+'\n')
    i += 1


def write_val_test(df, f, i):
    for _, row in df.iterrows():
        e1 = df1.loc[row[DOMAIN1_ID]][[DOMAIN1_TITLE_STR, DOMAIN1_DESC_STR]]
        e2 = df2.loc[row[DOMAIN2_ID]][[DOMAIN2_TITLE_STR, DOMAIN2_DESC_STR]]
        ep = {
            '_id': f'{dataset}:{i}',
            'question': e2[DOMAIN2_TITLE_STR],
            'answers': [e1[DOMAIN1_DESC_STR], e2[DOMAIN2_DESC_STR]],
            'pos_paras': [{'title': ' ', 'text': e1[DOMAIN1_DESC_STR]}, {'title': ' ', 'text': e2[DOMAIN2_DESC_STR]}],
            'neg_paras': [],
            'type': 'comparison',
            'bridge': e1[DOMAIN1_DESC_STR],
            'domain': [0,1]
        }
        f.write(json.dumps(ep)+'\n')
        i += 1
        ep = {
            '_id': f'{dataset}:{i}',
            'question': e1[DOMAIN1_TITLE_STR],
            'answers': [e1[DOMAIN1_DESC_STR], e2[DOMAIN2_DESC_STR]],
            'pos_paras': [{'title': ' ', 'text': e1[DOMAIN1_DESC_STR]}, {'title': ' ', 'text': e2[DOMAIN2_DESC_STR]}],
            'neg_paras': [],
            'type': 'comparison',
            'bridge': e2[DOMAIN2_DESC_STR],  
            'domain': [0,1]
        }
        f.write(json.dumps(ep)+'\n')
        i += 1
    return i

val_f = open(f'{dataset}/val.json', 'w')
test_f = open(f'{dataset}/test.json', 'w')

shuffled_matches = matches_df.sample(frac=1, random_state=0)
n_val = len(shuffled_matches)//2
i = write_val_test(shuffled_matches[:n_val], val_f, i)
i = write_val_test(shuffled_matches[n_val:], test_f, i)

logging.info('Done writing datasets')


if False:
    from transformers import AutoTokenizer, AutoModel
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import seaborn as sns

    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    model = AutoModel.from_pretrained('roberta-base').cuda()

    def get_encs(docs, max_len=300):
        with torch.no_grad():
            title_encs = []
            all_titles = docs
            bs = 256
            for titles in tqdm(np.array_split(all_titles, len(all_titles)//bs), leave=False):
                title_encs.append(model(**tokenizer.batch_encode_plus(
                    titles, max_length=max_len, 
                    pad_to_max_length=True, 
                    return_tensors='pt').to('cuda:0'))[0][:, 0, :].to('cpu'))
            title_encs = torch.cat(title_encs, 0)
        return title_encs

    def get_rank(idx, n, scores):
        idxs_sorted = torch.argsort(scores, descending=True)
        target = torch.nn.functional.one_hot(torch.tensor(idx), n)
        return 1 + torch.nonzero(target[idxs_sorted]).item()

    title1_encs = get_encs(df1[DOMAIN1_TITLE_STR], 70)
    title2_encs = get_encs(df2[DOMAIN2_TITLE_STR], 70)

    title1_encs.shape, title2_encs.shape

    desc1_encs = get_encs(df1[DOMAIN1_DESC_STR], 300)
    desc2_encs = get_encs(df2[DOMAIN2_DESC_STR], 300)

    ranks_dom1 = []
    ranks_dom2 = []

    for _, matches in matches_df.iterrows():
        id1, id2 = matches.tolist()
        idx1, idx2 = df1.index.get_loc(id1), df2.index.get_loc(id2)
        query_emb = title2_encs[idx2]
        scores1 = desc1_encs @ query_emb
        scores2 = desc2_encs @ query_emb
        _, res1 = torch.topk(scores1, k=100)
        _, res2 = torch.topk(scores2, k=100)
        rank1 = get_rank(idx1, len(df1), scores1)
        rank2 = get_rank(idx2, len(df2), scores2)
        ranks_dom1.append(rank1)
        ranks_dom2.append(rank2)

    plt.figure(figsize=(12,3))
    plt.subplot(131)
    sns.histplot([r/len(df1) for r in ranks_dom1], label=f"{dataset.split('-')[0]}")
    sns.histplot([r/len(df2) for r in ranks_dom2], label=f"{dataset.split('-')[1]}")
    plt.legend()
    plt.subplot(132)
    sns.histplot(ranks_dom1)
    plt.title(f"{dataset.split('-')[0]} Rank")
    plt.subplot(133)
    plt.title(f"{dataset.split('-')[1]} Rank")
    sns.histplot(ranks_dom2)
    plt.tight_layout()
    plt.show(f'{dataset}/ranks')

    idx1 = [df1.index.get_loc(id_) for id_ in matches_df['id1'].tolist()]
    idx2 = [df2.index.get_loc(id_) for id_ in matches_df['id2'].tolist()]

    tsne = TSNE(n_components=2, perplexity=15, random_state=42, init="random", learning_rate=200)
    vis_dims2 = tsne.fit_transform(torch.cat([desc1_encs, desc2_encs], 0))

    tsne = TSNE(n_components=2, perplexity=15, random_state=42, init="random", learning_rate=200)
    tvis_dims2 = tsne.fit_transform(torch.cat([title1_encs, title2_encs], 0))

    plt.figure(figsize=(10,5))
    plt.subplot(121)
    plt.scatter(*vis_dims2[:len(desc1_encs)].T, color='#1f77b4', alpha=0.03, label=f"{dataset.split('-')[0]}")
    plt.scatter(*vis_dims2[len(desc1_encs):].T, color='#ff7f0e', alpha=0.03, label=f"{dataset.split('-')[1]}")
    plt.scatter(*vis_dims2[idx1].T, color='#2ca02c', alpha=1, s=1, label=f"matched_{dataset.split('-')[0]}")
    plt.scatter(*vis_dims2[[i+len(desc1_encs) for i in idx2]].T, color='#d62728', alpha=1, s=1, label=f"matched_{dataset.split('-')[1]}")
    plt.legend()
    plt.title('Description Embeddings')

    plt.subplot(122)
    plt.scatter(*tvis_dims2[:len(title1_encs)].T, color='#1f77b4', alpha=0.05, label=f"{dataset.split('-')[0]}")
    plt.scatter(*tvis_dims2[len(title1_encs):].T, color='#ff7f0e', alpha=0.05, label=f"{dataset.split('-')[1]}")
    plt.scatter(*tvis_dims2[idx1].T, color='#2ca02c', alpha=1, s=1, label=f"matched_{dataset.split('-')[0]}")
    plt.scatter(*tvis_dims2[[i+len(title1_encs) for i in idx2]].T, color='#d62728', alpha=1, s=1, label=f"matched_{dataset.split('-')[1]}")
    plt.legend()
    plt.title('Title Embeddings')

    plt.tight_layout()
    plt.save(f'{dataset}/embs')
