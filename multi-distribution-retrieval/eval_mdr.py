import argparse
import collections
import glob
import json
import logging

import faiss
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from mdr.retrieval.models.mhop_retriever import RobertaRetriever
from mdr.retrieval.utils.utils import load_saved, move_to_cuda


def merge(D, I, D_alt, I_alt, num, num_alt):
    if type(num) == int:
        assert num <= D.shape[1] and num_alt <= D_alt.shape[1]
        D_new, I_new = D[:, :num], I[:, :num]
        D_alt_new, I_alt_new = D_alt[:, :num_alt], I_alt[:, :num_alt]
        D_merged = np.concatenate([D_new, D_alt_new], 1)
        I_merged = np.concatenate([I_new, I_alt_new], 1)
        domains = np.zeros_like(D)
        domains[:, num:] = 1
    else:
        D_merged = np.concatenate([D, D_alt], 1)
        I_merged = np.concatenate([I, I_alt], 1)
        domains = np.zeros_like(D_merged)
        domains[:, D.shape[1]:] = 1
    sort_idx = np.argsort(-D_merged, axis=1)  # negative for descending
    D_merged = D_merged[np.arange(D_merged.shape[0])[:, None], sort_idx]
    I_merged = I_merged[np.arange(I_merged.shape[0])[:, None], sort_idx]
    domains = domains[np.arange(domains.shape[0])[:, None], sort_idx]
    if type(num) != int:
        D_merged = D_merged[:, :D.shape[1]]
        I_merged = I_merged[:, :D.shape[1]]
        domains = domains[:, :D.shape[1]]
    return D_merged, I_merged, domains


@torch.no_grad()
def retrieve_per_query_oracle(
    args, questions, ds_indices, ds_items, has_domains, domains,
    tokenizer, model, index, index_alt, num, id2doc, id2doc_alt):

    if not args.indexpath_alt:
        raise NotImplementedError

    metrics = []
    num_questions = len(questions) 
    for b_start in tqdm(range(0, num_questions, args.batch_size), leave=False):
        batch_q = questions[b_start:b_start + args.batch_size] 
        batch_idxs = ds_indices[b_start:b_start + args.batch_size] 
        batch_ann = ds_items[b_start:b_start + args.batch_size]
        if has_domains:
            batch_domains = domains[b_start:b_start + args.batch_size]
        bsize = len(batch_q)

        batch_q_encodes = tokenizer.batch_encode_plus(
            batch_q, max_length=args.max_q_len, pad_to_max_length=True, return_tensors="pt")
        batch_q_encodes = move_to_cuda(dict(batch_q_encodes))
        q_embeds = model.encode_q(
            batch_q_encodes["input_ids"], batch_q_encodes["attention_mask"],
            batch_q_encodes.get("token_type_ids", None))
        q_embeds_numpy = q_embeds.cpu().contiguous().numpy()

        # get the nearest neighbors for each embedded question    
        D, I = index.search(q_embeds_numpy, args.topk) 
        D_alt, I_alt = index_alt.search(q_embeds_numpy, args.topk) 

        for idx in range(bsize):
            pos_paras = batch_ann[idx]["pos_paras"]
            gt_texts = [s['text'] for s in pos_paras]
            retrieved_texts = [
                id2doc[str(id)]['text'] for i, id in enumerate(I[idx])]
            retrieved_texts_alt = [
                id2doc_alt[str(id)]['text'] for i, id in enumerate(I_alt[idx])]
            recall = 0
            mAP_m = 0
            if batch_ann[idx]['type'] == 'bridge':
                if gt_texts[0] in retrieved_texts or gt_texts[0] in retrieved_texts_alt:
                    recall = 1
            elif batch_ann[idx]['type'] == 'comparison':
                if gt_texts[0] in retrieved_texts:
                    if gt_texts[1] in retrieved_texts_alt:
                        if retrieved_texts.index(gt_texts[0]) + retrieved_texts_alt.index(gt_texts[1]) + 2 <= args.topk:
                            recall = 1
                            rank1 = retrieved_texts.index(gt_texts[0])+1
                            rank2 = retrieved_texts_alt.index(gt_texts[1])+1
                            mAP_m = 0.5*(1/min(rank1, rank2) + 2/(rank1 + rank2))
                        else:
                            recall = 0.5
                            rank1 = retrieved_texts.index(gt_texts[0])+1
                            rank2 = retrieved_texts_alt.index(gt_texts[1])+1
                            mAP_m = 0.5*1/min(rank1, rank2)
                    else:
                        recall = 0.5
                        rank1 = retrieved_texts.index(gt_texts[0])+1
                        mAP_m = 0.5*(1/rank1)
                elif gt_texts[0] in retrieved_texts_alt:
                    if gt_texts[1] in retrieved_texts:
                        if retrieved_texts_alt.index(gt_texts[0]) + retrieved_texts.index(gt_texts[1]) + 2 <= args.topk:
                            recall = 1
                            rank1 = retrieved_texts_alt.index(gt_texts[0])+1
                            rank2 = retrieved_texts.index(gt_texts[1])+1
                            mAP_m = 0.5*(1/min(rank1, rank2) + 2/(rank1 + rank2))
                        else:
                            recall = 0.5
                            rank1 = retrieved_texts_alt.index(gt_texts[0])+1
                            rank2 = retrieved_texts.index(gt_texts[1])+1
                            mAP_m = 0.5*1/min(rank1, rank2)
                    else:
                        recall = 0.5
                        rank1 = retrieved_texts_alt.index(gt_texts[0])+1
                        mAP_m = 0.5*(1/rank1 + 0)
                elif gt_texts[1] in retrieved_texts:
                    recall = 0.5
                    rank1 = retrieved_texts.index(gt_texts[1])+1
                    mAP_m = 0.5*(1/rank1)
                elif gt_texts[1] in retrieved_texts_alt:
                    recall = 0.5
                    rank1 = retrieved_texts_alt.index(gt_texts[1])+1
                    mAP_m = 0.5*(1/rank1)
            else:
                raise NotImplementedError

            metrics.append({
                "_id": batch_ann[idx]["_id"],
                "question": batch_ann[idx]["question"],
                "recall": recall,
                "mAP_m": mAP_m,
                "type": batch_ann[idx]['type'],
                "domain": tuple(
                    'W' if d==0 else 'E' for d in batch_domains[idx]
                    ) if "sp" in batch_ann[idx] else tuple(
                        'W' if d==0 else 'A' for d in batch_domains[idx]),
            })

    return metrics


@torch.no_grad()
def retrieve(
    args, questions, ds_indices, ds_items, has_domains, domains,
    tokenizer, model, index, index_alt, num, id2doc, id2doc_alt):

    metrics = []
    retrieval_outputs = []
    hop1relevance_domain0 = collections.defaultdict(dict)
    hop1relevance_domain1 = collections.defaultdict(dict)
    num_questions = len(questions) 
    for b_start in tqdm(range(0, num_questions, args.batch_size), leave=False):
        batch_q = questions[b_start:b_start + args.batch_size] 
        batch_idxs = ds_indices[b_start:b_start + args.batch_size] 
        batch_ann = ds_items[b_start:b_start + args.batch_size]
        if has_domains:
            batch_domains = domains[b_start:b_start + args.batch_size]
        bsize = len(batch_q)

        batch_q_encodes = tokenizer.batch_encode_plus(
            batch_q, max_length=args.max_q_len, pad_to_max_length=True, return_tensors="pt")
        batch_q_encodes = move_to_cuda(dict(batch_q_encodes))
        q_embeds = model.encode_q(
            batch_q_encodes["input_ids"], batch_q_encodes["attention_mask"],
            batch_q_encodes.get("token_type_ids", None))
        q_embeds_numpy = q_embeds.cpu().contiguous().numpy()

        # get the nearest neighbors for each embedded question    
        D, I = index.search(q_embeds_numpy, args.topk) 
        if args.indexpath_alt:
            D_alt, I_alt = index_alt.search(q_embeds_numpy, args.topk) 
        else:
            D_alt, I_alt = None, None

        if type(num) == int:
            num_alt = args.topk - num
        else:
            num_alt = None
        D_merged, I_merged, domains_merged = merge(D, I, D_alt, I_alt, num=num, num_alt=num_alt)

        # save the relevance scores for later analysis 
        for idx in range(len(batch_idxs)):
            ex_idx = batch_idxs[idx]
            hop1relevance_domain0[ex_idx] = {'scores': D[idx], 'docids': I[idx]}
            if D_alt is not None:
                hop1relevance_domain1[ex_idx] = {'scores': D_alt[idx], 'docids': I_alt[idx]}

        for idx in range(bsize):
            ds_idx = batch_idxs[idx]
            pos_paras = batch_ann[idx]["pos_paras"]
            gt_texts = [s['text'] for s in pos_paras]
            curr_domain = domains_merged[idx]
            retrieved_texts = [
                id2doc[str(id)]['text'] if curr_domain[i] == 0 else id2doc_alt[str(id)]['text']
                for i, id in enumerate(I_merged[idx])
            ]
            recall = 0
            mAP_m = 0
            if batch_ann[idx]['type'] == 'bridge':
                if gt_texts[0] in retrieved_texts:
                    recall = 1
            elif batch_ann[idx]['type'] == 'comparison':
                for i in range(len(gt_texts)):
                    if gt_texts[i] in retrieved_texts:
                        recall += 1/len(gt_texts)
                rank = []
                for i, text in enumerate(retrieved_texts):
                    if text in gt_texts:
                        rank.append(i+1)
                for i, r in enumerate(sorted(rank)[:2]):
                    mAP_m += (i+1)/r
                mAP_m /= 2
            else:
                raise NotImplementedError

            metrics.append({
                "_id": batch_ann[idx]["_id"],
                "question": batch_ann[idx]["question"],
                "recall": recall,
                "mAP_m": mAP_m,
                "type": batch_ann[idx]['type'],
                "domain": tuple(
                    'W' if d==0 else 'E' for d in batch_domains[idx]
                    ) if "sp" in batch_ann[idx] else tuple(
                        'W' if d==0 else 'A' for d in batch_domains[idx]),
            })

            retrieval_outputs.append({
                "_id": batch_ann[idx]["_id"],
                "question": batch_ann[idx]["question"],
                "candidate_chains": retrieved_texts,
            })

    return metrics, retrieval_outputs, hop1relevance_domain0, hop1relevance_domain1


def main(args):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if (logger.hasHandlers()):
        logger.handlers.clear()
    console = logging.StreamHandler()
    logger.addHandler(console)

    # load the qa pairs
    logger.info("Loading data...")
    ds_items = [json.loads(_) for _ in open(args.raw_data).readlines()]
    logger.info(f"Size: {len(ds_items)}")

    # load the model 
    logger.info("Loading trained model...")
    bert_config = AutoConfig.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = RobertaRetriever(bert_config, args)
    if args.model_path != args.model_name:
        model = load_saved(model, args.model_path, exact=False)

    cuda = torch.device('cuda')
    model.to(cuda)
    model.eval()

    # load the encoded passages and build the indices 
    logger.info("Building index...")
    d = 768 if 'roberta' in args.model_name else 384
    xb = np.load(args.indexpath).astype('float32')

    if args.indexpath_alt:
        logger.info("Building alternate index...")
        xb_alt = np.load(args.indexpath_alt).astype('float32')

    index = faiss.IndexFlatIP(d)
    index.add(xb)

    if args.indexpath_alt:
        index_alt = faiss.IndexFlatIP(d)
        index_alt.add(xb_alt)

    # load the passages in raw format
    logger.info(f"Loading corpus...")
    id2doc = json.load(open(args.corpus_dict))
    if isinstance(id2doc["0"], list):
        id2doc = {k: {"title":v[0], "text": v[1]} for k, v in id2doc.items()}

    logger.info(f"Corpus size {len(id2doc)}")

    if args.corpus_dict_alt:
        logger.info(f"Loading corpus alt...")
        id2doc_alt = json.load(open(args.corpus_dict_alt))
        if isinstance(id2doc_alt["0"], list):
            id2doc_alt = {k: {"title":v[0], "text": v[1]} for k, v in id2doc_alt.items()}
        logger.info(f"Corpus size {len(id2doc_alt)}")

    logger.info("Encoding questions and searching")
    questions = [_["question"][:-1] if _["question"].endswith("?") else _["question"] for _ in ds_items]
    ds_indices = [_["_id"][:-1] if _["_id"].endswith("?") else _["_id"] for _ in ds_items]

    has_domains = 0
    if "domain" in list(ds_items[0].keys()):
        has_domains = 1
        if "CQA" not in args.raw_data:
            domains = [_["domain"] for _ in ds_items]
        else:
            domains = []
            for d_ex in ds_items:
                sp_ex = d_ex["sp"]
                domain_ex = []
                sptitle0 = sp_ex[0]['title']
                if "e" in sptitle0 and "_p" in sptitle0:
                    domain_ex.append(1)
                else:
                    domain_ex.append(0)

                sptitle1 = sp_ex[1]['title']
                if "e" in sptitle1 and "_p" in sptitle1:
                    domain_ex.append(1)
                else:
                    domain_ex.append(0)
                domains.append(domain_ex)

    dfs = []
    dfs_mAP_m = []
    metric_dicts = []

    wiki_fracs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 'merged']
    for wiki_fraction in wiki_fracs:
        if type(wiki_fraction) == float:
            num = int((1-wiki_fraction)*args.topk)
            num_alt = args.topk - num
            assert num + num_alt == args.topk
        else:
            num = wiki_fraction

        metrics, _, _, _ = retrieve(
            args, questions, ds_indices, ds_items, has_domains, domains,
            tokenizer, model, index, index_alt, num, id2doc, id2doc_alt)
        metric_dicts.append(metrics)
        df = pd.DataFrame(metrics)
        df1 = df.groupby(['type', 'domain'])['recall'].agg(q_num='count', recall='mean')
        dfs.append(df1)
        df2 = df.groupby(['type', 'domain'])['mAP_m'].agg(q_num='count', mAP_m='mean')
        dfs_mAP_m.append(df2)

    wiki_fracs += ['per_query']
    metrics = retrieve_per_query_oracle(
        args, questions, ds_indices, ds_items, has_domains, domains,
        tokenizer, model, index, index_alt, num, id2doc, id2doc_alt)
    metric_dicts.append(metrics)
    df = pd.DataFrame(metrics)
    df1 = df.groupby(['type', 'domain'])['recall'].agg(q_num='count', recall='mean')
    dfs.append(df1)
    df2 = df.groupby(['type', 'domain'])['mAP_m'].agg(q_num='count', mAP_m='mean')
    dfs_mAP_m.append(df2)

    k1 = pd.concat(
        [dfs[0]['q_num']/100]+[
            x['recall'].rename(f'frac={f}') for x, f in zip(dfs, wiki_fracs)
        ], axis=1).T
    k2 = pd.concat(
        [dfs_mAP_m[0]['q_num']/100]+[
            x['mAP_m'].rename(f'frac={f}') for x, f in zip(dfs_mAP_m, wiki_fracs)
        ], axis=1).T

    df = k1['comparison'].join(k2['comparison'], lsuffix='_recall', rsuffix='_mAP')
    df[["('W', 'A')_recall", "('W', 'A')_mAP"]] = df[["('W', 'A')_recall", "('W', 'A')_mAP"]].apply(lambda x: round(x*100, 2))
    df["('W', 'A')_recall"] = df["('W', 'A')_recall"].apply(lambda x: f'{x:.2f}')
    df["('W', 'A')_mAP"] = df["('W', 'A')_mAP"].apply(lambda x: f'{x:.2f}')

    df['recall/mAP'] =  df["('W', 'A')_recall"] + '/' + df["('W', 'A')_mAP"]
    df = df.rename(columns={"domain": "method", "('W', 'A')_recall": "recall", "('W', 'A')_mAP": "mAP"})
    df = df[['recall/mAP']]

    print(df.to_string())


def parse_arguments():
    parser = argparse.ArgumentParser(description='Description of your script')
    
    parser.add_argument('--dataset', type=str, default='Walmart-Amazon', help='Dataset name')
    parser.add_argument('--known_domain', type=str, default='Walmart', help='Known domain')
    parser.add_argument('--model_name', type=str, default='roberta-base', help='Model name')
    parser.add_argument('--topk', type=int, default=10, help='Topk value')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--max_q_len', type=int, default=70, help='Maximum query length')
    
    args = parser.parse_args()
    args.base_dir = f'../dataset_generation/{args.dataset}'
    
    args.corpus_dict = f'{args.base_dir}/corpus_encs/train-{args.known_domain}/{args.model_name}/{args.dataset.split("-")[0]}/id2doc.json'
    args.corpus_dict_alt = f'{args.base_dir}/corpus_encs/train-{args.known_domain}/{args.model_name}/{args.dataset.split("-")[1]}/id2doc.json'
    args.indexpath = f'{args.base_dir}/corpus_encs/train-{args.known_domain}/{args.model_name}/{args.dataset.split("-")[0]}/idx.npy'
    args.indexpath_alt = f'{args.base_dir}/corpus_encs/train-{args.known_domain}/{args.model_name}/{args.dataset.split("-")[1]}/idx.npy'
    args.model_path = glob.glob(f'{args.base_dir}/logs/{args.known_domain}_single_/*/{args.known_domain}_single-seed16-bsz64-fp16True-lr5e-05-decay0.0-warm0.1-{args.model_name}/checkpoint_best.pt')[0]
    args.raw_data = f'{args.base_dir}/val.json'
    
    return args


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
