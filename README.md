# Resources and Evaluations for Multi-Distribution Dense Information Retrieval

This repository contains code for our [paper](http://arxiv.org/abs/2306.12601).

***Overview*** We study the underexplored problem of multi-distribution information retrieval (IR) where given a query, systems need to retrieve passages from within multiple collections, each drawn from a different distribution. Some of these collections and distributions might not be available at training time. To evaluate methods or multi-distribution retrieval, we design three benchmarks for this task from existing single-distribution datasets, namely, a dataset based on question answering and two based on entity matching. We explore simple methods to improve quality on this task which allocate the fixed retrieval budget (top-k passages) strategically across domains. We hope these resources facilitate futher exploration!

<p align="center"><img width="95%" src="imgs/system_overview.png" /></p>

### Obtaining the datasets

Run the following commands to download and process the datasets into the required format:
```bash
cd dataset_generation
bash run.sh
```

### Using the datasets

The commands above will download the three datasets with the following directory structure:
```
dataset_generation
├── Amazon-Google
├── concurrentqa
└── Walmart-Amazon
    ├── Walmart_corpus.json
    ├── Amazon_corpus.json
    ├── train_Walmart.json
    ├── train_Amazon.json
    ├── val.json
    └── test.json
```

The `xyz_corpus.json` contains the corpus of documents from which retrieval is performed. The corpus files have the following format:
```json
{"id": "unique_passage_id", "title": "passage_title", "text": "passage_body"}
```

The `train_xyz.json` file contains the training data from the known distribution (say Walmart products) and is used to train the retrieval encoders. Each training example has the following relevant fields:
```json
{
 "_id": "unique_query_id",
 "question": "query",
 "pos_paras": [{"title": "passage_title", "text": "passage_body"}],
 "neg_paras": []
}
```
Note that negative paragraphs are not provided in the dataset. During training, a negative paragraphs is sampled for each example.

The `val.json` and `test.json` files have a similar format with the difference that the queries here require two passages to be retrieved -- one from each of the two distributions. So the `pos_paras` field would now have two passages. For example, the query could be some product like 'Acer Iconia Tablet Bluetooth Keyboard' and the two passages would be the product listing from Walmart and Amazon.

The task is to train the retriever on one of the distributions (i.e. one `train_xyz.json`) and devise mechanisms to correctly retrieve both passages for the test queries (`test.json`). In the Walmart-Amazon case, this would mean that we train on Walmart queries in `train_Walmart.json` (the known distribution) and evaluate on queries in `test.json` that require retrieval from both Walmart (known) and Amazon (unknown) corpora.

### Running the experiments

1. Requirements are provided in `multi-distribution-retrieval/requirements.txt` and a setup script is given at `multi-distribution-retrieval/setup.sh`.
2. Download the datasets using the instructions provided above
3. The next step is to train the encoders on the known domain. Once this is done, we encode the known and unknown corpora using this trained encoder to speed up inference by not having to encode the entire corpus for each query. The following commands are used for this:
    ```bash
    cd multi-distribution-retrieval
    bash train_and_encode.sh
    ```
    This trains two encoders on the Walmart Amazon dataset. The first is with Walmart as the known domain and the second with Amazon as the known domain. Next, the two corpora are encoded using these two encoders leading to four sets of corpus encodings.
4. With the encoders trained and corpora encoded, we can now run the evaluation. This is done using `eval_mdr.py` script in the `multi-distribution-retrieval` directory. It prints out the recall@k and average precision numbers for the various allocation strategies i.e. naive merging, various per-task allocation and per-query allocation.
    
    Run this as:
    ```bash
    python eval_mdr.py --dataset Walmart-Amazon --known_domain Walmart --model_name roberta-base --topk 10 --batch_size 32
    ```

### Citations

If you use this codebase, or otherwise found our work valuable, please cite:
```bibtex
@article{chatterjee2023retrieval,
  title={Resources and Evaluations for Multi-Distribution Dense Information Retrieval},
  author={Chatterjee, Soumya and Khattab, Omar and Arora, Simran},
  journal={Proceedings of the 2023 ACM SIGIR Workshop on Retrieval-Enhanced Machine Learning (REML ’23)},
  year={2023}
}
```

As well as the original creators of the datasets you use:

***ConcurrentQA***
```bibtex
@article{arora2023reasoning,
    title={Reasoning over Public and Private Data in Retrieval-Based Systems}, 
    author={Simran Arora and Patrick Lewis and Angela Fan and Jacob Kahn and Christopher Ré},
    year={2023},
    journal={Transactions of Computational Linguistics (TACL '23)},
}
```

***Walmart-Amazon***
```bibtex
@misc{magellandata,
    title={The Magellan Data Repository},
    howpublished={\url{https://sites.google.com/site/anhaidgroup/projects/data}},
    author = {Das, Sanjib and Doan, AnHai and G. C., Paul Suganthan and Gokhale, Chaitanya and Konda, Pradap and Govind, Yash and Paulsen, Derek},
    institution={University of Wisconsin-Madison},
    year = {2017}
}
```

***Amazon-Google***
```bibtex
@article{kopcke2010evaluation,
  title={Evaluation of entity resolution approaches on real-world match problems},
  author={K{\"o}pcke, Hanna and Thor, Andreas and Rahm, Erhard},
  journal={Proceedings of the VLDB Endowment},
  volume={3},
  number={1-2},
  pages={484--493},
  year={2010},
  publisher={VLDB Endowment}
}
```

The code is based on those of [ConcurrentQA](https://github.com/facebookresearch/concurrentqa) and [Multi-Hop Dense Text Retrieval](https://github.com/facebookresearch/multihop_dense_retrieval). If you use our code, please also cite them:

```bibtex
@article{xiong2020answering,
  title={Answering Complex Open-Domain Questions with Multi-Hop Dense Retrieval},
  author={Xiong, Wenhan and Li, Xiang Lorraine and Iyer, Srinivasan and Du, Jingfei and Lewis, Patrick and Wang, William Yang and Mehdad, Yashar and Yih, Wen-tau and Riedel, Sebastian and Kiela, Douwe and O{\u{g}}uz, Barlas},
  journal={International Conference on Learning Representations},
  year={2021}
}
```