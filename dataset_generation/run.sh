# Download datasets

mkdir data
pushd data

wget -nc https://dbs.uni-leipzig.de/file/Amazon-GoogleProducts.zip
unzip -n Amazon-GoogleProducts.zip

wget -nc https://pages.cs.wisc.edu/~anhai/data/corleone_data/products.tar.gz --no-check-certificate
tar xfz products.tar.gz

wget https://dl.fbaipublicfiles.com/concurrentqa/data/CQA_dev_all.json
wget https://dl.fbaipublicfiles.com/concurrentqa/data/CQA_train_all.json
wget https://dl.fbaipublicfiles.com/concurrentqa/data/CQA_test_all.json

wget https://dl.fbaipublicfiles.com/concurrentqa/corpora/enron_only_corpus.json
wget https://dl.fbaipublicfiles.com/concurrentqa/corpora/wiki_only_corpus.json

popd

# Prepare datasets to required format

python datagen.py Walmart-Amazon
python datagen.py Amazon-Google
python cqa_datagen.py
