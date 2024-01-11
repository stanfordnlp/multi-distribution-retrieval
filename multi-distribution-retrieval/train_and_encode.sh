MAX_C_LEN=300
MAX_Q_LEN=70
MAX_Q_SP_LEN=350

domain1=Walmart
domain2=Amazon

BASE_DIR=../dataset_generation/${domain1}-${domain2}
MODEL_NAME=roberta-base

# Train encoders

for domain in ${domain2} ${domain1}; do
    CUDA_VISIBLE_DEVICES=0,1,2,3 python mdr/retrieval/train_single.py \
        --do_train \
        --prefix ${domain}_single \
        --predict_batch_size 2000 \
        --model_name ${MODEL_NAME} \
        --train_batch_size 64 \
        --learning_rate 5e-5 \
        --fp16 \
        --train_file ${BASE_DIR}/train_${domain}.json \
        --predict_file ${BASE_DIR}/val.json \
        --seed 16 \
        --eval-period -1 \
        --max_c_len ${MAX_C_LEN} \
        --max_q_len ${MAX_Q_LEN} \
        --max_q_sp_len ${MAX_Q_SP_LEN} \
        --shared-encoder \
        --warmup-ratio 0.1 \
        --output_dir ${BASE_DIR}/logs/${domain}_single_
done


# Encode corpora

for train_dom in ${domain1} ${domain2}; do
    for domain in ${domain1} ${domain2}; do
        CUDA_VISIBLE_DEVICES=0,1,2,3 python encode_corpus.py \
            --do_predict \
            --predict_batch_size 2000 \
            --model_name ${MODEL_NAME} \
            --predict_file ${BASE_DIR}/${domain}_corpus.json \
            --init_checkpoint ${BASE_DIR}/logs/${train_dom}_single_/*/${train_dom}_single-seed16-bsz64-fp16True-lr5e-05-decay0.0-warm0.1-${MODEL_NAME}/checkpoint_best.pt \
            --embed_save_path ${BASE_DIR}/corpus_encs/train-${train_dom}/${MODEL_NAME}/${domain} \
            --fp16 \
            --max_c_len ${MAX_C_LEN} \
            --num_workers 20 
    done
done
