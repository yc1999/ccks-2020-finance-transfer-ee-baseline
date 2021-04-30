#!/bin/sh

export K_LIST="5 7 9"
export OUTPUT_DIR=""
export TRAIN_DATA=""
export TRAIN_DATA_DIR=""
export TEST_DATA_DIR=""

for K in $K_LIST
do
  for((GROUP=0;GROUP<=4;GROUP=GROUP+1))
  do
    OUTPUT_DIR="few-shot/$K-shot/group-$GROUP"
    CACHE_DIR="few-shot/$K-shot/group-$GROUP"
    TRAIN_DATA="$K-shot/group-$GROUP.pkl"
    DEV_DATA="few-shot-dev.pkl"

    echo "现在正在执行的是 $K-shot group-$GROUP"
    echo "CACHE_DIR: $CACHE_DIR"
    echo "OUTPUT_DIR: $OUTPUT_DIR"
    echo "TRAIN_DATA: $TRAIN_DATA"
    echo "DEV_DATA: $DEV_DATA"
    echo "--------------------------------------"

    CUDA_VISIBLE_DEVICES=3 python fin_args_qa_thresh.py \
    --train_file ./dataset/$TRAIN_DATA \
    --dev_file ./dataset/$DEV_DATA \
    --train_batch_size 4 \
    --eval_batch_size 4 \
    --learning_rate 4e-5 \
    --num_train_epochs 10 \
    --output_dir output/fin_args_qa_thresh_output/$OUTPUT_DIR \
    --model_dir output/fin_args_qa_thresh_output/$OUTPUT_DIR/best-model \
    --nth_query 5 \
    --normal_file ./question_templates/chinese-description.csv \
    --des_file ./question_templates/chinese-description.csv \
    --eval_per_epoch 3 \
    --max_seq_length 512 \
    --n_best_size 20 \
    --max_answer_length 10 \
    --larger_than_cls \
    --do_train \
    --do_eval \
    --model /home/yc21/project/bert/torch_roberta_wwm \
    --do_lower_case \
    --gpu_ids 0 \
    --dataset dataset/$CACHE_DIR \
    --task_type trans \
    --arch bert
    wait
  done
done