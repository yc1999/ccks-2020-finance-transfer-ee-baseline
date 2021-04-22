#!/bin/sh

export train_batch_size=2

while [ $train_batch_size -lt 26 ]
do
  echo "------------------train_batch_size:$train_batch_size----------------"
  CUDA_VISIBLE_DEVICES=2 python ./run_bert.py \
  --do_train \
  --save_best \
  --monitor valid_f1 \
  --mode max \
  --task_type trans \
  --epochs 80 \
  --train_batch_size $train_batch_size &
  wait
  echo "----------------------train end--------------------------"
  CUDA_VISIBLE_DEVICES=2 python ./run_bert.py \
  --do_test \
  --task_type trans &
  wait
  echo "----------------------test end--------------------------"

   train_batch_size=`expr $train_batch_size + 2`
done
