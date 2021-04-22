#!/bin/sh

export k=1

while [ $k -lt 10 ]
do
  echo "------------------k:$k----------------"
  python ../Cls_data_preprocess.py \
  --k $k &
  wait

  echo "----------------------train start---------------------------"
  CUDA_VISIBLE_DEVICES=2 python ./run_bert.py \
  --do_train \
  --save_best \
  --monitor valid_f1 \
  --mode max \
  --task_type trans \
  --epochs 80 \
  --train_batch_size 2 &
  wait
  echo "----------------------train end---------------------------"

  echo "----------------------test start--------------------------"
  CUDA_VISIBLE_DEVICES=2 python ./run_bert.py \
  --do_test \
  --task_type trans &
  wait
  echo "----------------------test end--------------------------"

   k=`expr $k + 2`
done