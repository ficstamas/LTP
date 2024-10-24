#!/usr/bin/env bash

TASKS=("MNLI" "RTE" "MRPC" "STSB" "SST2" "QNLI" "QQP")


for task in "${TASKS[@]}"
do
  python examples/text-classification run_glue.py \
         --model_name_or_path kssteven/ibert-roberta-base \
         --output_dir "ckpt/$task/" \
         --task "$task" --do_train --do_eval \
         --evaluation_strategy epoch \
         --max_seq_length 256 \
         --per_device_train_batch_size 16 \
         --learning_rate 2e-5 \
         --num_train_epochs 3 \
         --save_strategy "epoch" \
         --save_total_limit 1
done
