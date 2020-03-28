#!/usr/bin/env bash

input_file="/content/input-data/complete_dataset.json"
output_dir="/content/grover/model_checkpoints"
model_type="base"

python run_discrimination.py \
    --config_file=/content/grover/lm/configs/${model_type}.json \
    --input_data=${input_file} \
    --output_dir=${output_dir} \
    --use_tpu=False \
    --mode_type=${model_type} \
    --do_train=True \