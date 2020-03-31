#!/usr/bin/env bash
export GOOGLE_APPLICATION_CREDENTIALS=/content/iowa-project-2.json

input_file="/content/input-data/complete_dataset.json"
output_dir="gs://model_checkpoint_bucket/grover-discrimination-pretrained-large-run-12"
init_checkpoint="gs://model_checkpoint_bucket/grover-discrimination-pretrained-large-run-11"
model_type="large"

python run_discrimination.py \
    --config_file=/content/grover/lm/configs/${model_type}.json \
    --input_data=${input_file} \
    --output_dir=${output_dir} \
    --init_checkpoint=${init_checkpoint} \
    --use_tpu=True \
    --model_type=${model_type} \
    --do_train=True \