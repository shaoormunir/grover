#!/usr/bin/env bash
export GOOGLE_APPLICATION_CREDENTIALS=/content/iowa-project-2.json

input_file="/content/input-data/complete_dataset.json"
output_dir="gs://model_checkpoint_bucket/grover-discrimination-run-2"
model_type="base"
tpu_name = "grpc://"$COLAB_TPU_ADDR

python run_discrimination.py \
    --config_file=/content/grover/lm/configs/${model_type}.json \
    --input_data=${input_file} \
    --output_dir=${output_dir} \
    --init_checkpoint=44000 \
    --use_tpu=True \
    --tpu_name=${tpu_name} \
    --model_type=${model_type} \
    --do_train=True \