#!/bin/bash

NUM_GPUS=8
EDIT_TYPE=location

# Kiwi-Sparkle
CKPT_NAME=Kiwi-Sparkle-720P-81F
CKPT_PATH=/path/to/Kiwi-Sparkle-720P-81F/model.safetensors

for rank in $(seq 0 $((NUM_GPUS-1))); do
  python3 test_benchmark_sparkle_bench.py \
    --num_rank $NUM_GPUS \
    --rank $rank \
    --max_frame 81 \
    --edit_type $EDIT_TYPE \
    --save_dir infer_results/$CKPT_NAME \
    --ckpt_path $CKPT_PATH &
done
wait