#!/bin/bash

export TORCH_DISTRIBUTED_USE_IPV6=1

NUM_GPUs=8

# 选择你的模型 (取消注释其中一个)
MODEL_NAME=black-forest-labs/FLUX.2-klein-9B

echo "Set Image Editing Model: ${MODEL_NAME}"

# --- 根据模型名称设置差异化参数 ---
if [ "$MODEL_NAME" == "black-forest-labs/FLUX.2-klein-9B" ]; then
    STEPS=4
    NEG_PROMPT="background blur, blurry background, bokeh, shallow depth of field, soft focus, out of focus, lens blur, depth perception blur, macro photography, motion blur, ghosting, trailing, long exposure, smearing, double exposure"
else
    # 默认兜底配置
    STEPS=50
    NEG_PROMPT=""
fi

# Background Replacement
python remove_foreground_sparkle.py \
    --model "$MODEL_NAME" \
    --negative-prompt "$NEG_PROMPT" \
    --num-inference-steps $STEPS \
    --cfg-scale 4.0 \
    --cache-backend cache_dit \
    --tensor-parallel-size $NUM_GPUs \
    --edit-type location \
    --num-shards 1 \
    --shard-index 0