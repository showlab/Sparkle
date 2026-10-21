#!/bin/bash

# ================= 配置参数 =================
SHARD_INDEX=0     # 当前机器处理的分片 (vLLM默认利用当前机器全部GPU处理一个SHARD)
NUM_SHARDS=1      # 总分片数量
# ============================================

echo "开始启动多 GPU 并行处理..."
echo "总分片数: $NUM_SHARDS, 当前分片: $SHARD_INDEX, 默认vLLM推理使用本机全部GPU"

python infer_foreground_sparkle_stage3.py \
    --num_shards $NUM_SHARDS \
    --shard_index $SHARD_INDEX \
    --edit_type location

# 等待所有后台任务执行完毕
wait

echo "当前机器分片推理已执行完毕！"