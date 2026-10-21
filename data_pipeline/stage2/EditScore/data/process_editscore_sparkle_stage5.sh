#!/bin/bash

export PYTHONPATH="./editscore${PYTHONPATH:+:$PYTHONPATH}"

# ================= 配置参数 =================
NUM_GPUs=8        # 请根据你实际使用的当前机器 GPU 数量修改 (例如 4 或 8)
INDEX_BEGIN=0     # 当前机器处理的分片起始索引 (如果你有多台机器, 第二台可以从 8 开始, 每台机器处理第INDEX_BEGIN起，NUM_GPUs个分片)
NUM_SHARDS=8      # 总分片数量
# ============================================

echo "开始启动多 GPU 并行处理..."
echo "总分片数: $NUM_SHARDS, 当前机器使用 GPU 数量: $NUM_GPUs, 起始索引: $INDEX_BEGIN"

# 循环分配任务给每个 GPU
for (( i=0; i<$NUM_GPUs; i++ ))
do
    # 计算当前进程对应的分片索引
    CURRENT_SHARD=$((i + INDEX_BEGIN))

    echo "启动进程: GPU $i 正在处理 Shard $CURRENT_SHARD ..."

    # 核心命令：绑定 CUDA 设备，传入分片参数，并在末尾加上 & 让其在后台运行
    CUDA_VISIBLE_DEVICES=$i python data/process_editscore_sparkle_stage5.py \
        --edit_type location \
        --shard_index $CURRENT_SHARD \
        --num_shards $NUM_SHARDS &
done

# 等待所有后台任务执行完毕
wait

echo "当前机器所有 $NUM_GPUs 个进程均已执行完毕！"