torchrun --nproc_per_node=8 --nnodes=1 --master_port=30000 examples/wan/wan_i2v_distilled_sparkle.py \
  --edit_type=location \
  --num_shards=1 \
  --shard_index=0
