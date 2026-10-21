torchrun --nproc_per_node=8 --nnodes=1 --master_port=30000 examples/wan2.2_fun/predict_v2v_control_batch_sparkle.py \
  --edit_type=location \
  --num_shards=1 \
  --shard_index=0