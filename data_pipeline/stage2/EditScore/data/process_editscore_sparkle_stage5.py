import vllm  # import vllm first to avoid conflict
import os
import json
import argparse
import numpy as np
import pandas as pd
from io import BytesIO
from PIL import Image
from tqdm import tqdm
from decord import VideoReader, cpu
from editscore import EditScore

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--edit_type', type=str, required=True)
    parser.add_argument('--num_shards', type=int, default=1)
    parser.add_argument('--shard_index', type=int, default=0)
    args = parser.parse_args()

    edit_type = args.edit_type
    num_shards = args.num_shards
    shard_index = args.shard_index

    base_dir = '../../location_train_preview100'
    prompt_file = f'{base_dir}/stage3_generated_prompts_{edit_type}_filtered.csv'
    edited_dir = f'{base_dir}/stage5_edited_video_{edit_type}'
    output_file = f'{base_dir}/stage5_editscore/{edit_type}_{num_shards}_{shard_index}.jsonl'
    num_sample_frames = 4

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # load prompts
    samples=pd.read_csv(prompt_file).to_dict('records')

    # shard (contiguous)
    total = len(samples)
    per_shard = total // num_shards
    remainder = total % num_shards
    start = per_shard * shard_index + min(shard_index, remainder)
    end = start + per_shard + (1 if shard_index < remainder else 0)
    samples = samples[start:end]
    print(f'[shard {shard_index}/{num_shards}] {len(samples)} samples, output -> {output_file}')

    # resume
    done_ids = set()
    if os.path.exists(output_file):
        with open(output_file) as f:
            for line in f:
                done_ids.add(json.loads(line)['id'])
        print(f'Resuming: {len(done_ids)} already done')

    scorer = EditScore(
        backbone='qwen3vl_vllm',
        model_name_or_path='Qwen/Qwen3-VL-8B-Instruct',
        lora_path='EditScore/EditScore-Qwen3-VL-8B-Instruct',
        score_range=25,
        num_pass=4,
        max_model_len=8192,
        max_num_batched_tokens=8192,
        tensor_parallel_size=1,
    )

    pbar = tqdm(samples, desc=f'shard {shard_index}')
    n_ok, n_fail = 0, 0
    for sample in pbar:
        iid = sample['task']
        if iid in done_ids:
            continue

        prompt = sample['prompt']
        source_path=os.path.join(base_dir, sample['src_video'])
        edited_path = f'{edited_dir}/{iid}.mp4'

        try:
            source_vr = VideoReader(source_path, ctx=cpu(0))
            edited_vr = VideoReader(edited_path, ctx=cpu(0))

            T = min(len(source_vr), len(edited_vr))
            # uniformly sample num_sample_frames from [1, T-1] (exclude frame 0)
            indices = np.linspace(1, T - 1, num_sample_frames).round().astype(int).tolist()

            source_frames = source_vr.get_batch(indices).asnumpy()
            edited_frames = edited_vr.get_batch(indices).asnumpy()

            pairs = [
                [Image.fromarray(s), Image.fromarray(e)]
                for s, e in zip(source_frames, edited_frames)
            ]
            prompts = [prompt] * num_sample_frames

            results = scorer.batch_evaluate(pairs, prompts)

            record = {
                'id': iid,
                'prompt': prompt,
                'frame_indices': indices,
                'editscore': results,
            }

            with open(output_file, 'a') as f:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
            n_ok += 1

        except Exception as e:
            n_fail += 1
            print(f'FAIL {iid}: {e}')

        pbar.set_postfix(ok=n_ok, fail=n_fail)
