import base64
import os
import json
import argparse
import pandas as pd
from io import BytesIO

from vllm import LLM, SamplingParams
from tqdm import tqdm
from PIL import Image
import decord

def frame_to_base64(frame_np):
    """Convert a single numpy frame to base64 data URL."""
    img = Image.fromarray(frame_np)
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_base64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{img_base64_str}"

def read_video_2fps(video_path):
    """Read local video, return all frames sampled at 2 FPS as base64."""
    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    fps = vr.get_avg_fps()
    frame_interval = max(1, int(round(fps / 2.0)))
    frame_indices = list(range(0, len(vr), frame_interval))
    frames_np = vr.get_batch(frame_indices).asnumpy()
    return [frame_to_base64(f) for f in frames_np]


# ==============================================================================
# Round 2 Prompt: Track labeled objects across subsequent frames
# ==============================================================================
SYSTEM_PROMPT_TRACKING = """
You are an expert visual grounding analyzer. Your task is to locate specific target objects in the provided image based on an input list of labels.

### Constraints
- Output ONLY a valid JSON array of objects. No markdown code blocks, no preamble.
- JSON Schema: `{"bbox_2d": [x1, y1, x2, y2] or [], "label": "string"}`
- BBox Format: `[top-left x, top-left y, bottom-right x, bottom-right y]`
- **Strict Label Matching (CRITICAL)**: Your output array MUST contain exactly the same labels provided in the input "Label List". Do not add, modify, or omit any labels.
- **Overlay Detection (CRITICAL)**: Text overlays, titles, subtitles, watermarks, and logos are valid tracking targets. They may shift position or change content across frames. Detect them if they are visible, regardless of whether the text content has changed.
- **Anti-Hallucination & Visibility Rule (CRITICAL)**: You must ONLY output actual bounding box coordinates for objects that are CLEARLY VISIBLE.
- If a requested object is occluded, moved out of frame, or simply not visible, you MUST set its `bbox_2d` value to an empty list `[]`. Do not guess its location.

### Example
**User Input:**
Label List: ["red apple", "wooden chair", "white title text overlay"]

**Scenario:** The "red apple" and "white title text overlay" are visible, but "wooden chair" has moved out of frame.

**Expected Output:**
[
    {"bbox_2d": [150, 200, 250, 300], "label": "red apple"},
    {"bbox_2d": [], "label": "wooden chair"},
    {"bbox_2d": [10, 20, 400, 60], "label": "white title text overlay"}
]
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--edit_type", type=str, required=True)
    parser.add_argument("--num_shards", type=int, default=None)
    parser.add_argument("--shard_index", type=int, default=None)
    args = parser.parse_args()

    edit_type = args.edit_type

    # ---- load filter id set ----
    filter_path = f"../location_train_preview100/stage3_generated_prompts_{edit_type}_filtered.csv"
    print(f"Loading filter ids from {filter_path} ...")

    keep_ids = set(pd.read_csv(filter_path, usecols=['task'], dtype={'task': str})['task'].dropna())

    print(f"Filter: {len(keep_ids)} ids to keep.")

    # ---- load round1 results ----
    r1_path = f"../location_train_preview100/stage3_foreground_grounding_{edit_type}_gathered.jsonl"
    print(f"Loading Round 1 results from {r1_path} ...")
    with open(r1_path, 'r') as f:
        samples = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(samples)} Round 1 samples.")

    # ---- apply filter ----
    before = len(samples)
    samples = [s for s in samples if s['id'] in keep_ids]
    print(f"After filter: {len(samples)} / {before} samples remain.")

    # ---- sharding ----
    assert (
            (args.num_shards is None and args.shard_index is None) or
            (args.num_shards is not None and args.shard_index is not None)
    )

    if args.num_shards is not None:
        total = len(samples)
        shard_size = total // args.num_shards
        start = args.shard_index * shard_size
        end = start + shard_size if args.shard_index < args.num_shards - 1 else total
        samples = samples[start:end]
        print(f"Shard {args.shard_index}/{args.num_shards}, samples [{start}:{end}), count={len(samples)}")

    # ---- resume ----
    save_path = f"../location_train_preview100/stage4_foreground_grounding/{edit_type}_{args.num_shards}_{args.shard_index}.jsonl"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    done_ids = set()
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            for line in f:
                if line.strip():
                    done_ids.add(json.loads(line)['id'])
        print(f"Resuming: {len(done_ids)} already done, skipping.")

    samples = [s for s in samples if s['id'] not in done_ids]
    print(f"Remaining: {len(samples)} samples to process.")

    # ---- skip samples with no labels (nothing to track) ----
    samples_with_labels = [s for s in samples if s.get('round1_labels')]
    print(
        f"  {len(samples_with_labels)} have labels to track, {len(samples) - len(samples_with_labels)} skipped (empty labels).")

    # ---- model ----
    model_name = "Qwen/Qwen3-VL-32B-Instruct"
    print(f"Loading {model_name}...")

    llm = LLM(
        model=model_name,
        tensor_parallel_size=8,
        mm_encoder_tp_mode="data",
        enable_expert_parallel=False,
        async_scheduling=True,
        limit_mm_per_prompt={"video": 0, "image": 1},  # Round 2: 1 image per request
        media_io_kwargs={"video": {"num_frames": -1}},
        disable_log_stats=True,
    )

    sampling_params = SamplingParams(max_tokens=8192, temperature=0)

    stats = {"ok": 0, "skip_video": 0, "skip_infer": 0}

    with open(save_path, 'a') as f:
        for item in tqdm(samples_with_labels, desc=f"{edit_type} R2 shard={args.shard_index}"):
            sample_id = item['id']
            labels = item['round1_labels']

            # ---- download video & sample 2FPS ----
            try:
                source_frames_b64 = read_video_2fps(os.path.join("../location_train_preview100", item['source_path']))
                if not source_frames_b64:
                    stats["skip_video"] += 1
                    continue
            except Exception as e:
                print(f"Video error on {sample_id}: {e}")
                stats["skip_video"] += 1
                continue

            # ---- build tracking messages for all frames ----
            try:
                input_text = f"Label List: {json.dumps(labels)}"
                tracking_messages = []
                for frame_b64 in source_frames_b64:
                    msg = [
                        {"role": "system", "content": SYSTEM_PROMPT_TRACKING},
                        {"role": "user", "content": [
                            {"type": "image_url", "image_url": {"url": frame_b64}},
                            {"type": "text", "text": input_text},
                        ]}
                    ]
                    tracking_messages.append(msg)

                outputs_r2 = llm.chat(
                    messages=tracking_messages,
                    sampling_params=sampling_params,
                    chat_template_content_format="openai",
                    use_tqdm=False,
                )

                all_frames_objects = []
                for out in outputs_r2:
                    resp_txt = out.outputs[0].text.strip()
                    parsed_objs = json.loads(resp_txt)
                    if not isinstance(parsed_objs, list):
                        raise ValueError(f"Round 2 output is not a JSON list: {resp_txt}")
                    all_frames_objects.append(parsed_objs)

            except Exception as e:
                print(f"Inference/Parsing error on {sample_id}: {e}")
                stats["skip_infer"] += 1
                continue

            # ---- write result ----
            ret = {
                "id": sample_id,
                "source_path": item['source_path'],
                "prompt": item['prompt'],
                "edit_type": edit_type,
                "round1_labels": labels,
                "round1_objects": item.get('round1_objects', []),
                "frame_objects": all_frames_objects,
            }

            f.write(json.dumps(ret) + '\n')
            f.flush()
            stats["ok"] += 1

    print(f"Done. Stats: {json.dumps(stats)}")
