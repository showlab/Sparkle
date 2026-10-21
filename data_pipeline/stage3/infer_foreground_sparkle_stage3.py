import time
import base64
import os
import json
import argparse
from io import BytesIO

from vllm import LLM, SamplingParams
from tqdm import tqdm
from PIL import Image
import decord
import pandas as pd


def frame_to_base64(frame_np):
    """Convert a single numpy frame to base64 data URL."""
    img = Image.fromarray(frame_np)
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_base64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{img_base64_str}"


def image_file_to_base64(path):
    """Read a local image file and return base64 data URL."""
    img = Image.open(path).convert("RGB")
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_base64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{img_base64_str}"

# ==============================================================================
# Round 1 Prompt: Compare source vs edited frame 0 to identify foreground
# ==============================================================================
SYSTEM_PROMPT_ROUND_1 = """
You are an expert visual grounding analyzer and object detector. Your goal is to identify and provide bounding boxes for specific foreground entities in the SOURCE image that must remain visually identical when a given editing instruction is applied.

### Context
You will be given:
1. **Source Image**: The original starting scene (first image).
2. **Edited Image**: The result after applying the editing instruction (second image).
3. **Instruction**: A text prompt describing the intended edit.

By comparing the two images, you can precisely determine which objects were preserved (foreground) and which were changed (background/edited region).

### Definitions
- **Foreground Objects**: Distinct entities (e.g., people, animals, vehicles, handheld items) that are visually identical between the source and edited images — they were NOT modified by the edit.
- **Overlaid Elements**: Any text overlays, titles, subtitles, watermarks, or logos that appear on top of the scene. These are considered foreground and MUST be detected.
- **Physical Supports (Contextual Anchors)**: Objects that physically support or interact with the foreground objects (e.g., a chair a person is sitting on, a table holding a book). These must also be preserved to maintain physical plausibility.
- **Background / Edited Region**: Parts of the scene that differ between source and edited images — these are what the edit changed and should NOT be included.

### Task
1. Compare the source and edited images side by side.
2. Identify all distinct entities that remain visually unchanged between the two images.
3. Also detect any overlaid text, titles, subtitles, watermarks, or logos visible in the source image.
4. For each preserved entity and overlay, detect its bounding box in the SOURCE image and generate an ultra-brief label.
5. Output the final result as a strict JSON array of objects.

### Constraints
- **Preserve Physical Logic (CRITICAL)**: If a foreground object is preserved, you MUST also preserve its physical support to prevent physically impossible scenes (e.g., a floating person), UNLESS the editing instruction explicitly targets the supporting object.
- **Detect Overlays (CRITICAL)**: Any text, title, subtitle, watermark, or logo overlaid on the scene MUST be detected as foreground, even if it partially occludes other objects.
- **Strict Output Format**: Output ONLY a valid JSON array of objects. No preamble, no postamble, no markdown code block formatting.
- **JSON Schema**: Each object must strictly follow: `{"bbox_2d": [x1, y1, x2, y2], "label": "string"}`
- **BBox Coordinate Format**: `[top-left x, top-left y, bottom-right x, bottom-right y]`
- **Ultra-Concise Labels**: 2-6 words max. Core noun + 1-2 key features. Omit articles ("The", "A").
- **ABSOLUTE NO PRONOUNS (CRITICAL)**: NEVER use pronouns (his, her, its, their). Each label must be self-contained. Use explicit references (e.g., "Man's white shirt" not "His shirt", or just "White button-up shirt").
- **Negative Case**: If the instruction alters everything, or no foreground objects are preserved, respond with `[]`.

### Example Output
[
    {"bbox_2d": [134, 746, 308, 925], "label": "small cake"},
    {"bbox_2d": [150, 700, 350, 950], "label": "wooden coffee table"},
    {"bbox_2d": [10, 20, 400, 60], "label": "white title text overlay"},
    {"bbox_2d": [350, 658, 517, 821], "label": "silver fork"}
]
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--edit_type", type=str, required=True)
    parser.add_argument("--num_shards", type=int, default=None)
    parser.add_argument("--shard_index", type=int, default=None)
    args = parser.parse_args()

    edit_type = args.edit_type
    csv_path = f"../location_train_preview100/stage2_generated_prompts_{edit_type}_filtered.csv"
    source_frame0_dir = f"../location_train_preview100/stage2_source_frame0_{edit_type}"
    edited_frame0_dir = f"../location_train_preview100/stage2_edited_frame0_{edit_type}"

    print(f"Loading {csv_path} ...")
    samples = pd.read_csv(csv_path).to_dict("records")
    print(f"Loaded {len(samples)} samples.")

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
    save_path = f"../location_train_preview100/stage3_foreground_grounding/{edit_type}_{args.num_shards}_{args.shard_index}.jsonl"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    done_ids = set()
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            for line in f:
                if line.strip():
                    done_ids.add(json.loads(line)['id'])
        print(f"Resuming: {len(done_ids)} already done, skipping.")

    samples = [s for s in samples if s['task'] not in done_ids]
    print(f"Remaining: {len(samples)} samples to process.")

    # ---- model ----
    model_name = "Qwen/Qwen3-VL-32B-Instruct"
    print(f"Loading {model_name}...")

    llm = LLM(
        model=model_name,
        tensor_parallel_size=8,
        mm_encoder_tp_mode="data",
        enable_expert_parallel=False,
        async_scheduling=True,
        limit_mm_per_prompt={"video": 0, "image": 2},
        media_io_kwargs={"video": {"num_frames": -1}},
        disable_log_stats=True,
    )

    sampling_params = SamplingParams(max_tokens=8192, temperature=0)

    stats = {"ok": 0, "skip_source": 0, "skip_edited": 0, "skip_infer": 0}

    with open(save_path, 'a') as f:
        for item in tqdm(samples, desc=f"{edit_type} Stage3 shard={args.shard_index}"):
            sample_id = item['task']

            # ---- load source frame 0 ----
            source_frame0_path = os.path.join(source_frame0_dir, f"{sample_id}.png")
            if not os.path.exists(source_frame0_path):
                print(f"Missing source_frame0 for {sample_id}: {source_frame0_path}")
                stats["skip_source"] += 1
                continue

            try:
                source_frame0_b64 = image_file_to_base64(source_frame0_path)
            except Exception as e:
                print(f"Failed to read source_frame0 for {sample_id}: {e}")
                stats["skip_source"] += 1
                continue

            # ---- load edited frame 0 ----
            edited_frame0_path = os.path.join(edited_frame0_dir, f"{sample_id}.png")
            if not os.path.exists(edited_frame0_path):
                print(f"Missing edited_frame0 for {sample_id}: {edited_frame0_path}")
                stats["skip_edited"] += 1
                continue

            try:
                edited_frame0_b64 = image_file_to_base64(edited_frame0_path)
            except Exception as e:
                print(f"Failed to read edited_frame0 for {sample_id}: {e}")
                stats["skip_edited"] += 1
                continue

            # ---- Round 1 inference ----
            try:
                messages_r1 = [
                    {"role": "system", "content": SYSTEM_PROMPT_ROUND_1},
                    {"role": "user", "content": [
                        {"type": "text", "text": "Source Image:"},
                        {"type": "image_url", "image_url": {"url": source_frame0_b64}},
                        {"type": "text", "text": "Edited Image:"},
                        {"type": "image_url", "image_url": {"url": edited_frame0_b64}},
                        {"type": "text", "text": f"Instruction: {item['prompt']}"},
                    ]}
                ]

                outputs_r1 = llm.chat(
                    messages=[messages_r1],
                    sampling_params=sampling_params,
                    chat_template_content_format="openai",
                    use_tqdm=False,
                )

                resp_text_r1 = outputs_r1[0].outputs[0].text.strip()
                parsed_r1 = json.loads(resp_text_r1)

                if not isinstance(parsed_r1, list):
                    raise ValueError(f"Round 1 output is not a JSON list: {resp_text_r1}")

                labels = [obj["label"] for obj in parsed_r1 if "label" in obj]

            except Exception as e:
                print(f"Inference/Parsing error on {sample_id}: {e}")
                stats["skip_infer"] += 1
                continue

            # ---- write result ----
            ret = {
                "id": sample_id,
                "source_path": item['src_video'],
                "prompt": item['prompt'],
                "edit_type": edit_type,
                "round1_labels": labels,
                "round1_objects": parsed_r1,
            }

            f.write(json.dumps(ret) + '\n')
            f.flush()
            stats["ok"] += 1

    print(f"Done. Stats: {json.dumps(stats)}")
