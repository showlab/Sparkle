import argparse
import json
import os
import time
import pandas as pd
from huggingface_hub import snapshot_download

from tqdm import tqdm

from lightx2v import LightX2VPipeline

INSTRUCTION_STARTERS = [
    "Replace the background with",
    "Change the background to",
    "Transform the background into",
    "Place the subject in",
    "Set the scene to",
    "Swap the background to",
    "Move the subject to",
    "Transport the subject to",
    "Shift the background to",
    "Put the subject against",
]

# PROMPT_PREFIX = "A scene filmed from a stationary tripod-mounted camera with zero pan, tilt, or zoom."
PROMPT_PREFIX = (
    "A scene filmed from a single fixed viewpoint that never changes. "
    "The framing, angle, and field of view remain perfectly constant throughout. "
    "The horizon line stays at the same position. "
    "Only natural elements within the scene may move."
)

NEG_CAM = "camera motion, camera shake, panning, tilting, zooming, dolly, tracking shot, handheld, perspective change, dolly in, dolly out, truck left, truck right, pedestal up, pedestal down"
NEG_FREEZE = "motionless, static scene, still image"
NEG_QUALITY = "logo，watermark, text, subtitle, distort, low quality, 色调艳丽，过曝，静态，细节模糊不清，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，倒着走"
NEGATIVE_PROMPT = ", ".join([NEG_CAM, NEG_FREEZE, NEG_QUALITY])


def strip_instruction_starter(prompt: str) -> str:
    for starter in INSTRUCTION_STARTERS:
        if prompt.startswith(starter):
            prompt = prompt[len(starter):].strip()
            # Handle possible leading punctuation/whitespace after stripping
            if prompt and prompt[0] in (",", ".", ";", ":"):
                prompt = prompt[1:].strip()
            break
    return prompt


def load_and_shard(csv_path: str, num_shards: int, shard_index: int):
    samples=pd.read_csv(csv_path).to_dict(orient='records')

    # Contiguous sharding
    total = len(samples)
    shard_size = total // num_shards
    remainder = total % num_shards
    if shard_index < remainder:
        start = shard_index * (shard_size + 1)
        end = start + shard_size + 1
    else:
        start = shard_index * shard_size + remainder
        end = start + shard_size
    return samples[start:end]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--edit_type", type=str, required=True)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    args = parser.parse_args()

    data_root = "../../location_train_preview100"
    csv_path = f"{data_root}/stage3_generated_prompts_{args.edit_type}_filtered.csv"
    image_dir = f"{data_root}/stage3_edited_frame0_{args.edit_type}"
    output_dir = f"{data_root}/stage3_edited_background_video_{args.edit_type}"
    os.makedirs(output_dir, exist_ok=True)

    model_path = snapshot_download(
        repo_id="Wan-AI/Wan2.2-I2V-A14B",
        local_files_only=True,
    )

    # Load and shard
    samples = load_and_shard(csv_path, args.num_shards, args.shard_index)
    print(f"[Shard {args.shard_index}/{args.num_shards}] Total samples in shard: {len(samples)}")

    # Resume: skip already-processed
    existing = set()
    for fname in os.listdir(output_dir):
        if fname.endswith(".mp4"):
            existing.add(fname[:-4])  # strip .mp4
    before = len(samples)
    samples = [s for s in samples if s["task"] not in existing]
    print(f"Skipped {before - len(samples)} already-processed, {len(samples)} remaining")

    if not samples:
        print("Nothing to do.")
        exit(0)

    # Init pipeline
    pipe = LightX2VPipeline(
        model_path=model_path,
        model_cls="wan2.2_moe_distill",
        task="i2v",
        low_noise_original_ckpt="./wan2.2_i2v_A14b_low_noise_lightx2v_4step.safetensors",
        high_noise_original_ckpt="./wan2.2_i2v_A14b_high_noise_lightx2v_4step_1030.safetensors",
    )

    pipe.enable_parallel(
        seq_p_size=8,
        seq_p_attn_type="ulysses",
    )

    pipe.create_generator(
        attn_mode="sage_attn2",
        infer_steps=4,
        height=720,
        width=1280,
        num_frames=81,
        guidance_scale=1,
        sample_shift=5.0,
    )

    seed = 42
    success, fail = 0, 0
    pbar = tqdm(samples, desc=f"I2V shard={args.shard_index}")

    for sample in pbar:
        sample_id = sample["task"]
        image_path = os.path.join(image_dir, f"{sample_id}.png")
        save_path = os.path.join(output_dir, f"{sample_id}.mp4")

        try:
            raw_prompt = sample["prompt"]
            scene_desc = strip_instruction_starter(raw_prompt)

            prompt = f"{PROMPT_PREFIX} {scene_desc}"

            # print("Prompt: ", prompt)

            pipe.generate(
                seed=seed,
                image_path=image_path,
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                save_result_path=save_path,
            )
            success += 1
        except Exception as e:
            fail += 1
            print(f"[ERROR] {sample_id}: {e}")

        pbar.set_postfix(ok=success, fail=fail)

    print(f"Done. success={success}, fail={fail}")
