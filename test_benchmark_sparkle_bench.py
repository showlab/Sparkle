import os
import csv
import json
import argparse

import torch
from PIL import Image
from tqdm import tqdm

from diffsynth import VideoData, save_video
from diffsynth.pipelines.wan_video_mllm import WanVideoPipeline, ModelConfig

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ================= Video helpers =================
def concat_video(video1, video2, bg=(0, 0, 0)):
    cat_video = []
    for img1, img2 in zip(video1, video2):
        w1, h1 = img1.size
        w2, h2 = img2.size
        H = max(h1, h2)
        W = w1 + w2
        canvas = Image.new("RGB", (W, H), bg)
        canvas.paste(img1, (0, 0))
        canvas.paste(img2, (w1, 0))
        cat_video.append(canvas)
    return cat_video


# ================= Model =================
def model_init(device, ckpt_path, ref_pad_first=False):
    if '14b' in ckpt_path:
        pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=[
                ModelConfig(model_id="Wan-AI/Wan2.1-T2V-14B",
                            origin_file_pattern="diffusion_pytorch_model*.safetensors"),
                ModelConfig(model_id="Wan-AI/Wan2.1-T2V-14B", origin_file_pattern="Wan2.1_VAE.pth"),
            ],
            mllm_max_frame=10,
            mllm_max_pixels_per_frame=262144,
            max_object_token=384,
            num_ref_queries=384,
            ref_pad_first=ref_pad_first,
        )
    else:
        pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=[
                ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B",
                            origin_file_pattern="diffusion_pytorch_model*.safetensors"),
                ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="Wan2.2_VAE.pth"),
            ],
            mllm_model="Qwen/Qwen2.5-VL-3B-Instruct",
            ref_pad_first=ref_pad_first,
        )
    pipe.mllm.eval()
    pipe.load_lora(pipe, ckpt_path, alpha=1)
    pipe.to(torch.bfloat16)
    pipe.to(device)
    return pipe


# ================= Bench =================
def test_sparkle_bench(
        pipe,
        save_dir,
        edit_type,
        bench_root="data/Sparkle-Bench",
        max_frame=81,
        max_pixels=600 * 600,
        cfg_scale=5,
        num_rank=1,
        rank=0,
        **kwargs):
    os.makedirs(save_dir, exist_ok=True)

    bench_csv = f"{bench_root}/{edit_type}_bench.csv"
    meta_jsonl = f"{bench_root}/{edit_type}_metadata.jsonl"

    # Build id -> chosen_keyword lookup from metadata jsonl
    id2keyword = {}
    with open(meta_jsonl, "r") as f:
        for line in f:
            rec = json.loads(line)
            id2keyword[rec["id"]] = rec["metadata"]["chosen_keyword"]

    # Read bench csv: prompt + original_video
    with open(bench_csv, "r") as f:
        rows = list(csv.DictReader(f))
    print(f"Total items for {edit_type}: {len(rows)}")

    n_done, n_skip, n_fail = 0, 0, 0
    pbar = tqdm(list(enumerate(rows)))
    for idx, row in pbar:
        if num_rank > 1 and idx % num_rank != rank:
            continue

        prompt = row["prompt"]
        original_video = row["original_video"]  # e.g. source_videos/location/Sparkle_location_010913.mp4
        item_id = os.path.splitext(os.path.basename(original_video))[0]
        source_path = os.path.join(bench_root, original_video)

        chosen_keyword = id2keyword.get(item_id)
        if chosen_keyword is None:
            print(f"[skip] missing metadata for {item_id}")
            n_fail += 1
            pbar.set_postfix(done=n_done, skip=n_skip, fail=n_fail)
            continue

        subtheme, scene = chosen_keyword.split(": ")
        scene_key = scene.replace(" ", "_")

        out_subdir = f"{save_dir}/{subtheme}---{scene_key}"
        os.makedirs(out_subdir, exist_ok=True)
        out_path = f"{out_subdir}/{item_id}_edited.mp4"
        concat_path = f"{out_subdir}/{item_id}_concat.mp4"
        if os.path.exists(out_path):
            n_skip += 1
            pbar.set_postfix(done=n_done, skip=n_skip, fail=n_fail)
            continue

        if not os.path.exists(source_path):
            print(f"[skip] missing source video: {source_path}")
            n_fail += 1
            pbar.set_postfix(done=n_done, skip=n_skip, fail=n_fail)
            continue

        src_video = VideoData(source_path, length=max_frame, max_pixels=max_pixels)
        print(prompt, source_path, src_video.height, src_video.width)

        video = pipe(
            prompt=prompt,
            source_input=src_video,
            src_video=src_video,
            src_mask=None,
            height=src_video.height,
            width=src_video.width,
            num_frames=min(max_frame, len(src_video)),
            seed=0,
            cfg_scale=cfg_scale,
            tiled=True,
        )
        full_video = concat_video(src_video, video)
        save_video(video, out_path, fps=15, quality=5)
        save_video(full_video, concat_path, fps=15, quality=5)
        n_done += 1
        pbar.set_postfix(done=n_done, skip=n_skip, fail=n_fail)


# ================= Main =================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--num_rank', type=int, default=1)
    parser.add_argument('--edit_type', type=str, default='location')
    parser.add_argument('--save_dir', type=str, default='')
    parser.add_argument('--max_frame', type=int, default=81)
    parser.add_argument('--max_pixels', type=int, default=720 * 1280)
    parser.add_argument('--cfg_scale', type=int, default=5)
    parser.add_argument('--ref_pad_first', type=bool, default=False)
    parser.add_argument('--bench_root', type=str, default='data/Sparkle-Bench')
    args = parser.parse_args()

    save_dir = os.path.join(args.save_dir, f'sparkle_bench/{args.edit_type}/')

    pipe = model_init(device=f"cuda:{args.rank}", ckpt_path=args.ckpt_path,
                      ref_pad_first=args.ref_pad_first)
    args.rank = args.rank % args.num_rank

    test_sparkle_bench(
        pipe,
        save_dir=save_dir,
        edit_type=args.edit_type,
        bench_root=args.bench_root,
        max_frame=args.max_frame,
        max_pixels=args.max_pixels,
        cfg_scale=args.cfg_scale,
        rank=args.rank,
        num_rank=args.num_rank,
    )
