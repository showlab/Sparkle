import json
import argparse
import numpy as np
import os
import cv2
from io import BytesIO
from PIL import Image
import decord
import imageio.v3 as iio
from controlnet_aux import LineartDetector
from tqdm import tqdm
import pandas as pd


def read_video_from_path(video_path: str) -> tuple:
    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    fps = vr.get_avg_fps()
    frames = vr.get_batch(range(len(vr))).asnumpy()
    return frames, fps


def load_mask(mask_path: str) -> np.ndarray:
    data = np.load(mask_path)
    packed_mask = data['mask']
    shape = tuple(int(s) for s in data['shape'])
    total = shape[0] * shape[1] * shape[2]
    video_mask = np.unpackbits(packed_mask)[:total].reshape(shape).astype(bool)
    return video_mask


def save_video(frames: np.ndarray, save_path: str, fps: float = 24.0):
    iio.imwrite(
        save_path,
        frames,
        fps=int(round(fps)),
        codec="h264",
        plugin="pyav",
    )


def compute_edge_video(frames: np.ndarray, detector) -> np.ndarray:
    T, H, W, _ = frames.shape
    edges = np.empty((T, H, W), dtype=np.uint8)
    for t in range(T):
        pil_img = Image.fromarray(frames[t])
        edge = detector(pil_img)
        edge_np = np.array(edge.convert("L"))
        if edge_np.shape[0] != H or edge_np.shape[1] != W:
            edge_np = cv2.resize(edge_np, (W, H), interpolation=cv2.INTER_LINEAR)
        edges[t] = edge_np
    return edges


def mix_edge_with_mask(fg_edge: np.ndarray, bg_edge: np.ndarray,
                       mask: np.ndarray) -> np.ndarray:
    mixed = np.where(mask, fg_edge, bg_edge)
    return np.stack([mixed, mixed, mixed], axis=-1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--edit_type", type=str, required=True)
    parser.add_argument("--num_shards", type=int, default=None)
    parser.add_argument("--shard_index", type=int, default=None)
    return parser.parse_args()


# ================= Main =================
if __name__ == '__main__':
    args = parse_args()
    edit_type = args.edit_type

    metadata_csv = f'../location_train_preview100/stage3_generated_prompts_{edit_type}_filtered.csv'
    bg_video_dir = f'../location_train_preview100/stage3_edited_background_video_{edit_type}'
    mask_dir = f'../location_train_preview100/stage4_source_video_mask_{edit_type}'
    output_dir = f'../location_train_preview100/stage5_edited_video_canny_{edit_type}'

    os.makedirs(output_dir, exist_ok=True)

    lineart_detector = LineartDetector.from_pretrained("lllyasviel/Annotators").to("cuda")

    # ---- 1. Gather all data into video_info ----
    video_info = {}

    df = pd.read_csv(metadata_csv)

    for data in tqdm(df.to_dict(orient='records'), desc='Loading metadata'):
        vid = data["task"]
        source_path = os.path.join(f'../location_train_preview100/', data["src_video"])
        bg_path = os.path.join(bg_video_dir, f"{vid}.mp4")
        mask_path = os.path.join(mask_dir, f"{vid}.npz")

        if not os.path.exists(bg_path) or not os.path.exists(mask_path):
            continue

        video_info[vid] = {
            "source_path": source_path,
            "background_video": bg_path,
            "mask_path": mask_path,
        }

    # ---- 2. Sort & shard ----
    valid_ids = sorted(video_info.keys())
    print(f"Total valid samples: {len(valid_ids)}")

    if args.num_shards is not None and args.shard_index is not None:
        total = len(valid_ids)
        shard_size = total // args.num_shards
        remainder = total % args.num_shards
        if args.shard_index < remainder:
            start = args.shard_index * (shard_size + 1)
            end = start + shard_size + 1
        else:
            start = remainder * (shard_size + 1) + (args.shard_index - remainder) * shard_size
            end = start + shard_size
        valid_ids = valid_ids[start:end]
        print(
            f"Shard {args.shard_index}/{args.num_shards}: processing {len(valid_ids)} samples (indices {start}-{end - 1})")

    # ---- 3. Process ----
    success, fail = 0, 0
    pbar = tqdm(valid_ids, desc="Processing videos")

    for vid in pbar:
        info = video_info[vid]
        canny_save_path = os.path.join(output_dir, f"{vid}.mp4")

        if os.path.exists(canny_save_path):
            success += 1
            pbar.set_postfix(success=success, fail=fail)
            continue

        try:
            source_frames, source_fps = read_video_from_path(info["source_path"])

            mask = load_mask(info["mask_path"])
            bg_frames, _ = read_video_from_path(info["background_video"])

            T_src, H_src, W_src, _ = source_frames.shape
            T_bg, H_bg, W_bg, _ = bg_frames.shape

            assert mask.shape == (T_src, H_src, W_src), \
                f"Mask shape {mask.shape} != source {(T_src, H_src, W_src)}"

            if H_bg != H_src or W_bg != W_src:
                raise ValueError(
                    f"Background resolution ({H_bg}, {W_bg}) != source ({H_src}, {W_src})"
                )

            T = min(T_src, T_bg)
            source_frames = source_frames[:T]
            mask = mask[:T]
            bg_frames = bg_frames[:T]

            fg_edge = compute_edge_video(source_frames, lineart_detector)
            bg_edge = compute_edge_video(bg_frames, lineart_detector)
            mixed_edge = mix_edge_with_mask(fg_edge, bg_edge, mask)

            save_video(mixed_edge, canny_save_path, fps=source_fps)

            success += 1

        except Exception as e:
            print(f"[Error] {vid}: {e}")
            fail += 1
            continue

        pbar.set_postfix(success=success, fail=fail)

    print(f"\nDone. success={success}, fail={fail}")
