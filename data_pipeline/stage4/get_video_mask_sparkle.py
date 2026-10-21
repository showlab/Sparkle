from accelerate import Accelerator
import torch
import numpy as np
import os
import argparse
from io import BytesIO
import json
from tqdm import tqdm
from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor
from decord import VideoReader, cpu
import math
import pandas as pd

def get_video_mask(video_path: str, bbox_prompts: dict, model, processor, device):

    vr = VideoReader(video_path, ctx=cpu(0))
    fps = vr.get_avg_fps()
    video_frames = vr.get_batch(range(len(vr))).asnumpy()

    video_height, video_width = video_frames.shape[1], video_frames.shape[2]

    frame_interval = max(1, int(round(fps / 2.0)))

    frame_objects_list = bbox_prompts.get("frame_objects", [])

    target_anchor_indices = list(range(len(frame_objects_list)))

    valid_anchor_indices = [idx for idx in target_anchor_indices if idx < len(frame_objects_list)]
    if not valid_anchor_indices:
        return None

    vote_threshold = math.ceil(len(valid_anchor_indices) * 0.5)

    all_runs_masks = {idx: [] for idx in range(len(video_frames))}

    valid_runs_count = 0

    for target_idx in valid_anchor_indices:
        target_frame_idx = target_idx * frame_interval

        if target_frame_idx >= len(video_frames):
            continue

        frame_objs = frame_objects_list[target_idx]

        obj_ids = []
        input_boxes = [[]]

        for idx, obj in enumerate(frame_objs):
            bbox = obj.get("bbox_2d", [])
            if not bbox or len(bbox) == 0:
                continue

            x1 = int(bbox[0] / 1000.0 * video_width)
            y1 = int(bbox[1] / 1000.0 * video_height)
            x2 = int(bbox[2] / 1000.0 * video_width)
            y2 = int(bbox[3] / 1000.0 * video_height)

            x1 = max(0, min(min(x1, x2), video_width - 1))
            y1 = max(0, min(min(y1, y2), video_height - 1))
            x2 = max(0, min(max(x1, x2), video_width - 1))
            y2 = max(0, min(max(y1, y2), video_height - 1))

            obj_ids.append(idx + 1)
            input_boxes[0].append([x1, y1, x2, y2])

        if not obj_ids:
            continue

        fwd_obj_ids = obj_ids.copy()
        bwd_obj_ids = obj_ids.copy()

        forward_session = processor.init_video_session(video=video_frames, inference_device=device,
                                                       dtype=torch.bfloat16)
        backward_session = processor.init_video_session(video=video_frames, inference_device=device,
                                                        dtype=torch.bfloat16)

        processor.add_inputs_to_inference_session(forward_session, frame_idx=target_frame_idx, obj_ids=fwd_obj_ids,
                                                  input_boxes=input_boxes)
        model(inference_session=forward_session, frame_idx=target_frame_idx)

        processor.add_inputs_to_inference_session(backward_session, frame_idx=target_frame_idx, obj_ids=bwd_obj_ids,
                                                  input_boxes=input_boxes)
        model(inference_session=backward_session, frame_idx=target_frame_idx)

        current_run_masks = {}
        valid_runs_count += 1

        for out in model.propagate_in_video_iterator(forward_session, reverse=False):
            video_res_masks = processor.post_process_masks(
                [out.pred_masks], original_sizes=[[forward_session.video_height, forward_session.video_width]],
                binarize=False
            )[0]
            masks = (video_res_masks > 0.0).cpu().numpy()

            if masks.ndim == 4:
                masks = masks[:, 0, :, :]
            elif masks.ndim == 3:
                pass
            else:
                raise ValueError(f"Unexpected mask dimension: {masks.shape}")

            current_run_masks[out.frame_idx] = np.any(masks, axis=0)

        for out in model.propagate_in_video_iterator(backward_session, reverse=True):
            video_res_masks = processor.post_process_masks(
                [out.pred_masks], original_sizes=[[backward_session.video_height, backward_session.video_width]],
                binarize=False
            )[0]
            masks = (video_res_masks > 0.0).cpu().numpy()

            if masks.ndim == 4:
                masks = masks[:, 0, :, :]
            elif masks.ndim == 3:
                pass
            else:
                raise ValueError(f"Unexpected mask dimension: {masks.shape}")

            current_run_masks[out.frame_idx] = np.any(masks, axis=0)

        for f_idx, aggregated_mask in current_run_masks.items():
            all_runs_masks[f_idx].append(aggregated_mask)

    if valid_runs_count == 0:
        return None

    final_video_masks = []

    for frame_idx in range(len(video_frames)):
        masks_list = all_runs_masks[frame_idx]

        if len(masks_list) > 0:
            stacked_masks = np.stack(masks_list, axis=0)
            voted_mask = np.sum(stacked_masks, axis=0) >= vote_threshold
        else:
            voted_mask = np.zeros((video_height, video_width), dtype=bool)

        final_video_masks.append(voted_mask)

    return np.stack(final_video_masks, axis=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_shards", type=int, default=None)
    parser.add_argument("--shard_index", type=int, default=None)
    parser.add_argument("--edit_type", type=str, required=True)
    args = parser.parse_args()

    edit_type = args.edit_type

    # ──────────────────────────────────────────────
    # 1. Load prompts metadata (for source_path + id list)
    # ──────────────────────────────────────────────
    prompts_file = f"../location_train_preview100/stage3_generated_prompts_{edit_type}_filtered.csv"
    print(f"Loading prompts from {prompts_file}...")

    df = pd.read_csv(prompts_file)
    prompts_data = {item['task']: item for item in df.to_dict(orient='records')}

    print(f"Loaded {len(prompts_data)} prompts.")

    # ──────────────────────────────────────────────
    # 2. Load foreground grounding bbox info
    # ──────────────────────────────────────────────
    grounding_file = f"../location_train_preview100/stage4_foreground_grounding_{edit_type}_gathered.jsonl"
    print(f"Loading grounding from {grounding_file}...")
    grounding_data = {}
    with open(grounding_file, 'r') as f:
        for line in f:
            item = json.loads(line)
            grounding_data[item['id']] = item
    print(f"Loaded {len(grounding_data)} grounding entries.")

    # ──────────────────────────────────────────────
    # 3. Build dataset: only keep ids present in both
    # ──────────────────────────────────────────────
    common_ids = sorted(set(prompts_data.keys()) & set(grounding_data.keys()))
    dataset = []
    for cid in common_ids:
        entry = grounding_data[cid]  # has frame_objects for SAM3
        entry['source_path'] = prompts_data[cid]['src_video']
        dataset.append(entry)
    print(f"Matched dataset: {len(dataset)} samples.")

    # ──────────────────────────────────────────────
    # 4. Sharding
    # ──────────────────────────────────────────────
    assert (
            (args.num_shards is None and args.shard_index is None) or
            (args.num_shards is not None and args.shard_index is not None)
    )

    if args.num_shards is not None:
        total = len(dataset)
        shard_size = total // args.num_shards
        remainder = total % args.num_shards
        start = args.shard_index * shard_size + min(args.shard_index, remainder)
        end = start + shard_size + (1 if args.shard_index < remainder else 0)
        dataset = dataset[start:end]
        print(f"Shard {args.shard_index}/{args.num_shards}: {len(dataset)} samples (indices {start}-{end - 1}).")

    mask_save_dir = f"../location_train_preview100/stage4_source_video_mask_{edit_type}"
    os.makedirs(mask_save_dir, exist_ok=True)

    # ──────────────────────────────────────────────
    # 5. Init model
    # ──────────────────────────────────────────────
    print('Initializing SAM3 Model...')
    device = Accelerator().device
    model = Sam3TrackerVideoModel.from_pretrained("facebook/sam3").to(device, dtype=torch.bfloat16)
    processor = Sam3TrackerVideoProcessor.from_pretrained("facebook/sam3")

    # ──────────────────────────────────────────────
    # 6. Processing loop
    # ──────────────────────────────────────────────
    ok, fail = 0, 0
    for item in tqdm(dataset, desc="Processing videos"):
        video_id = item['id']
        save_path = os.path.join(mask_save_dir, f"{video_id}.npz")

        if os.path.exists(save_path):
            ok += 1
            continue

        try:
            video_mask = get_video_mask(
                video_path=os.path.join("../location_train_preview100", item["source_path"]),
                bbox_prompts=item,
                model=model,
                processor=processor,
                device=device
            )

            if video_mask is not None:
                packed_mask = np.packbits(video_mask)
                np.savez_compressed(save_path, mask=packed_mask, shape=video_mask.shape)
                ok += 1
            else:
                fail += 1

        except Exception as e:
            print(f"Error processing video {video_id}: {e}")
            fail += 1
            continue
        finally:
            torch.cuda.empty_cache()

        tqdm.write(f"ok={ok} fail={fail}")

    print(f"Done. ok={ok} fail={fail}")
