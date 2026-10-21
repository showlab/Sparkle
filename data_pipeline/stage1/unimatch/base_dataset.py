import torch
import os
import sys
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from io import BytesIO
from datasets import load_dataset, Features, Value, Sequence
from typing import Dict, Union, List, Any
from tqdm import tqdm
import decord
import json
import pandas as pd


class BaseVideoDataset(Dataset):
    def __init__(
        self,
        csv_path,
        video_root,
        num_shards=None,
        shard_index=None,
    ):
        self.video_root = video_root

        self.data = pd.read_csv(csv_path, dtype=str, keep_default_na=False).to_dict(
            orient="records"
        )

        assert (num_shards is None and shard_index is None) or (
            num_shards is not None and shard_index is not None
        )

        if num_shards is not None:
            # contiguous sharding, 对齐 HuggingFace `shard(contiguous=True)`
            n = len(self.data)
            div, mod = divmod(n, num_shards)
            start = div * shard_index + min(shard_index, mod)
            end = start + div + (1 if shard_index < mod else 0)
            self.data = self.data[start:end]
            print(
                f"Current eval shard: {shard_index} / {num_shards}, it has {len(self)} samples."
            )

    @staticmethod
    def read_media(
        obj: Union[BytesIO, str],
        ext=".mp4",
        video_backend="decord",
        target_fps=2,
        target_long_edge=576,
        target_short_edge=320,
    ) -> np.ndarray:

        assert isinstance(obj, BytesIO) or isinstance(
            obj, str
        ), "obj must be BytesIO or str type"

        # 确保一开始指针在头部
        if isinstance(obj, BytesIO):
            obj.seek(0)

        ext = ext.lower()

        if ext in [".mp4", ".avi", ".mov", ".mkv"]:
            try:
                if video_backend == "decord":
                    # ==========================================
                    # 第一步：轻量级探测原始分辨率
                    # ==========================================
                    probe_vr = decord.VideoReader(obj)

                    H_ori, W_ori, _ = probe_vr[0].shape
                    original_fps = probe_vr.get_avg_fps()
                    total_frames = len(probe_vr)

                    step = original_fps / target_fps
                    if step >= 1.0:
                        indices = np.arange(0, total_frames, step).astype(int)
                    else:
                        indices = np.arange(total_frames)
                    indices = np.clip(indices, 0, total_frames - 1)

                    del probe_vr  # 释放探测器

                    # ==========================================
                    # 重置 BytesIO 指针！
                    # ==========================================
                    if isinstance(obj, BytesIO):
                        obj.seek(0)

                    # ==========================================
                    # 第二步：判断竖屏，动态分配 width 和 height
                    # ==========================================
                    is_vertical = H_ori > W_ori
                    if is_vertical:
                        target_w, target_h = target_short_edge, target_long_edge
                    else:
                        target_w, target_h = target_long_edge, target_short_edge

                    # ==========================================
                    # 第三步：带正确尺寸的正式读取
                    # ==========================================
                    vr = decord.VideoReader(obj, width=target_w, height=target_h)
                    media_data = vr.get_batch(indices).asnumpy()

                    del vr
                    return media_data

                else:
                    raise NotImplementedError(f"Unsupported backend: {video_backend}")

            except Exception as e:
                raise RuntimeError(f"Failed to read video! Error: {str(e)}")
        else:
            # Image 处理逻辑...
            pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        video_id = data["task"]

        video_path_rel = data["src_video"]
        video_path_abs = os.path.join(self.video_root, video_path_rel)

        vframes = self.read_media(
            video_path_abs,
            ext=".mp4",
            video_backend="decord",
            target_fps=2,
            target_long_edge=576,
            target_short_edge=320,
        )
        return vframes, video_path_abs, video_path_rel


if __name__ == "__main__":
    dset = BaseVideoDataset(
        csv_path="../../location_train_preview100/location_train_preview100.csv",
        video_root="../../location_train_preview100",
    )

    for video, video_path_abs, video_path_rel in tqdm(dset, total=len(dset)):
        print(video.shape)
        exit(0)
