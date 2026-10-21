import os
import json
import pandas as pd
from tqdm import tqdm

if __name__ == "__main__":
    base_dir = "../location_train_preview100"

    qwen_reason = os.path.join(base_dir, "stage1_qwen3vl_camera_gathered.jsonl")
    csv_path = os.path.join(base_dir, "stage1_fixed_camera_unimatch.csv")
    save_path = os.path.join(base_dir, "stage1_fixed_camera_unimatch_qwen3vl.csv")

    # ================= Step 1: Filter static videos =================
    qwen_unimatch_static_videos = set()

    with open(qwen_reason, "r") as f:
        for line in tqdm(f, desc="Filtering static videos"):
            item = json.loads(line)

            if item["reasoning"]["is_static"]:
                assert item["video_path"] not in qwen_unimatch_static_videos
                qwen_unimatch_static_videos.add(item["video_path"])

    print(f"Number of Qwen3-VL static videos: {len(qwen_unimatch_static_videos)}")

    # ================= Step 2: Filter and export CSV =================
    df = pd.read_csv(csv_path)
    filtered_df = df[df["src_video"].isin(qwen_unimatch_static_videos)]

    filtered_df.to_csv(save_path, index=False)

    print(f"Number of original CSV rows: {len(df)}")
    print(f"Number of filtered CSV rows: {len(filtered_df)}")
    print(f"Number of unmatched static videos: {len(qwen_unimatch_static_videos - set(filtered_df['src_video']))}")
    print(f"Saved to: {save_path}")