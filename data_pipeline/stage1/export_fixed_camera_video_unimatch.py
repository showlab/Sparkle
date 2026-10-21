import json
import pandas as pd
from tqdm import tqdm

if __name__ == "__main__":
    unimatch_flow = '../location_train_preview100/stage1_unimatch_motion_gathered.jsonl'
    csv_path = '../location_train_preview100/source_video.csv'
    save_path = '../location_train_preview100/stage1_fixed_camera_unimatch.csv'

    # ================= Step 1: Filter static videos =================
    unimatch_static_videos = set()

    with open(unimatch_flow, 'r') as f:
        for line in tqdm(f, desc="Filtering static videos"):
            item = json.loads(line)

            static_ratio = sum(item['camera_moving']) / len(item['camera_moving'])
            if static_ratio == 0:
                assert item['video_path'] not in unimatch_static_videos
                unimatch_static_videos.add(item['video_path'])

    print(f'Number of Unimatch static videos: {len(unimatch_static_videos)}')

    # ================= Step 2: Filter and export CSV =================
    df = pd.read_csv(csv_path)
    filtered_df = df[df['src_video'].isin(unimatch_static_videos)]

    filtered_df.to_csv(save_path, index=False)

    print(f'Number of original CSV rows: {len(df)}')
    print(f'Number of filtered CSV rows: {len(filtered_df)}')
    print(f'Number of unmatched static videos: {len(unimatch_static_videos - set(filtered_df["src_video"]))}')
    print(f'Saved to: {save_path}')