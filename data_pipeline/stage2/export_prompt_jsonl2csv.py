import json
import os
import pandas as pd
from tqdm import tqdm

if __name__ == "__main__":
    input_path = "../location_train_preview100/stage2_generated_prompts_location_gathered.jsonl"
    output_path = "../location_train_preview100/stage2_generated_prompts_location.csv"

    data = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Converting JSONL to CSV"):
            if not line.strip():
                continue

            item = json.loads(line)

            video_path = item["video_path"]

            data.append({
                "prompt": item["generation"]["edit_instruction"],
                "src_video": video_path,
                "task": os.path.splitext(os.path.basename(video_path))[0],
            })

    df = pd.DataFrame(data, columns=["prompt", "src_video", "task"])
    df.to_csv(output_path, index=False, encoding="utf-8")

    print(f"Saved {len(df)} samples to {output_path}")