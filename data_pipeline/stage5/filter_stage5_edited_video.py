import os
import sys
import json
import numpy as np
from tqdm import tqdm
import pandas as pd

if __name__ == "__main__":
    edit_type = "location"
    threshold = 8

    editscore_path = f"../location_train_preview100/stage5_editscore_{edit_type}_gathered.jsonl"
    full_set_path = f"../location_train_preview100/stage3_generated_prompts_{edit_type}_filtered.csv"
    output_path = f"../location_train_preview100/stage5_generated_prompts_{edit_type}.csv"

    valid_id = set()
    with open(editscore_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            editscore_all = data["editscore"]
            overall_scores = [score['O_score'] for score in editscore_all]
            avg_overall_score = np.mean(overall_scores)
            if avg_overall_score >= threshold:
                assert data["id"] not in valid_id
                valid_id.add(data["id"])

    # Read CSV
    df = pd.read_csv(full_set_path)

    # Filter samples
    filtered_df = df[df["task"].isin(valid_id)].copy()

    print(f"Number of filtered samples for {edit_type}: {len(filtered_df)}")

    # Save CSV
    filtered_df.to_csv(output_path, index=False)
