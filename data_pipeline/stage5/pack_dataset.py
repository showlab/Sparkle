import shutil
from pathlib import Path

import pandas as pd
from tqdm import tqdm


if __name__ == "__main__":
    # ================= Configuration =================
    root = Path("../location_train_preview100")
    edit_type = "location"

    input_csv = root / f"stage5_generated_prompts_{edit_type}.csv"
    output_csv = root / "dataset_sparkle.csv"

    source_dir = root / "source_video"
    edited_dir = root / f"stage5_edited_video_{edit_type}"

    final_source_dir = root / "source_video_sparkle"
    final_edited_dir = root / "edited_video_sparkle"
    # ================================================

    final_source_dir.mkdir(parents=True, exist_ok=True)
    final_edited_dir.mkdir(parents=True, exist_ok=True)

    # Load CSV
    df = pd.read_csv(input_csv)
    print(f"Total samples: {len(df)}")

    final_samples = []
    missing_source = 0
    missing_edited = 0

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Processing videos"
    ):
        task = row["task"]

        # Original paths
        source_path = source_dir / f"{task}.mp4"
        edited_path = edited_dir / f"{task}.mp4"

        # Skip samples with missing videos
        if not source_path.is_file():
            missing_source += 1
            continue

        if not edited_path.is_file():
            missing_edited += 1
            continue

        # Final paths
        final_source_path = final_source_dir / f"{task}.mp4"
        final_edited_path = final_edited_dir / f"{task}.mp4"

        # Copy videos
        shutil.copy2(source_path, final_source_path)
        shutil.copy2(edited_path, final_edited_path)

        # Final metadata
        final_samples.append({
            "prompt": row["prompt"],
            "src_video": f"source_video_sparkle/{task}.mp4",
            "tgt_video": f"edited_video_sparkle/{task}.mp4",
            "task": task,
        })

    # Save CSV
    final_df = pd.DataFrame(
        final_samples,
        columns=["prompt", "src_video", "tgt_video", "task"]
    )
    final_df.to_csv(output_csv, index=False)

    print("\n========== Summary ==========")
    print(f"Original samples: {len(df)}")
    print(f"Final samples:    {len(final_df)}")
    print(f"Missing source:   {missing_source}")
    print(f"Missing edited:   {missing_edited}")
    print(f"Output CSV:       {output_csv}")
    print("=============================")