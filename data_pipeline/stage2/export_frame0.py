import os
import pandas as pd
import imageio.v3 as iio
from decord import VideoReader, cpu
from tqdm import tqdm

if __name__ == "__main__":
    edit_type="location"
    data_dir = "../location_train_preview100"

    csv_path = os.path.join(data_dir, f"stage2_generated_prompts_{edit_type}.csv")
    save_dir = os.path.join(data_dir, f"stage2_source_frame0_{edit_type}")

    os.makedirs(save_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    success = 0
    skipped = 0
    failed = 0

    for data in tqdm(df.to_dict("records"), desc="Extracting frame0"):
        source_obj = os.path.join(data_dir, data["src_video"])
        out_path = os.path.join(save_dir, data["task"] + ".png")

        if os.path.exists(out_path):
            skipped += 1
            continue

        try:
            vr = VideoReader(source_obj, ctx=cpu(0))
            frame0 = vr[0].asnumpy()
            iio.imwrite(out_path, frame0)

            success += 1

        except Exception as e:
            print(f"[Frame extraction error] {source_obj}: {e}")
            failed += 1

    print(f"Total: {len(df)}")
    print(f"Success: {success}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")
    print(f"Saved to: {save_dir}")