import vllm  # import vllm first to avoid conflict
from PIL import Image
from editscore import EditScore
from tqdm import tqdm
import json
import os
import argparse
import pandas as pd


def build_removal_prompt(labels: list[str]) -> str:
    if len(labels) == 1:
        return f"Remove the {labels[0]}"
    elif len(labels) == 2:
        return f"Remove the {labels[0]} and {labels[1]}"
    else:
        return "Remove the " + ", ".join(labels[:-1]) + f", and {labels[-1]}"


def load_foreground_labels(fg_path: str) -> dict[str, list[str]]:
    id2labels: dict[str, list[str]] = {}
    with open(fg_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            labels = item.get("round1_labels", [])
            if labels:
                id2labels[item["id"]] = labels
    print(f"Loaded foreground labels for {len(id2labels)} samples from {fg_path}")
    return id2labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--edit_type", type=str, required=True)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    args = parser.parse_args()

    edit_type = args.edit_type
    num_shards = args.num_shards
    shard_index = args.shard_index

    base_dir = "../../location_train_preview100"
    metadata_path = f"{base_dir}/stage2_generated_prompts_{edit_type}_filtered.csv"
    fg_path = f"{base_dir}/stage3_foreground_grounding_{edit_type}_gathered.jsonl"
    source_dir = f"{base_dir}/stage2_edited_frame0_{edit_type}"
    edited_dir = f"{base_dir}/stage3_edited_frame0_{edit_type}"
    output_file = (
        f"{base_dir}/stage3_editscore/{edit_type}_{num_shards}_{shard_index}.jsonl"
    )
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # load metadata
    samples = pd.read_csv(metadata_path).to_dict("records")
    print(f"Loaded {len(samples)} samples from {metadata_path}")

    # load foreground labels
    id2labels = load_foreground_labels(fg_path)

    # shard (contiguous, same as HF datasets)
    total = len(samples)
    per_shard = total // num_shards
    remainder = total % num_shards
    start = per_shard * shard_index + min(shard_index, remainder)
    end = start + per_shard + (1 if shard_index < remainder else 0)
    samples = samples[start:end]
    print(
        f"[shard {shard_index}/{num_shards}] {len(samples)} samples, output -> {output_file}"
    )

    # resume: skip already scored ids
    done_ids = set()
    if os.path.exists(output_file):
        with open(output_file) as f:
            for line in f:
                done_ids.add(json.loads(line)["id"])
        print(f"Resuming: {len(done_ids)} already done")

    scorer = EditScore(
        backbone="qwen3vl_vllm",
        model_name_or_path="Qwen/Qwen3-VL-8B-Instruct",
        lora_path="EditScore/EditScore-Qwen3-VL-8B-Instruct",
        score_range=25,
        num_pass=4,
        max_model_len=8192,
        max_num_batched_tokens=8192,
        tensor_parallel_size=1,
    )

    pbar = tqdm(samples, desc=f"shard {shard_index}")
    n_ok, n_fail, n_nolbl = 0, 0, 0
    for sample in pbar:
        iid = sample["task"]
        if iid in done_ids:
            continue

        # look up foreground labels -> build removal prompt
        labels = id2labels.get(iid)
        if not labels:
            n_nolbl += 1
            pbar.set_postfix(ok=n_ok, fail=n_fail, nolbl=n_nolbl)
            continue

        prompt = build_removal_prompt(labels)

        source_path = f"{source_dir}/{iid}.png"
        edited_path = f"{edited_dir}/{iid}.png"

        if not os.path.exists(source_path) or not os.path.exists(edited_path):
            n_fail += 1
            print(
                f"MISS {iid}: source={os.path.exists(source_path)} edited={os.path.exists(edited_path)}"
            )
            pbar.set_postfix(ok=n_ok, fail=n_fail, nolbl=n_nolbl)
            continue

        try:
            source_img = Image.open(source_path).convert("RGB")
            edited_img = Image.open(edited_path).convert("RGB")

            result = scorer.evaluate([source_img, edited_img], prompt)

            record = {
                "id": iid,
                "prompt": prompt,
                "editscore": result,
            }

            with open(output_file, "a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_ok += 1

        except Exception as e:
            n_fail += 1
            print(f"FAIL {iid}: {e}")

        pbar.set_postfix(ok=n_ok, fail=n_fail, nolbl=n_nolbl)
