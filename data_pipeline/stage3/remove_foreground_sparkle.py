import argparse
import json
import time
from pathlib import Path
import pandas as pd

import torch
from PIL import Image
from tqdm import tqdm

from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.platforms import current_omni_platform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remove foreground from edited_frame0 using Qwen-Image-Edit.")
    parser.add_argument("--edit-type", type=str, required=True)
    parser.add_argument(
        "--model",
        default="Qwen/Qwen-Image-Edit",
        help="Diffusion model name or local path.",
    )

    # Sharding
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--shard-index", type=int, default=None)

    # Generation hyper-parameters
    parser.add_argument("--negative-prompt", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--guidance-scale-2", type=float, default=None)
    parser.add_argument("--num-outputs-per-prompt", type=int, default=1)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=640)
    parser.add_argument("--color-format", type=str, default="RGB")

    # Cache / parallelism
    parser.add_argument("--cache-backend", type=str, default=None, choices=["cache_dit", "tea_cache"])
    parser.add_argument("--ulysses-degree", type=int, default=1)
    parser.add_argument("--ring-degree", type=int, default=1)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--enable-expert-parallel", action="store_true")
    parser.add_argument("--cfg-parallel-size", type=int, default=1, choices=[1, 2])
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--vae-use-slicing", action="store_true")
    parser.add_argument("--vae-use-tiling", action="store_true")
    parser.add_argument("--enable-cpu-offload", action="store_true")
    parser.add_argument("--enable-layerwise-offload", action="store_true")
    parser.add_argument("--enable-diffusion-pipeline-profiler", action="store_true")

    # cache-dit specific
    parser.add_argument("--cache-dit-fn-compute-blocks", type=int, default=1)
    parser.add_argument("--cache-dit-bn-compute-blocks", type=int, default=0)
    parser.add_argument("--cache-dit-max-warmup-steps", type=int, default=4)
    parser.add_argument("--cache-dit-residual-diff-threshold", type=float, default=0.24)
    parser.add_argument("--cache-dit-max-continuous-cached-steps", type=int, default=3)
    parser.add_argument("--cache-dit-enable-taylorseer", action="store_true", default=False)
    parser.add_argument("--cache-dit-taylorseer-order", type=int, default=1)
    parser.add_argument("--cache-dit-scm-steps-mask-policy", type=str, default=None,
                        choices=[None, "slow", "medium", "fast", "ultra"])
    parser.add_argument("--cache-dit-scm-steps-policy", type=str, default="dynamic", choices=["dynamic", "static"])

    # tea-cache specific
    parser.add_argument("--tea-cache-rel-l1-thresh", type=float, default=0.2)

    return parser.parse_args()


# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #

def build_cache_config(args) -> dict | None:
    if args.cache_backend == "cache_dit":
        return {
            "Fn_compute_blocks": args.cache_dit_fn_compute_blocks,
            "Bn_compute_blocks": args.cache_dit_bn_compute_blocks,
            "max_warmup_steps": args.cache_dit_max_warmup_steps,
            "residual_diff_threshold": args.cache_dit_residual_diff_threshold,
            "max_continuous_cached_steps": args.cache_dit_max_continuous_cached_steps,
            "enable_taylorseer": args.cache_dit_enable_taylorseer,
            "taylorseer_order": args.cache_dit_taylorseer_order,
            "scm_steps_mask_policy": args.cache_dit_scm_steps_mask_policy,
            "scm_steps_policy": args.cache_dit_scm_steps_policy,
        }
    elif args.cache_backend == "tea_cache":
        return {"rel_l1_thresh": args.tea_cache_rel_l1_thresh}
    return None


def build_removal_prompt(labels: list[str]) -> str:
    """Build a natural-language removal prompt from a list of labels."""
    if len(labels) == 1:
        return f"Remove the {labels[0]}"
    elif len(labels) == 2:
        return f"Remove the {labels[0]} and {labels[1]}"
    else:
        return "Remove the " + ", ".join(labels[:-1]) + f", and {labels[-1]}"


def save_images(images, output_path: Path, num_outputs: int) -> None:
    """Save one or more (possibly layered) output images."""
    suffix = output_path.suffix or ".png"
    stem = output_path.stem

    if num_outputs <= 1:
        img = images[0]
        if isinstance(img, list):
            for sub_idx, sub_img in enumerate(img):
                sub_img.save(output_path.parent / f"{stem}_{sub_idx}{suffix}")
        else:
            img.save(output_path)
    else:
        for idx, img in enumerate(images):
            if isinstance(img, list):
                for sub_idx, sub_img in enumerate(img):
                    sub_img.save(output_path.parent / f"{stem}_{idx}_{sub_idx}{suffix}")
            else:
                img.save(output_path.parent / f"{stem}_{idx}{suffix}")


def load_foreground_labels(fg_path: str) -> dict[str, list[str]]:
    """Load Round 1 foreground grounding JSONL, return id -> labels mapping."""
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


# --------------------------------------------------------------------------- #
#  Main                                                                        #
# --------------------------------------------------------------------------- #

def main():
    args = parse_args()
    edit_type = args.edit_type

    # ---- paths ----
    metadata_path = f"../location_train_preview100/stage2_generated_prompts_{edit_type}_filtered.csv"
    fg_path = f"../location_train_preview100/stage3_foreground_grounding_{edit_type}_gathered.jsonl"
    edited_frame0_dir = Path(f"../location_train_preview100/stage2_edited_frame0_{edit_type}")
    output_dir = Path(f"../location_train_preview100/stage3_edited_frame0_{edit_type}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- load metadata ----
    print(f"Loading metadata from {metadata_path} ...")
    samples=pd.read_csv(metadata_path).to_dict(orient='records')
    print(f"Loaded {len(samples)} samples.")

    # ---- load foreground labels ----
    id2labels = load_foreground_labels(fg_path)

    # ---- sharding ----
    assert (
            (args.num_shards is None and args.shard_index is None) or
            (args.num_shards is not None and args.shard_index is not None)
    )

    if args.num_shards is not None:
        total = len(samples)
        shard_size = total // args.num_shards
        start = args.shard_index * shard_size
        end = start + shard_size if args.shard_index < args.num_shards - 1 else total
        samples = samples[start:end]
        print(f"Shard {args.shard_index}/{args.num_shards}, samples [{start}:{end}), count={len(samples)}")

    # ---- model ----
    parallel_config = DiffusionParallelConfig(
        ulysses_degree=args.ulysses_degree,
        ring_degree=args.ring_degree,
        cfg_parallel_size=args.cfg_parallel_size,
        tensor_parallel_size=args.tensor_parallel_size,
        enable_expert_parallel=args.enable_expert_parallel,
    )

    cache_config = build_cache_config(args)

    omni = Omni(
        model=args.model,
        enable_layerwise_offload=args.enable_layerwise_offload,
        vae_use_slicing=args.vae_use_slicing,
        vae_use_tiling=args.vae_use_tiling,
        cache_backend=args.cache_backend,
        cache_config=cache_config,
        parallel_config=parallel_config,
        enforce_eager=args.enforce_eager,
        enable_cpu_offload=args.enable_cpu_offload,
        enable_diffusion_pipeline_profiler=args.enable_diffusion_pipeline_profiler,
        init_timeout=1800,
    )

    # ---- processing loop ----
    skipped = 0
    errors = 0
    no_labels = 0
    t_start = time.perf_counter()

    # resume: scan existing output images
    done_ids = {p.stem for p in output_dir.glob("*.png")}
    if done_ids:
        print(f"Resuming: {len(done_ids)} already done.")

    pbar = tqdm(samples, desc=f"{edit_type} fg-remove shard={args.shard_index}")
    for item in pbar:
        sample_id = item["task"]
        output_path = output_dir / f"{sample_id}.png"

        if sample_id in done_ids:
            skipped += 1
            pbar.set_postfix(skip=skipped, err=errors, nolbl=no_labels, refresh=False)
            continue

        # ---- look up labels ----
        labels = id2labels.get(sample_id)
        if not labels:
            no_labels += 1
            pbar.set_postfix(skip=skipped, err=errors, nolbl=no_labels, refresh=False)
            continue

        prompt = build_removal_prompt(labels)

        # ---- load edited frame0 ----
        edited_path = edited_frame0_dir / f"{sample_id}.png"
        if not edited_path.exists():
            print(f"[WARN] Missing edited_frame0: {edited_path}")
            errors += 1
            pbar.set_postfix(skip=skipped, err=errors, nolbl=no_labels, refresh=False)
            continue

        try:
            input_image = Image.open(edited_path).convert(args.color_format)
        except Exception as e:
            print(f"[WARN] Failed to open {edited_path}: {e}")
            errors += 1
            pbar.set_postfix(skip=skipped, err=errors, nolbl=no_labels, refresh=False)
            continue

        # ---- generate ----
        generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(args.seed)

        try:
            outputs = omni.generate(
                {
                    "prompt": prompt,
                    "negative_prompt": args.negative_prompt,
                    "multi_modal_data": {"image": input_image},
                },
                OmniDiffusionSamplingParams(
                    generator=generator,
                    true_cfg_scale=args.cfg_scale,
                    guidance_scale=args.guidance_scale,
                    guidance_scale_2=args.guidance_scale_2,
                    num_inference_steps=args.num_inference_steps,
                    num_outputs_per_prompt=args.num_outputs_per_prompt,
                    layers=args.layers,
                ),
                use_tqdm=False,
            )
        except Exception as e:
            print(f"[ERROR] Generation failed for {sample_id}: {e}")
            errors += 1
            pbar.set_postfix(skip=skipped, err=errors, nolbl=no_labels, refresh=False)
            continue

        # ---- validate & save ----
        if not outputs:
            print(f"[WARN] Empty outputs for {sample_id}")
            errors += 1
            continue

        first_output = outputs[0]
        if not hasattr(first_output, "request_output") or not first_output.request_output:
            print(f"[WARN] No request_output for {sample_id}")
            errors += 1
            continue

        req_out = first_output.request_output
        if not isinstance(req_out, OmniRequestOutput) or not hasattr(req_out, "images"):
            print(f"[WARN] Invalid request_output for {sample_id}")
            errors += 1
            continue

        images = req_out.images
        if not images:
            print(f"[WARN] No images in output for {sample_id}")
            errors += 1
            continue

        save_images(images, output_path, args.num_outputs_per_prompt)
        pbar.set_postfix(skip=skipped, err=errors, nolbl=no_labels, refresh=False)

    # ---- summary ----
    t_total = time.perf_counter() - t_start
    processed = len(samples) - skipped - errors - no_labels
    print(
        f"\n{'=' * 60}\n"
        f"Done.  total={len(samples)}  processed={processed}  "
        f"skipped={skipped}  no_labels={no_labels}  errors={errors}  "
        f"wall_time={t_total:.1f}s\n"
        f"{'=' * 60}"
    )


if __name__ == "__main__":
    main()
