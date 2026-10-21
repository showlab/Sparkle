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

PROMPT_SUFFIX = (
    "Adjust the lighting, shadows, and color tone of the foreground objects "
    "so they blend naturally into the new background."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sparkle: batch edit first frames with Qwen-Image-Edit.")
    parser.add_argument("--model", default="Qwen/Qwen-Image-Edit")
    parser.add_argument("--edit-type", type=str, required=True, help="Edit type, e.g. location, weather, style, etc.")

    # Sharding
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--shard-index", type=int, default=None)

    # Output
    parser.add_argument("--skip-existing", action="store_true")

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


def save_images(images, output_path: Path, num_outputs: int) -> None:
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


def main():
    args = parse_args()

    # ------------------------------------------------------------------ #
    # 1. Derive paths from edit_type                                     #
    # ------------------------------------------------------------------ #
    edit_type = args.edit_type
    prompts_path = f"../location_train_preview100/stage2_generated_prompts_{edit_type}.csv"
    source_frame0_dir = Path(f"../location_train_preview100/stage2_source_frame0_{edit_type}")
    output_dir = Path(f"../location_train_preview100/stage2_edited_frame0_{edit_type}")

    # ------------------------------------------------------------------ #
    # 2. Load prompts CSV                                                #
    # ------------------------------------------------------------------ #
    print(f"Loading prompts from {prompts_path} ...")
    samples = pd.read_csv(prompts_path).to_dict("records")
    print(f"Loaded {len(samples)} samples.")

    # Sharding
    if args.num_shards is not None and args.shard_index is not None:
        total_all = len(samples)
        shard_size = (total_all + args.num_shards - 1) // args.num_shards
        start = args.shard_index * shard_size
        end = min(start + shard_size, total_all)
        samples = samples[start:end]
        print(f"Shard {args.shard_index}/{args.num_shards}: {len(samples)} samples (range [{start}, {end})).")

    # ------------------------------------------------------------------ #
    # 3. Build output directory                                           #
    # ------------------------------------------------------------------ #
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 4. Initialise the Omni pipeline                                     #
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    # 5. Batch processing loop                                            #
    # ------------------------------------------------------------------ #
    total = len(samples)
    skipped = 0
    errors = 0
    t_total_start = time.perf_counter()

    pbar = tqdm(samples, desc="Editing images", total=total)
    for item in pbar:
        sample_id: str = item["task"]
        edit_prompt: str = item["prompt"]

        # Output path: edited_frame0/{id}.png  (mirrors source_frame0/{id}.png)
        output_path = output_dir / f"{sample_id}.png"

        # ------ optional skip ------
        if args.skip_existing and output_path.exists():
            skipped += 1
            pbar.set_postfix(skip=skipped, err=errors, refresh=False)
            continue

        # ------ load source frame0 from disk ------
        source_frame0_path = source_frame0_dir / f"{sample_id}.png"
        try:
            input_image = Image.open(source_frame0_path).convert(args.color_format)
        except Exception as e:
            print(f"[WARN] Failed to load frame0 for id={sample_id!r} at {source_frame0_path}: {e}. Skipping.")
            errors += 1
            pbar.set_postfix(skip=skipped, err=errors, refresh=False)
            continue

        # ------ build prompt ------
        full_prompt = f"{edit_prompt} {PROMPT_SUFFIX}"

        # ------ generate ------
        generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(args.seed)

        try:
            outputs = omni.generate(
                {
                    "prompt": full_prompt,
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
            print(f"[ERROR] Generation failed for id={sample_id!r}: {e}")
            errors += 1
            pbar.set_postfix(skip=skipped, err=errors, refresh=False)
            continue

        # ------ validate & save ------
        if not outputs:
            print(f"[WARN] Empty outputs for id={sample_id!r}, skipping.")
            errors += 1
            pbar.set_postfix(skip=skipped, err=errors, refresh=False)
            continue

        first_output = outputs[0]
        if not hasattr(first_output, "request_output") or not first_output.request_output:
            print(f"[WARN] No request_output for id={sample_id!r}, skipping.")
            errors += 1
            pbar.set_postfix(skip=skipped, err=errors, refresh=False)
            continue

        req_out = first_output.request_output
        if not isinstance(req_out, OmniRequestOutput) or not hasattr(req_out, "images"):
            print(f"[WARN] Invalid request_output structure for id={sample_id!r}, skipping.")
            errors += 1
            pbar.set_postfix(skip=skipped, err=errors, refresh=False)
            continue

        images = req_out.images
        if not images:
            print(f"[WARN] No images in output for id={sample_id!r}, skipping.")
            errors += 1
            pbar.set_postfix(skip=skipped, err=errors, refresh=False)
            continue

        save_images(images, output_path, args.num_outputs_per_prompt)

        pbar.set_postfix(skip=skipped, err=errors, refresh=False)

    # ------------------------------------------------------------------ #
    # 6. Summary                                                          #
    # ------------------------------------------------------------------ #
    t_total = time.perf_counter() - t_total_start
    processed = total - skipped - errors
    print(
        f"\n{'=' * 60}\n"
        f"Done.  total={total}  processed={processed}  "
        f"skipped={skipped}  errors={errors}  "
        f"wall_time={t_total:.1f}s\n"
        f"{'=' * 60}"
    )


if __name__ == "__main__":
    main()
