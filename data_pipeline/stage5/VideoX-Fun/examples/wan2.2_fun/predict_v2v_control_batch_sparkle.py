import os
import sys
import json
import argparse

import numpy as np
import torch
import decord
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from PIL import Image
from transformers import AutoTokenizer
from tqdm import tqdm
import pandas as pd
from huggingface_hub import snapshot_download

current_file_path = os.path.abspath(__file__)
project_roots = [
    os.path.dirname(current_file_path),
    os.path.dirname(os.path.dirname(current_file_path)),
    os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))),
]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from videox_fun.dist import set_multi_gpus_devices, shard_model
from videox_fun.models import (
    AutoencoderKLWan,
    AutoencoderKLWan3_8,
    AutoTokenizer,
    CLIPModel,
    WanT5EncoderModel,
    Wan2_2Transformer3DModel,
)
from videox_fun.data.dataset_image_video import process_pose_file
from videox_fun.models.cache_utils import get_teacache_coefficients
from videox_fun.pipeline import Wan2_2FunControlPipeline, WanPipeline
from videox_fun.utils.fp8_optimization import (
    convert_model_weight_to_float8,
    convert_weight_dtype_wrapper,
    replace_parameters_by_name,
)
from videox_fun.utils.lora_utils import merge_lora, unmerge_lora
from videox_fun.utils.utils import (
    filter_kwargs,
    get_image_latent,
    get_image_to_video_latent,
    get_video_to_video_latent,
    save_videos_grid,
)
from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

INSTRUCTION_STARTERS = [
    "Replace the background with",
    "Change the background to",
    "Transform the background into",
    "Place the subject in",
    "Set the scene to",
    "Swap the background to",
    "Move the subject to",
    "Transport the subject to",
    "Shift the background to",
    "Put the subject against",
]


def strip_instruction_starter(prompt: str) -> str:
    for starter in INSTRUCTION_STARTERS:
        if prompt.startswith(starter):
            prompt = prompt[len(starter) :].strip()
            if prompt and prompt[0] in (",", ".", ";", ":"):
                prompt = prompt[1:].strip()
            break
    return prompt


def get_video_info(video_path: str) -> tuple:
    """Read video to get sample_size, video_length, fps."""
    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    fps = int(round(vr.get_avg_fps()))
    video_length = len(vr)
    h, w, _ = vr[0].shape
    sample_size = [h, w]
    return sample_size, video_length, fps


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_shards", type=int, default=None)
    parser.add_argument("--shard_index", type=int, default=None)
    parser.add_argument("--edit_type", type=str, required=True)
    return parser.parse_args()


# ================= GPU & Model Config =================
GPU_memory_mode = "model_full_load"
ulysses_degree = 8
ring_degree = 1
fsdp_dit = True
fsdp_text_encoder = True
compile_dit = False

enable_teacache = True
teacache_threshold = 0.10
num_skip_start_steps = 5
teacache_offload = False

cfg_skip_ratio = 0

enable_riflex = False
riflex_k = 6

config_path = "config/wan2.2/wan_civitai_i2v.yaml"
model_name = snapshot_download(
    repo_id="alibaba-pai/Wan2.2-Fun-A14B-Control",
    local_files_only=True,
)

sampler_name = "Flow"
shift = 5

transformer_path = None
transformer_high_path = None
vae_path = None
lora_path = None
lora_high_path = None

weight_dtype = torch.bfloat16
negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，倒着走"
guidance_scale = 6.0
seed = 43
num_inference_steps = 10
lora_weight = 0.55
lora_high_weight = 0.55

# ================= Main =================
if __name__ == "__main__":
    args = parse_args()
    edit_type = args.edit_type

    # ---- Paths ----
    metadata_csv = f"../../location_train_preview100/stage3_generated_prompts_{edit_type}_filtered.csv"
    frame0_dir = f"../../location_train_preview100/stage2_edited_frame0_{edit_type}"
    canny_video_dir = (
        f"../../location_train_preview100/stage5_edited_video_canny_{edit_type}"
    )
    output_dir = f"../../location_train_preview100/stage5_edited_video_{edit_type}"

    os.makedirs(output_dir, exist_ok=True)

    # ---- 1. Load metadata ----
    samples = []

    df = pd.read_csv(metadata_csv, dtype={"task": str})

    for data in tqdm(df.to_dict(orient="records"), desc="Loading metadata"):
        vid = data["task"]
        frame0_path = os.path.join(frame0_dir, f"{vid}.png")
        canny_path = os.path.join(canny_video_dir, f"{vid}.mp4")

        if os.path.exists(frame0_path) and os.path.exists(canny_path):
            edit_prompt = strip_instruction_starter(data["prompt"])
            edit_prompt = edit_prompt.strip()

            samples.append(
                {
                    "id": vid,
                    "prompt": edit_prompt,
                    "frame0": frame0_path,
                    "canny_video": canny_path,
                }
            )

    samples.sort(key=lambda s: s["id"])
    print(f"Total valid samples: {len(samples)}")

    # ---- 2. Shard ----
    if args.num_shards is not None and args.shard_index is not None:
        total = len(samples)
        shard_size = total // args.num_shards
        remainder = total % args.num_shards
        if args.shard_index < remainder:
            start = args.shard_index * (shard_size + 1)
            end = start + shard_size + 1
        else:
            start = (
                remainder * (shard_size + 1)
                + (args.shard_index - remainder) * shard_size
            )
            end = start + shard_size
        samples = samples[start:end]
        print(
            f"Shard {args.shard_index}/{args.num_shards}: processing {len(samples)} samples (indices {start}-{end - 1})"
        )

    # ---- 3. Load model (once) ----
    device = set_multi_gpus_devices(ulysses_degree, ring_degree)
    config = OmegaConf.load(config_path)
    boundary = config["transformer_additional_kwargs"].get("boundary", 0.875)

    transformer = Wan2_2Transformer3DModel.from_pretrained(
        os.path.join(
            model_name,
            config["transformer_additional_kwargs"].get(
                "transformer_low_noise_model_subpath", "transformer"
            ),
        ),
        transformer_additional_kwargs=OmegaConf.to_container(
            config["transformer_additional_kwargs"]
        ),
        low_cpu_mem_usage=False,
        torch_dtype=weight_dtype,
    )
    if (
        config["transformer_additional_kwargs"].get(
            "transformer_combination_type", "single"
        )
        == "moe"
    ):
        transformer_2 = Wan2_2Transformer3DModel.from_pretrained(
            os.path.join(
                model_name,
                config["transformer_additional_kwargs"].get(
                    "transformer_high_noise_model_subpath", "transformer"
                ),
            ),
            transformer_additional_kwargs=OmegaConf.to_container(
                config["transformer_additional_kwargs"]
            ),
            low_cpu_mem_usage=False,
            torch_dtype=weight_dtype,
        )
    else:
        transformer_2 = None

    if transformer_path is not None:
        print(f"From checkpoint: {transformer_path}")
        if transformer_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open

            state_dict = load_file(transformer_path)
        else:
            state_dict = torch.load(transformer_path, map_location="cpu")
        state_dict = (
            state_dict["state_dict"] if "state_dict" in state_dict else state_dict
        )
        m, u = transformer.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

    if transformer_2 is not None and transformer_high_path is not None:
        print(f"From checkpoint: {transformer_high_path}")
        if transformer_high_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open

            state_dict = load_file(transformer_high_path)
        else:
            state_dict = torch.load(transformer_high_path, map_location="cpu")
        state_dict = (
            state_dict["state_dict"] if "state_dict" in state_dict else state_dict
        )
        m, u = transformer_2.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

    Chosen_AutoencoderKL = {
        "AutoencoderKLWan": AutoencoderKLWan,
        "AutoencoderKLWan3_8": AutoencoderKLWan3_8,
    }[config["vae_kwargs"].get("vae_type", "AutoencoderKLWan")]
    vae = Chosen_AutoencoderKL.from_pretrained(
        os.path.join(model_name, config["vae_kwargs"].get("vae_subpath", "vae")),
        additional_kwargs=OmegaConf.to_container(config["vae_kwargs"]),
    ).to(weight_dtype)

    if vae_path is not None:
        print(f"From checkpoint: {vae_path}")
        if vae_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open

            state_dict = load_file(vae_path)
        else:
            state_dict = torch.load(vae_path, map_location="cpu")
        state_dict = (
            state_dict["state_dict"] if "state_dict" in state_dict else state_dict
        )
        m, u = vae.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(
            model_name,
            config["text_encoder_kwargs"].get("tokenizer_subpath", "tokenizer"),
        ),
    )

    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(
            model_name,
            config["text_encoder_kwargs"].get("text_encoder_subpath", "text_encoder"),
        ),
        additional_kwargs=OmegaConf.to_container(config["text_encoder_kwargs"]),
        low_cpu_mem_usage=False,
        torch_dtype=weight_dtype,
    )
    text_encoder = text_encoder.eval()

    Chosen_Scheduler = {
        "Flow": FlowMatchEulerDiscreteScheduler,
        "Flow_Unipc": FlowUniPCMultistepScheduler,
        "Flow_DPM++": FlowDPMSolverMultistepScheduler,
    }[sampler_name]
    if sampler_name in ["Flow_Unipc", "Flow_DPM++"]:
        config["scheduler_kwargs"]["shift"] = 1
    scheduler = Chosen_Scheduler(
        **filter_kwargs(
            Chosen_Scheduler, OmegaConf.to_container(config["scheduler_kwargs"])
        )
    )

    pipeline = Wan2_2FunControlPipeline(
        transformer=transformer,
        transformer_2=transformer_2,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=scheduler,
    )
    if ulysses_degree > 1 or ring_degree > 1:
        from functools import partial

        transformer.enable_multi_gpus_inference()
        if transformer_2 is not None:
            transformer_2.enable_multi_gpus_inference()
        if fsdp_dit:
            shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
            pipeline.transformer = shard_fn(pipeline.transformer)
            if transformer_2 is not None:
                pipeline.transformer_2 = shard_fn(pipeline.transformer_2)
            print("Add FSDP DIT")
        if fsdp_text_encoder:
            shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
            pipeline.text_encoder = shard_fn(pipeline.text_encoder)
            print("Add FSDP TEXT ENCODER")

    if compile_dit:
        for i in range(len(pipeline.transformer.blocks)):
            pipeline.transformer.blocks[i] = torch.compile(
                pipeline.transformer.blocks[i]
            )
        if transformer_2 is not None:
            for i in range(len(pipeline.transformer_2.blocks)):
                pipeline.transformer_2.blocks[i] = torch.compile(
                    pipeline.transformer_2.blocks[i]
                )
        print("Add Compile")

    if GPU_memory_mode == "sequential_cpu_offload":
        replace_parameters_by_name(
            transformer,
            [
                "modulation",
            ],
            device=device,
        )
        transformer.freqs = transformer.freqs.to(device=device)
        if transformer_2 is not None:
            replace_parameters_by_name(
                transformer_2,
                [
                    "modulation",
                ],
                device=device,
            )
            transformer_2.freqs = transformer_2.freqs.to(device=device)
        pipeline.enable_sequential_cpu_offload(device=device)
    elif GPU_memory_mode == "model_cpu_offload_and_qfloat8":
        convert_model_weight_to_float8(
            transformer,
            exclude_module_name=[
                "modulation",
            ],
            device=device,
        )
        convert_weight_dtype_wrapper(transformer, weight_dtype)
        if transformer_2 is not None:
            convert_model_weight_to_float8(
                transformer_2,
                exclude_module_name=[
                    "modulation",
                ],
                device=device,
            )
            convert_weight_dtype_wrapper(transformer_2, weight_dtype)
        pipeline.enable_model_cpu_offload(device=device)
    elif GPU_memory_mode == "model_cpu_offload":
        pipeline.enable_model_cpu_offload(device=device)
    elif GPU_memory_mode == "model_full_load_and_qfloat8":
        convert_model_weight_to_float8(
            transformer,
            exclude_module_name=[
                "modulation",
            ],
            device=device,
        )
        convert_weight_dtype_wrapper(transformer, weight_dtype)
        if transformer_2 is not None:
            convert_model_weight_to_float8(
                transformer_2,
                exclude_module_name=[
                    "modulation",
                ],
                device=device,
            )
            convert_weight_dtype_wrapper(transformer_2, weight_dtype)
        pipeline.to(device=device)
    else:
        pipeline.to(device=device)

    coefficients = get_teacache_coefficients(model_name) if enable_teacache else None

    if lora_path is not None:
        pipeline = merge_lora(
            pipeline, lora_path, lora_weight, device=device, dtype=weight_dtype
        )
        if transformer_2 is not None:
            pipeline = merge_lora(
                pipeline,
                lora_high_path,
                lora_high_weight,
                device=device,
                dtype=weight_dtype,
                sub_transformer_name="transformer_2",
            )

    # ---- 4. Batch inference ----
    success, fail = 0, 0
    pbar = tqdm(samples, desc="Generating videos")

    for sample in pbar:
        vid = sample["id"]
        save_video_path = os.path.join(output_dir, f"{vid}.mp4")

        if os.path.exists(save_video_path):
            success += 1
            pbar.set_postfix(success=success, fail=fail)
            continue

        try:
            sample_size, video_length, fps = get_video_info(sample["canny_video"])

            control_video = sample["canny_video"]
            start_image = sample["frame0"]
            prompt = sample["prompt"]

            print(f"[{vid}] size={sample_size}, frames={video_length}, fps={fps}")

            # Align video_length to VAE temporal compression
            video_length = (
                int(
                    (video_length - 1)
                    // vae.config.temporal_compression_ratio
                    * vae.config.temporal_compression_ratio
                )
                + 1
                if video_length != 1
                else 1
            )
            latent_frames = (
                video_length - 1
            ) // vae.config.temporal_compression_ratio + 1

            if coefficients is not None:
                pipeline.transformer.enable_teacache(
                    coefficients,
                    num_inference_steps,
                    teacache_threshold,
                    num_skip_start_steps=num_skip_start_steps,
                    offload=teacache_offload,
                )
                if transformer_2 is not None:
                    pipeline.transformer_2.share_teacache(
                        transformer=pipeline.transformer
                    )

            if cfg_skip_ratio is not None:
                pipeline.transformer.enable_cfg_skip(
                    cfg_skip_ratio, num_inference_steps
                )
                if transformer_2 is not None:
                    pipeline.transformer_2.share_cfg_skip(
                        transformer=pipeline.transformer
                    )

            if enable_riflex:
                pipeline.transformer.enable_riflex(k=riflex_k, L_test=latent_frames)
                if transformer_2 is not None:
                    pipeline.transformer_2.enable_riflex(
                        k=riflex_k, L_test=latent_frames
                    )

            generator = torch.Generator(device=device).manual_seed(seed)

            with torch.no_grad():
                inpaint_video, inpaint_video_mask, clip_image = (
                    get_image_to_video_latent(
                        start_image,
                        None,
                        video_length=video_length,
                        sample_size=sample_size,
                    )
                )

                input_video, input_video_mask, _, _ = get_video_to_video_latent(
                    control_video,
                    video_length=video_length,
                    sample_size=sample_size,
                    fps=fps,
                    ref_image=None,
                )

                result = pipeline(
                    prompt,
                    num_frames=video_length,
                    negative_prompt=negative_prompt,
                    height=sample_size[0],
                    width=sample_size[1],
                    generator=generator,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    video=inpaint_video,
                    mask_video=inpaint_video_mask,
                    control_video=input_video,
                    control_camera_video=None,
                    ref_image=None,
                    boundary=boundary,
                    shift=shift,
                ).videos

            save_videos_grid(result, save_video_path, fps=fps)

            success += 1
            print(f"[Done] {vid} -> {save_video_path}")

        except Exception as e:
            print(f"[Error] {vid}: {e}")
            fail += 1
            continue

        pbar.set_postfix(success=success, fail=fail)

    if lora_path is not None:
        pipeline = unmerge_lora(
            pipeline, lora_path, lora_weight, device=device, dtype=weight_dtype
        )
        if transformer_2 is not None:
            pipeline = unmerge_lora(
                pipeline,
                lora_high_path,
                lora_high_weight,
                device=device,
                dtype=weight_dtype,
                sub_transformer_name="transformer_2",
            )

    print(f"\nDone. success={success}, fail={fail}")
