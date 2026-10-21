"""
Wan2.2 distilled model image-to-video generation example.
This example demonstrates how to use LightX2V with Wan2.2 distilled model for I2V generation.
"""

from lightx2v import LightX2VPipeline

# Initialize pipeline for Wan2.2 distilled I2V task
# For wan2.1, use model_cls="wan2.1_distill"
pipe = LightX2VPipeline(
    model_path="model_hub/Wan2.2-I2V-A14B",
    model_cls="wan2.2_moe_distill",
    task="i2v",
    # Distilled weights: For wan2.1, only need to specify dit_original_ckpt="/path/to/wan2.1_i2v_720p_lightx2v_4step.safetensors"
    low_noise_original_ckpt="wan2.2_i2v_A14b_low_noise_lightx2v_4step.safetensors",
    high_noise_original_ckpt="wan2.2_i2v_A14b_high_noise_lightx2v_4step_1030.safetensors",
)

# Alternative: create generator from config JSON file
# pipe.create_generator(
#     config_json="../configs/wan22/wan_moe_i2v_distill.json"
# )

# Enable offloading to significantly reduce VRAM usage
# Suitable for RTX 30/40/50 consumer GPUs
# pipe.enable_offload(
#     cpu_offload=True,
#     offload_granularity="block",
#     text_encoder_offload=True,
#     image_encoder_offload=False,
#     vae_offload=False,
# )

pipe.enable_parallel(
    seq_p_size=4,
    seq_p_attn_type="ulysses"
)

# Create generator manually with specified parameters
pipe.create_generator(
    attn_mode="sage_attn2",
    infer_steps=4,
    height=720,  # Can be set to 720 for higher resolution
    width=1280,  # Can be set to 1280 for higher resolution
    num_frames=81,
    guidance_scale=1,
    sample_shift=5.0,
)

seed = 42
# prompt = "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
# negative_prompt = "镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
# image_path = "assets/inputs/imgs/img_0.jpg"

# 第一组：抑制相机运动
NEG_CAM = "camera motion, camera shake, panning, tilting, zooming, dolly, tracking shot, handheld, perspective change, dolly in, dolly out, truck left, truck right, pedestal up, pedestal down"

# 第二组：防止完全静止
NEG_FREEZE = "motionless, static scene, still image"

# 第三组：防止物体凭空出现、场景突变
NEG_COHERENCE = ""

# 第四组：常规质量控制
NEG_QUALITY = "logo，watermark, text, subtitle, distort, low quality, 色调艳丽，过曝，静态，细节模糊不清，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，倒着走"

negative_prompt = ", ".join([NEG_CAM, NEG_FREEZE, NEG_COHERENCE, NEG_QUALITY])

# prompt = "a cozy library nook with flickering fireplace light, gentle dust motes floating in the air, and the subtle movement of pages turning in the distance. The warm lighting softly illuminates the wooden shelves and creates a calm, inviting atmosphere"
# image_path = "openve3m_background_change_048569.png"

# prompt = "Gentle turquoise waves roll toward the shore in rhythmic intervals, white foam spreading across the wet sand before receding. The palm tree fronds on both sides sway softly in a light ocean breeze. Several seagulls glide and circle lazily against the clear blue sky, occasionally dipping lower toward the water. Sunlight glints off the wave crests, casting shifting reflections. Footprints in the warm golden sand remain undisturbed. The wave foam fizzes and dissolves naturally at the waterline. Ambient tropical daylight, consistent lighting throughout."
# image_path = "openve3m_background_change_000556.png"

# save_result_path = "save_results/output.mp4"

import time

image_paths = [
    "openve3m_background_change_048569.png",
    "openve3m_background_change_000556.png"
]

prompt_prefix = "A scene filmed from a stationary tripod-mounted camera with zero pan, tilt, or zoom."

prompts = [
    "a cozy library nook with flickering fireplace light, gentle dust motes floating in the air, and the subtle movement of pages turning in the distance. The warm lighting softly illuminates the wooden shelves and creates a calm, inviting atmosphere",
    "Gentle turquoise waves roll toward the shore in rhythmic intervals, white foam spreading across the wet sand before receding. The palm tree fronds on both sides sway softly in a light ocean breeze. Several seagulls glide and circle lazily against the clear blue sky, occasionally dipping lower toward the water. Sunlight glints off the wave crests, casting shifting reflections. Footprints in the warm golden sand remain undisturbed. The wave foam fizzes and dissolves naturally at the waterline. Ambient tropical daylight, consistent lighting throughout."
]

for idx in range(len(prompts)):
    start_time = time.time()

    pipe.generate(
        seed=seed,
        image_path=image_paths[idx],
        prompt=prompt_prefix + " " + prompts[idx],
        negative_prompt=negative_prompt,
        save_result_path=f"save_results/output_{image_paths[idx][:-4]}.mp4"
    )

    end_time = time.time()

    print("Generation Time:", end_time - start_time)
