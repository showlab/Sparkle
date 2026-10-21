import time
import random
import base64

from vllm import LLM, SamplingParams
import os
from io import BytesIO
import json
from tqdm import tqdm
from PIL import Image
import decord
import argparse
from datasets import load_dataset

EDIT_TYPE_KEYWORDS = {
    "season": {
        "spring": [
            "cherry blossoms in full bloom", "fresh green sprouting leaves", "blooming wildflowers",
            "light spring rain with puddles", "pollen drifting in warm breeze", "melting snow revealing grass",
        ],
        "summer": [
            "lush green canopy under bright sunlight", "heat haze shimmering on ground",
            "intense midday sun with sharp shadows", "tropical foliage swaying in warm wind",
            "golden wheat field under clear blue sky", "buzzing summer meadow",
        ],
        "autumn": [
            "golden and red falling leaves", "harvest fields with dried crops",
            "misty autumn morning", "amber-toned forest canopy", "overcast sky with scattered leaves blowing",
            "pumpkin-dotted farmland with bare branches",
        ],
        "winter": [
            "snow-covered landscape with gentle snowfall", "frost-coated branches and icy surfaces",
            "bare trees in a blizzard", "frozen lake with swirling snow dust",
            "cold grey overcast sky with visible breath", "icicle-hung rooftops with drifting snow",
        ],
    },
    "time": {
        "dawn": [
            "soft pink and orange sunrise sky", "morning mist rolling over terrain",
            "dew drops glistening on surfaces", "first rays of light breaking through",
        ],
        "midday": [
            "harsh overhead sunlight with sharp shadows", "clear blue sky with bright exposure",
            "high-contrast noon lighting", "sun-bleached environment",
        ],
        "golden_hour": [
            "warm amber golden hour light", "long dramatic shadows stretching across ground",
            "soft warm glow on all surfaces", "sun low on horizon casting orange tones",
        ],
        "dusk": [
            "purple-orange gradient twilight sky", "silhouette lighting against fading sun",
            "dimming sky with early visible stars", "dusky blue-violet atmosphere",
        ],
        "night": [
            "moonlit scene with cool blue tones", "starry sky with ambient moonlight",
            "city lights and neon reflections at night", "dark scene with dramatic artificial lighting",
        ],
        "blue_hour": [
            "deep blue atmospheric tone", "calm transitional light between day and night",
            "serene blue-cast environment", "soft diffused blue ambient light",
        ],
    },
    "location": {
        "urban": [
            "busy city street with traffic", "rooftop overlooking skyline",
            "subway platform with passing trains", "narrow alley with flickering signs",
            "city plaza with flowing fountain", "highway overpass with moving vehicles below",
        ],
        "nature": [
            "dense forest with swaying trees", "sandy beach with rolling ocean waves",
            "mountain ridge with drifting clouds", "arid desert with wind-blown sand",
            "tranquil lake with gentle ripples", "waterfall cascading over mossy rocks",
        ],
        "indoor": [
            "cozy living room with flickering fireplace", "photography studio with softbox lighting",
            "industrial warehouse with hanging chain lights", "quiet library with dust motes in light",
            "busy restaurant kitchen with steam rising", "modern gym with reflective mirrors",
        ],
        "landmark": [
            "ancient stone ruins with wind-swept grass", "ornate temple courtyard with incense smoke",
            "medieval castle with waving banners", "suspension bridge with swaying cables over river",
            "coastal lighthouse with crashing waves", "grand monument plaza with flowing crowds",
        ],
        "exotic": [
            "underwater coral reef with swimming fish", "space station interior with floating particles",
            "volcanic terrain with rising steam and embers", "crystal ice cave with dripping meltwater",
            "floating market with bobbing boats on river", "bioluminescent cave with glowing organisms",
        ],
        "rural": [
            "farmland with swaying crops in wind", "vineyard rows with rustling leaves",
            "quiet countryside road with passing clouds", "small village with chimney smoke rising",
            "old barn with creaking doors in breeze", "open prairie with tall grass waving",
        ],
    },
    "style": {
        "art_style": [
            "oil painting style with visible brushstroke textures", "watercolor wash with bleeding edges",
            "ukiyo-e Japanese woodblock print style", "impressionist soft dappled light style",
            "bold pop art with halftone dots", "charcoal sketch with smudged shading",
        ],
        "era": [
            "medieval stone-and-timber village setting", "Victorian-era gaslit cobblestone street",
            "1920s art deco grand ballroom", "1980s retro neon-soaked arcade",
            "futuristic cyberpunk neon cityscape with holograms", "1960s mod interior with geometric patterns",
        ],
        "cinematic": [
            "film noir high-contrast black and white with rain", "Wes Anderson pastel symmetrical composition",
            "sci-fi dystopian industrial wasteland", "horror foggy atmosphere with dim flickering lights",
            "Blade Runner neon-rain urban night", "Miyazaki-esque lush fantastical countryside",
        ],
        "render_style": [
            "low-poly 3D geometric world", "retro pixel art 16-bit environment",
            "claymation stop-motion textured scene", "anime cel-shaded vibrant background",
            "photorealistic HDR ultra-detailed rendering", "voxel-based blocky colorful world",
        ],
        "mood": [
            "dreamlike surreal floating landscape", "dark gothic cathedral atmosphere",
            "whimsical fantasy enchanted forest", "minimalist clean white space",
            "grunge industrial rusty abandoned factory", "ethereal heavenly clouds with light rays",
        ],
    },
}


def build_system_prompt(edit_type: str) -> tuple[str, str]:
    """Build system prompt for the given edit type. Returns (system_prompt, chosen_keyword)."""
    type_keywords = EDIT_TYPE_KEYWORDS[edit_type]
    # randomly pick a sub-category, then a keyword
    sub_cat = random.choice(list(type_keywords.keys()))
    keyword = random.choice(type_keywords[sub_cat])

    # Diverse instruction starters — randomly pick one to encourage variety
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
    starter = random.choice(INSTRUCTION_STARTERS)

    # Explicit dynamic cue words — randomly pick 2-3 to embed
    DYNAMIC_CUES = [
        "dynamic", "lively", "animated", "in motion", "alive with movement",
        "with natural motion", "with gentle movement", "with flowing energy",
        "with subtle animation", "with vivid motion",
    ]
    chosen_cues = random.sample(DYNAMIC_CUES, k=random.randint(2, 3))
    cue_str = ", ".join(chosen_cues)

    base_instruction = (
        "You are a professional prompt engineer for image editing models. "
        "Given the first frame of a video, you will generate an EDIT INSTRUCTION that tells an image editing model "
        "how to change the background of this image.\n\n"
        "CRITICAL REQUIREMENTS:\n"
        f"1. Begin your edit instruction with: \"{starter} ...\". Do NOT describe the foreground subject.\n"
        f"2. The new background must feel {cue_str}. Include at least one moving/dynamic element "
        "(e.g., swaying trees, drifting clouds, flowing water, flickering lights, falling particles, rippling reflections). "
        "NEVER produce a completely static background.\n"
        "3. Keep the edit instruction concise: 1-3 sentences.\n"
    )

    edit_type_instructions = {
        "season": (
            f"TASK: Write an edit instruction to change the background to a different season.\n"
            f"Target season/atmosphere: **{keyword}**\n"
            f"The instruction should specify seasonal vegetation, weather, and lighting changes. "
            f"Include at least one dynamic seasonal element (falling leaves, snowfall, rain, blowing petals, etc.)."
        ),
        "time": (
            f"TASK: Write an edit instruction to shift the background to a different time of day.\n"
            f"Target time/lighting: **{keyword}**\n"
            f"The instruction should specify how the sky, lighting, shadows, and ambient atmosphere change. "
            f"Include at least one dynamic time-related element (drifting clouds, flickering streetlights, shifting light rays, etc.)."
        ),
        "location": (
            f"TASK: Write an edit instruction to replace the background with a completely different location.\n"
            f"Target location/setting: **{keyword}**\n"
            f"The instruction should describe the new environment with spatial depth and atmosphere. "
            f"Include at least one dynamic location-specific element (waves, wind-blown foliage, drifting smoke, flowing water, etc.)."
        ),
        "style": (
            f"TASK: Write an edit instruction to re-style the background in a different artistic style.\n"
            f"Target style: **{keyword}**\n"
            f"The instruction should describe how the background appearance transforms while keeping spatial layout coherent. "
            f"Include at least one dynamic stylistic element (animated textures, flickering effects, floating particles, etc.)."
        ),
    }

    system_prompt = base_instruction + "\n" + edit_type_instructions[edit_type]
    return system_prompt, f"{sub_cat}: {keyword}"


def extract_first_frame_as_base64(video_bytes: BytesIO) -> str:
    """Extract the first frame from video bytes, return as base64-encoded JPEG."""
    video_bytes.seek(0)
    vr = decord.VideoReader(video_bytes)
    first_frame = vr[0].asnumpy()  # H, W, C numpy array
    img = Image.fromarray(first_frame)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_shards", type=int, default=None)
    parser.add_argument("--shard_index", type=int, default=None)
    parser.add_argument("--edit_type", type=str, required=True,
                        choices=["season", "time", "location", "style"],
                        help="Type of background edit: season | time | location | style")
    args = parser.parse_args()

    dset_path = "../location_train_preview100/stage1_fixed_camera_unimatch_qwen3vl.csv"

    print("Loading dataset with huggingface datasets...")

    dataset = load_dataset("csv", data_files=dset_path, split="train")

    assert (
        (args.num_shards is None and args.shard_index is None) or
        (args.num_shards is not None and args.shard_index is not None)
    )

    if args.num_shards is not None:
        dataset = dataset.shard(num_shards=args.num_shards, index=args.shard_index, contiguous=True)
        print(f"Current eval shard: {args.shard_index} / {args.num_shards}, it has {len(dataset)} samples.")

    # Initialize model — image-only mode (no video)
    model_name = "Qwen/Qwen3-VL-32B-Instruct"
    print(f"Loading {model_name}...")
    llm = LLM(
        model=model_name,
        tensor_parallel_size=8,
        mm_encoder_tp_mode="data",
        enable_expert_parallel=False,
        async_scheduling=True,
        limit_mm_per_prompt={"image": 1, "video": 0},  # image-only mode: 1 image, no video
        disable_log_stats=True,
    )

    save_path = (
        f"../location_train_preview100/stage2_generated_prompts/"
        f"{args.edit_type}_{args.num_shards}_{args.shard_index}.jsonl"
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Stats
    total, success, fail = 0, 0, 0

    with open(save_path, 'w') as fout:
        for idx, item in tqdm(enumerate(dataset), total=len(dataset)):
            total += 1

            # --- Read local video & extract first frame ---
            try:
                video_path = os.path.join("../location_train_preview100", item["src_video"])

                with open(video_path, "rb") as f:
                    video_bytes = BytesIO(f.read())

                base64_image = extract_first_frame_as_base64(video_bytes)
                base64_image_url = f"data:image/jpeg;base64,{base64_image}"

            except Exception as e:
                print(f"[Local Video/Frame error] item={item}, err={e}")
                fail += 1
                continue

            # --- Build system prompt with random keyword ---
            system_prompt, chosen_keyword = build_system_prompt(args.edit_type)

            user_prompt = (
                "Here is the first frame of a video. Analyze the current background, "
                "then write an edit instruction for an image editing model to replace the background, "
                "following the system instructions.\n\n"
                "Output your response strictly in the following JSON format without any markdown blocks or additional text:\n"
                "{\n"
                '    "original_scene": "Brief description of the current background in the frame",\n'
                '    "edit_instruction": "The edit instruction for background replacement (e.g., Change the background to ...)"\n'
                "}"
            )

            messages = [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": base64_image_url,
                            }
                        },
                        {
                            "type": "text",
                            "text": user_prompt,
                        }
                    ]
                }
            ]

            sampling_params = SamplingParams(
                max_tokens=1024,
                temperature=0.7,  # some creativity for diverse prompts
            )

            # --- Inference ---
            try:
                outputs = llm.chat(
                    messages=messages,
                    sampling_params=sampling_params,
                    chat_template_content_format="openai",
                    use_tqdm=False,
                )
                resp = outputs[0].outputs[0].text.strip()
            except Exception as e:
                print(f'[Inference error] item={item}, err={e}')
                fail += 1
                continue

            # --- Parse JSON response ---
            try:
                resp_json = json.loads(resp)
            except json.JSONDecodeError as e:
                print(f"[JSON parse failed] output={resp[:200]}..., err={e}")
                fail += 1
                continue

            ret = {
                "video_path": item['src_video'],
                "edit_type": args.edit_type,
                "chosen_keyword": chosen_keyword,
                "generation": resp_json,
            }

            fout.write(json.dumps(ret) + '\n')
            fout.flush()

            success += 1

    print(f"Done. total={total}, success={success}, fail={fail}")