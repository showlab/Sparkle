import os
import csv
import json
import time
import argparse
import base64
from io import BytesIO
from collections import defaultdict
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor

import openai
from PIL import Image

# 6-dim rubric tailored for video background replacement
BACKGROUND_REPLACEMENT = """
You are a data rater specializing in grading video background replacement. You will be given two videos (source and edited) and the editing instruction. Your task is to evaluate the result on a 5-point scale across six dimensions:

Instruction Compliance
1. No change, or background entirely unrelated to the prompt, or foreground also replaced/distorted such that the edit fails as a whole.
2. Background only partially matches prompt content or style; major requested elements wrong or missing; or foreground noticeably altered.
3. Main background concept matches but with missing/extra elements, wrong sub-style, or partial spill onto the subject.
4. Requested background fully present and consistent with the prompt; only minor mismatches in tone, detail, or atmosphere.
5. Background exactly matches the prompt in content, style, mood, and any specified dynamics; foreground untouched.

Overall Visual Quality
This dimension covers global image quality AND foreground-background harmonization. The lighting, color temperature, and shadows on the foreground must match the new background environment. For example, when the prompt changes the time of day (e.g. day to night, noon to sunset), keeping the original daytime lighting on the foreground while the background is dark is a major harmonization failure. The same applies to season, location, and style edits that imply different ambient light.
1. Severe artefacts throughout (tearing, posterisation, color banding, heavy flicker), OR foreground lighting is grossly inconsistent with the new background (e.g. brightly lit subject against a night scene, conflicting light directions, no shadow adaptation).
2. Clear visual degradation (persistent blur, noise, unstable colors), OR obvious lighting / color-temperature mismatch between foreground and background visible at first glance.
3. Watchable but with visible flaws on closer look: occasional flicker, mild compression artefacts, soft regions, OR partial harmonization where the foreground tone is in the right direction but not fully matched to the background.
4. Clean output with only minor issues when zoomed in or paused; foreground lighting and color grading are well aligned with the background, with only subtle discrepancies.
5. Indistinguishable from real captured footage: sharp, stable, well-graded across the entire clip, with foreground lighting, color temperature, and shadows fully harmonized with the new background environment.

Foreground Integrity
1. Foreground severely damaged: missing limbs/parts, large holes, replaced with a different subject, or shape collapsed.
2. Noticeable foreground damage: partial erosion by background, distorted contours, identity drift across frames.
3. Foreground mostly preserved but with visible defects: edge halos, slight shape deformation, occasional color bleed.
4. Foreground well preserved with only minute edge artefacts; shape and identity stable throughout.
5. Foreground perfectly preserved: every pixel of shape, texture, and identity intact across all frames.

Foreground Motion Consistency
1. Foreground motion completely different from source: actions replaced, frozen, looped, or temporally scrambled.
2. Major motion deviations: different gestures, dropped actions, or strong temporal jitter not present in source.
3. Same general action is recognizable but with timing drift, trajectory shifts, or inconsistent speed versus source.
4. Motion closely tracks the source with only minor temporal misalignment or subtle smoothing.
5. Foreground motion is identical to the source video in trajectory, timing, and articulation, frame by frame.

Background Dynamics (Liveness)
This dimension measures whether the background motion matches the intensity and character implied by the prompt. The bar is appropriateness to the prompt, not absolute amount of motion. A "gentle swaying grass" prompt rendered as subtle wind-like sway is fully correct and should receive a high score; the same subtle motion for a "rushing waterfall" prompt is severely under-rendered.
1. Background motion contradicts the prompt: completely static when the prompt implies any motion, or wrong type/direction of motion (e.g. crashing waves rendered as a still pond).
2. Motion intensity is far below what the prompt implies (e.g. a "rushing river" rendered as barely moving water), or required dynamics are largely absent.
3. Motion type is in the right direction but noticeably under- or over-rendered, OR motion exists but feels stiff and unnatural.
4. Motion intensity and character are well matched to the prompt, with only minor stiffness, small frozen patches, or slight over/under rendering.
5. Background motion perfectly matches the prompt in both intensity and character, rendered naturally and continuously throughout the clip — gentle prompts receive gentle motion, energetic prompts receive energetic motion.
Special case: if the prompt explicitly asks for a static background (e.g. "still photo", "frozen scene", "no motion"), a faithfully static background scores 5 and any unwanted motion lowers the score accordingly.

Background Visual Quality
1. Background severely degraded: melting structures, broken geometry, heavy blur, or incoherent textures.
2. Clear distortion or blur in major background regions; structures wobble or warp over time.
3. Acceptable background with visible imperfections: soft textures, mild geometric inconsistency, minor temporal warping.
4. High-quality background with only minor issues on close inspection; geometry and textures stable.
5. Background is sharp, geometrically coherent, and temporally stable; on par with real footage.

Constraints
The scores for Overall Visual Quality, Foreground Integrity, Foreground Motion Consistency, Background Dynamics, and Background Visual Quality must not exceed the score for Instruction Compliance.

Example Response Format
Brief reasoning: No more than 30 words.
Instruction Compliance: 1-5.
Overall Visual Quality: 1-5.
Foreground Integrity: 1-5.
Foreground Motion Consistency: 1-5.
Background Dynamics: 1-5.
Background Visual Quality: 1-5.

Editing instruction is: {edit_prompt}.

Below are the videos before and after editing:
"""

# Order matters: index 0 is the cap dimension (Instruction Compliance).
SCORE_KEYS = [
    ("Instruction Compliance:", "instruction"),
    ("Overall Visual Quality:", "visual"),
    ("Foreground Integrity:", "fg_integrity"),
    ("Foreground Motion Consistency:", "fg_motion"),
    ("Background Dynamics:", "bg_dynamics"),
    ("Background Visual Quality:", "bg_visual"),
]
DIM_NAMES = [k for _, k in SCORE_KEYS]
N_DIMS = len(DIM_NAMES)

# We use Azure-hosted Gemini via the OpenAI-compatible API for its convenient
# concurrency story. If you have direct access to the Gemini API, feel free to
# swap this client for the native google-genai SDK; the request payload only
# requires (system prompt, source video, edited video) so the adaptation is
# straightforward.
GEMINI_API = openai.AzureOpenAI(
    azure_endpoint=os.environ['AZURE_ENDPOINT'],
    api_version="2024-03-01-preview",
    api_key=os.environ['GEMINI_API_KEY'],
)


def create_content(user_inputs):
    content = []
    for user_input in user_inputs:
        if isinstance(user_input, str):
            if user_input.endswith('.mp4') and os.path.isfile(user_input):
                with open(user_input, 'rb') as f:
                    video_bytes = f.read()
                content.append({
                    'type': 'image_url',
                    'image_url': {
                        'url': f"data:video/mp4;base64,{base64.b64encode(video_bytes).decode('utf-8')}"
                    }
                })
            else:
                content.append({'type': 'text', 'text': user_input})
        elif isinstance(user_input, Image.Image):
            if user_input.mode != 'RGB':
                user_input = user_input.convert('RGB')
            quality = 100
            while quality > 5:
                buffered = BytesIO()
                user_input.save(buffered, format="JPEG", quality=quality)
                if len(buffered.getvalue()) < 18 * 1024 * 1024:
                    break
                quality -= 5
            else:
                raise RuntimeError("Image too large even at low quality")
            content.append({
                'type': 'image_url',
                'image_url': {
                    'url': f"data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"
                }
            })
        else:
            raise NotImplementedError(f"Unsupported user input type: {type(user_input)}")
    return content


def check_format(out):
    """Parse 6-dim scores; clip dims 2-6 to Instruction Compliance as a soft cap."""
    try:
        scores = {}
        for line in out.splitlines():
            line = line.strip()
            for prefix, key in SCORE_KEYS:
                if line.startswith(prefix):
                    val = int(float(line.split(':')[-1].strip()))
                    if val not in range(1, 6):
                        return False
                    scores[key] = val
                    break
        if len(scores) != N_DIMS:
            return False
        cap = scores[DIM_NAMES[0]]
        for key in DIM_NAMES[1:]:
            if scores[key] > cap:
                scores[key] = cap
        return [scores[k] for k in DIM_NAMES]
    except Exception as e:
        print(e)
        return False


def run_two_videos_gemini(video_path_1, video_path_2, sys_prompt, meta_info, idx):
    content = create_content([sys_prompt, video_path_1, video_path_2])

    max_retries = 20
    for attempt in range(max_retries):
        try:
            response = GEMINI_API.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": content}],
                seed=42,
                temperature=0,
                timeout=60
            ).choices[0].message.content

            result = response.strip()
            scores = check_format(result)

            if scores:
                print(f"Task {idx} Success on attempt {attempt + 1}")
                return scores, result, meta_info

        except Exception as e:
            print(f"Task {idx} Attempt {attempt + 1} failed: {e}")
            time.sleep(3)

    return None


def batch_process_videos(tasks, max_workers=20):
    """tasks: List of (src_path, edited_path, sys_prompt, meta_info, idx)."""
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_two_videos_gemini, *task) for task in tasks]
    for future in concurrent.futures.as_completed(futures):
        results.append(future.result())
    return results


def avg_score_by_edit_type(jsonl_path):
    dim_sums = defaultdict(lambda: [0.0] * N_DIMS)
    record_count = defaultdict(int)

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                edit_type = record.get("edit_type")
                scores = record.get("scores", [])
                if edit_type and isinstance(scores, list) and len(scores) == N_DIMS:
                    for i in range(N_DIMS):
                        dim_sums[edit_type][i] += scores[i]
                    record_count[edit_type] += 1
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON on line {line_num}: {e}")

    avg_scores = {}
    for et, sums in dim_sums.items():
        n = record_count[et]
        if n == 0:
            continue
        means = [s / n for s in sums]
        avg_scores[et] = {
            **{name: m for name, m in zip(DIM_NAMES, means)},
            "overall": sum(means) / N_DIMS,
            "count": n,
        }
    return avg_scores


def _short(name):
    """Compact column header for a dim name, e.g. 'fg_integrity' -> 'FgIn'."""
    parts = name.split("_")
    if len(parts) == 1:
        return parts[0][:3].capitalize()
    return "".join(p[:2].capitalize() for p in parts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default="gemini-2.5-pro",
                        choices=["gemini-2.5-pro"],
                        help="Gemini model to use for evaluation")
    parser.add_argument('--video_paths', type=str, nargs='+', required=True,
                        help="List of inference output directories to evaluate, e.g. "
                             "/path/to/sparkle_bench_results/")
    parser.add_argument('--edit_types', type=str, nargs='+',
                        default=['location', 'season', 'style', 'time'])
    parser.add_argument('--bench_root', type=str,
                        default='data/Sparkle-Bench')
    args = parser.parse_args()
    model_id = args.model_id

    # ============== Step 1: dispatch + score ==============
    for save_dir in args.video_paths:
        for edit_type in args.edit_types:
            score_path = f"{save_dir}/{edit_type}_{model_id}_sparkle_score.jsonl"
            if os.path.exists(score_path):
                print(f"[skip] {score_path} already exists")
                continue

            bench_csv = f"{args.bench_root}/{edit_type}_bench.csv"
            meta_jsonl = f"{args.bench_root}/{edit_type}_metadata.jsonl"
            if not os.path.exists(bench_csv) or not os.path.exists(meta_jsonl):
                print(f"[skip] missing bench files for {edit_type}: "
                      f"{bench_csv} or {meta_jsonl}")
                continue

            # Build id -> chosen_keyword lookup from metadata jsonl
            id2keyword = {}
            with open(meta_jsonl, "r") as f:
                for line in f:
                    rec = json.loads(line)
                    id2keyword[rec["id"]] = rec["metadata"]["chosen_keyword"]

            # Read bench csv: prompt + original_video
            with open(bench_csv, "r") as f:
                rows = list(csv.DictReader(f))

            tasks = []
            n_missing_edited, n_missing_source, n_missing_meta = 0, 0, 0
            for idx, row in enumerate(rows):
                prompt = row["prompt"]
                original_video = row["original_video"]  # e.g. source_videos/location/Sparkle_location_010913.mp4
                item_id = os.path.splitext(os.path.basename(original_video))[0]
                source_path = os.path.join(args.bench_root, original_video)

                chosen_keyword = id2keyword.get(item_id)
                if chosen_keyword is None:
                    n_missing_meta += 1
                    continue

                subtheme, scene = chosen_keyword.split(": ")
                scene_key = scene.replace(" ", "_")
                edited_path = f"{save_dir}/{edit_type}/{subtheme}---{scene_key}/{item_id}_edited.mp4"

                if not os.path.exists(edited_path):
                    n_missing_edited += 1
                    continue
                if not os.path.exists(source_path):
                    n_missing_source += 1
                    continue

                sys_prompt = BACKGROUND_REPLACEMENT.format(edit_prompt=prompt)
                meta_info = {
                    "id": item_id,
                    "prompt": prompt,
                    "edit_type": edit_type,
                    "subtheme": subtheme,
                    "scene": scene,
                }
                tasks.append((source_path, edited_path, sys_prompt, meta_info, idx))

            print(f"[{save_dir}][{edit_type}] dispatching {len(tasks)} tasks "
                  f"(missing_edited={n_missing_edited}, "
                  f"missing_source={n_missing_source}, "
                  f"missing_meta={n_missing_meta})")

            res = batch_process_videos(tasks)

            with open(score_path, 'w') as f:
                for item in res:
                    if item:
                        scores, result, meta_info = item
                        record = {**meta_info, "scores": scores, "result": result}
                        f.write(json.dumps(record, ensure_ascii=False) + '\n')

    # ============== Step 2: aggregate stats ==============
    for save_dir in args.video_paths:
        print(f"\n=== {save_dir} ===")
        macro_sums = [0.0] * N_DIMS
        macro_buckets = 0
        all_n = 0
        for edit_type in args.edit_types:
            file_path = f"{save_dir}/{edit_type}_{model_id}_sparkle_score.jsonl"
            if not os.path.exists(file_path):
                continue
            averages = avg_score_by_edit_type(file_path)
            v = averages.get(edit_type)
            if v is None:
                print(f"  {edit_type}: no valid records")
                continue
            dim_str = " | ".join(f"{_short(name)} {v[name]:.2f}" for name in DIM_NAMES)
            print(f"  {edit_type:<10s} (n={v['count']:4d}) "
                  f"Overall {v['overall']:.2f} | {dim_str}")
            for i, name in enumerate(DIM_NAMES):
                macro_sums[i] += v[name]
            macro_buckets += 1
            all_n += v['count']

        if macro_buckets:
            macro_means = [s / macro_buckets for s in macro_sums]
            macro_overall = sum(macro_means) / N_DIMS
            dim_str = " | ".join(f"{_short(name)} {m:.2f}"
                                 for name, m in zip(DIM_NAMES, macro_means))
            print(f"  {'MACRO':<10s} (n={all_n:4d}) "
                  f"Overall {macro_overall:.2f} | {dim_str}")
