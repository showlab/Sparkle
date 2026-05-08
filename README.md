# <img src="assets/sparkle-icon.png" alt="" height="56" style="vertical-align: middle;"/><img src="assets/sparkle-text.svg" alt="Sparkle" height="56" style="vertical-align: middle;"/>: Realizing Lively Instruction-Guided Video Background Replacement via Decoupled Guidance

[Ziyun Zeng](https://stdkonjac.icu/), Yiqi Lin, [Guoqiang Liang](https://ethanliang99.github.io/), and [Mike Zheng Shou](https://cde.nus.edu.sg/ece/staff/shou-zheng-mike/)

[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2605.06535)
[![Project Page](https://img.shields.io/badge/Website-Project%20Page-green?logo=googlechrome&logoColor=white)](https://showlab.github.io/Sparkle/)
[![Code](https://img.shields.io/badge/Code-GitHub%20Repo-blue?logo=github)](https://github.com/showlab/Sparkle)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-Sparkle-orange.svg)](https://huggingface.co/datasets/stdKonjac/Sparkle)
[![Benchmark](https://img.shields.io/badge/🤗%20Benchmark-Sparkle--Bench-orange.svg)](https://huggingface.co/datasets/stdKonjac/Sparkle-Bench)
[![Model](https://img.shields.io/badge/🤗%20Model-Kiwi--Sparkle-orange.svg)](https://huggingface.co/stdKonjac/Kiwi-Sparkle-720P-81F)


## 📦 Dataset

**Sparkle** is a large-scale video background replacement dataset comprising ~140K high-quality source–edited video pairs. It is fully open-sourced at [🤗stdKonjac/Sparkle](https://huggingface.co/datasets/stdKonjac/Sparkle). For full methodology and dataset details, please refer to [our paper](https://arxiv.org/abs/2605.06535).

The dataset is organized into **five themes** along different background-change axes:

| Theme      | Description                                                                                                                      |
| ---------- |----------------------------------------------------------------------------------------------------------------------------------|
| `location` | Background replaced with a different physical environment (rural, nature, landmark, ...).                                        |
| `season`   | Background changed across seasons (spring, summer, autumn, winter).                                                              |
| `time`     | Background changed across times of day (dawn, dusk, night, ...).                                                                 |
| `style`    | Background restyled (era, mood, cinematic, ...).                                                                                 |
| `openve3m` | A re-creation of the OpenVE-3M background-replacement subset using our pipeline, retained for direct comparison with prior work. |

### 🗂️ Repository Structure

```
Sparkle/
├── README.md
├── prompts/                                        # training annotations + dataset-viewer source
│   ├── location_train.csv                          # 4 columns: prompt, src_video, tgt_video, task
│   ├── location_train_metadata.jsonl               # per-task metadata (edit_type, subtheme, original scene)
│   ├── season_train.csv
│   ├── season_train_metadata.jsonl
│   ├── time_train.csv
│   ├── time_train_metadata.jsonl
│   ├── style_train.csv
│   ├── style_train_metadata.jsonl
│   ├── openve3m_train.csv
│   └── openve3m_train_metadata.jsonl
│
├── location/                                       # online preview: first 100 samples
│   ├── source_video/
│   │   ├── Sparkle_location_000000.mp4
│   │   └── ... (100 files)
│   └── edited_video/
│       ├── Sparkle_location_000000.mp4
│       └── ... (100 files)
├── season/                                         # same structure as location/
├── time/
├── style/
├── openve3m/
│
├── location_source_video_part00.tar                # full corpus, sharded into ~5GB tars
├── location_source_video_part01.tar
├── location_edited_video_part00.tar
├── ...
├── season_*_partXX.tar
├── time_*_partXX.tar
├── style_*_partXX.tar
├── openve3m_*_partXX.tar
│
└── intermediate_data/                              # pipeline intermediates (described below)
    └── ...
```

### 🧾 Training Data Format

We follow the training data format of [Kiwi-Edit](https://github.com/showlab/Kiwi-Edit) for direct compatibility with downstream training pipelines.

Each theme's annotations live in `prompts/{edit_type}_train.csv`, a four-column table:

| Column      | Description |
| ----------- | ----------- |
| `prompt`    | The natural-language editing instruction. |
| `src_video` | Path to the source video, e.g. `location/source_video/Sparkle_location_000000.mp4`. |
| `tgt_video` | Path to the edited video, e.g. `location/edited_video/Sparkle_location_000000.mp4`. |
| `task`      | The unique sample id, e.g. `Sparkle_location_000000`. Joins to the `id` field in the JSONL metadata. |

Per-task auxiliary metadata is stored alongside in `prompts/{edit_type}_train_metadata.jsonl`. Each line is one sample:

```json
{
  "id": "Sparkle_location_000000",
  "prompt": "Shift the background to a rooftop overlooking a modern city skyline at dusk, ...",
  "metadata": {
    "edit_type": "location",
    "chosen_keyword": "urban: rooftop overlooking skyline",
    "original_scene": "A cobblestone street in a historical European city, ..."
  }
}
```

| Field                      | Description                                                                                                              |
| -------------------------- |--------------------------------------------------------------------------------------------------------------------------|
| `id`                       | Sample id, matches the `task` column in the CSV.                                                                         |
| `prompt`                   | Same as the `prompt` column in the CSV.                                                                                  |
| `metadata.edit_type`       | One of the five themes: `location` / `season` / `time` / `style` / `openve3m` (denoted as `openve3m_background_change`). |
| `metadata.chosen_keyword`  | The `subtheme: scene` label (e.g. `"urban: rooftop overlooking skyline"`). Not available for the `openve3m` theme.       |
| `metadata.original_scene`  | A description of the source video's first frame.                                                                         |

### 👀 Online Preview

The first 100 samples of every theme are stored as uncompressed `.mp4` files under `{edit_type}/source_video/` and `{edit_type}/edited_video/`, and can be played directly in the browser without downloading the full corpus.

For example, for the task `Sparkle_location_000000` (the first row in the **location** theme of the dataset viewer), you can directly browse its [Source Video](https://huggingface.co/datasets/stdKonjac/Sparkle/blob/main/location/source_video/Sparkle_location_000000.mp4) and [Edited Video](https://huggingface.co/datasets/stdKonjac/Sparkle/blob/main/location/edited_video/Sparkle_location_000000.mp4).

The dataset viewer at the top of the HF page lets you scroll through all five themes and read the corresponding prompts inline.

### ⬇️ Downloading the Full Corpus

The full ~140K-sample corpus is sharded into ~5GB `.tar` archives at the repository root, named `{edit_type}_{source_video|edited_video}_partXX.tar`.

**Step 1 — Download the tar shards.** Download everything (recommended for full reproduction):

```bash
hf download stdKonjac/Sparkle --repo-type=dataset --local-dir ./Sparkle
```

Or only a single theme (e.g. `location`):

```bash
hf download stdKonjac/Sparkle \
  --repo-type=dataset \
  --local-dir ./Sparkle \
  --include "location_*.tar" "prompts/location_*"
```

Or only the source videos of a theme:

```bash
hf download stdKonjac/Sparkle \
  --repo-type=dataset \
  --local-dir ./Sparkle \
  --include "location_source_video_*.tar"
```

**Step 2 — Extract the tars.** Each tar is **self-contained**: its internal paths are `{edit_type}/{source_video|edited_video}/{task}.mp4`, so extracting any subset of shards in place will populate the corresponding folders correctly. There is **no need to concatenate the parts** before extraction.

```bash
cd ./Sparkle
for f in *.tar; do tar -xf "$f"; done
```

After extraction, the directory layout matches the online preview structure, and the relative paths in `prompts/{edit_type}_train.csv` (e.g. `location/source_video/Sparkle_location_000000.mp4`) will resolve directly.

<details>
<summary><h3 style="display: inline">🧪 Pipeline Intermediates</h3></summary>

To support **full reproducibility, transparency, and downstream research**, we additionally release every intermediate artifact produced by the 5-stage Sparkle data pipeline (see *Figure 2: Data Pipeline* in [our paper](https://arxiv.org/abs/2605.06535)) under `intermediate_data/`. **The first 100 samples of every theme are uncompressed and previewable directly in the browser**, mirroring the layout of the `{edit_type}/` preview folders described above.

Taking `Sparkle_location_000000` as a running example, the artifact layout looks like:

```
Sparkle/
└── intermediate_data/
    └── location/
        ├── source_frame0/                          # Stage 2 input: 0-th frame of the source video
        │   └── Sparkle_location_000000.png
        ├── edited_frame0/                          # Stage 2 output: first frame after preliminary background replacement
        │   └── Sparkle_location_000000.png
        ├── edited_frame0_foreground_removed/       # Stage 3 intermediate: foreground-removed clean background image
        │   └── Sparkle_location_000000.png
        ├── edited_background_video/                # Stage 3 output: 81-frame pure background video (no foreground)
        │   └── Sparkle_location_000000.mp4
        ├── source_video_mask/                      # Stage 4 output: BAIT-tracked foreground mask (packed bits)
        │   └── Sparkle_location_000000.npz
        └── edited_video_canny/                     # Stage 5 intermediate: decoupled foreground + background Canny edges
            └── Sparkle_location_000000.mp4
```

For the same task `Sparkle_location_000000`, every artifact is browsable online:

| Pipeline stage | Artifact                                         | Preview                                                                                                                                                                                                                |
|----------------|--------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Stage 2 (in)   | Source first frame                               | [`source_frame0/Sparkle_location_000000.png`](https://huggingface.co/datasets/stdKonjac/Sparkle/blob/main/intermediate_data/location/source_frame0/Sparkle_location_000000.png)                                        |
| Stage 2 (out)  | Preliminarily edited first frame                 | [`edited_frame0/Sparkle_location_000000.png`](https://huggingface.co/datasets/stdKonjac/Sparkle/blob/main/intermediate_data/location/edited_frame0/Sparkle_location_000000.png)                                        |
| Stage 3 (mid)  | Foreground-removed clean background image        | [`edited_frame0_foreground_removed/Sparkle_location_000000.png`](https://huggingface.co/datasets/stdKonjac/Sparkle/blob/main/intermediate_data/location/edited_frame0_foreground_removed/Sparkle_location_000000.png) |
| Stage 3 (out)  | Pure background video (81 frames, no foreground) | [`edited_background_video/Sparkle_location_000000.mp4`](https://huggingface.co/datasets/stdKonjac/Sparkle/blob/main/intermediate_data/location/edited_background_video/Sparkle_location_000000.mp4)                   |
| Stage 4        | BAIT-tracked foreground mask                     | [`source_video_mask/Sparkle_location_000000.npz`](https://huggingface.co/datasets/stdKonjac/Sparkle/blob/main/intermediate_data/location/source_video_mask/Sparkle_location_000000.npz)                                |
| Stage 5 (mid)  | Decoupled foreground + background Canny edges    | [`edited_video_canny/Sparkle_location_000000.mp4`](https://huggingface.co/datasets/stdKonjac/Sparkle/blob/main/intermediate_data/location/edited_video_canny/Sparkle_location_000000.mp4)                              |

**Loading the foreground mask.** The masks in `source_video_mask/` are bit-packed for storage efficiency. Each `.npz` file contains two arrays: `mask` (a `np.uint8` array of bits) and `shape` (the original `(T, H, W)` mask shape, where ``T ≤ 81``). Unpack with:

```python
import numpy as np

def load_mask(mask_path: str) -> np.ndarray:
    data = np.load(mask_path)
    packed_mask = data["mask"]
    shape = tuple(int(s) for s in data["shape"])
    total = shape[0] * shape[1] * shape[2]
    video_mask = np.unpackbits(packed_mask)[:total].reshape(shape).astype(bool)
    return video_mask  # boolean array of shape (T, H, W)
```

**Downloading the full intermediates.** Like the main corpus, the full intermediates for every theme are sharded into ~5GB `.tar` archives, stored under `intermediate_data/` and named `{edit_type}_{subdir}_partXX.tar` where `{subdir}` is one of the six folder names above. Download and extract them as follows:

```bash
# Download all intermediates for a single theme (e.g. location)
hf download stdKonjac/Sparkle \
  --repo-type=dataset \
  --local-dir ./Sparkle \
  --include "intermediate_data/location_*_part*.tar"

# Extract in place; tar-internal paths are {edit_type}/{subdir}/{file},
# so the working directory must be intermediate_data/ for the layout to align.
cd ./Sparkle/intermediate_data
for f in location_*_part*.tar; do tar -xf "$f"; done
```

After extraction, the layout matches the online preview structure exactly, populating `intermediate_data/location/{source_frame0, edited_frame0, ...}/`.

#### 📋 Per-task Pipeline Metadata

In addition to the per-task artifacts, each theme's `intermediate_data/{edit_type}/` folder also contains five `.jsonl` files recording metadata produced at various stages of the pipeline (e.g., quality scores, foreground grounding labels). These records are useful for **reproducing our quality filtering**, **inspecting per-stage rejection statistics**, or **building stricter / looser variants of Sparkle for downstream research**.

**`edited_frame0_score.jsonl`** records per-sample [EditScore](https://github.com/VectorSpaceLab/EditScore) evaluation of the Stage 2 output (`edited_frame0/{task}.png`). One JSON object per line:

```json
{
  "id": "Sparkle_location_000000",
  "prompt": "Shift the background to a rooftop overlooking a modern city skyline at dusk, ...",
  "editscore": {
    "prompt_following": 9.7,
    "consistency": 8.8,
    "perceptual_quality": 8.5,
    "overall": 8.62887857991077,
    "SC_reasoning": "The edited image perfectly follows the instruction: ...",
    "PQ_reasoning": "The image displays a realistic cityscape with convincing lighting ..."
  }
}
```

| Field                            | Description                                                                  |
|----------------------------------|------------------------------------------------------------------------------|
| `id`                             | Sample id, matches the `task` column in the CSV.                             |
| `prompt`                         | The editing instruction.                                                     |
| `editscore.prompt_following`     | Sub-score (0–10): how well the edit follows the instruction.                 |
| `editscore.consistency`          | Sub-score (0–10): subject and identity consistency with the source frame.    |
| `editscore.perceptual_quality`   | Sub-score (0–10): perceptual quality of the edited image.                    |
| `editscore.overall`              | Aggregated overall score. **We filter out samples with `overall < 8`.**      |
| `editscore.SC_reasoning`         | Free-text rationale for the consistency / instruction-following sub-scores. |
| `editscore.PQ_reasoning`         | Free-text rationale for the perceptual-quality sub-score.                    |

**`edited_frame0_foreground_removed_score.jsonl`** records per-sample [EditScore](https://github.com/VectorSpaceLab/EditScore) evaluation of the Stage 3 intermediate output (`edited_frame0_foreground_removed/{task}.png`), measuring the foreground-removal quality. The schema is identical to `edited_frame0_score.jsonl`:

```json
{
  "id": "Sparkle_location_000000",
  "prompt": "...",
  "editscore": {
    "prompt_following": ...,
    "consistency": ...,
    "perceptual_quality": ...,
    "overall": ...,
    "SC_reasoning": "...",
    "PQ_reasoning": "..."
  }
}
```

At this stage we apply a stricter threshold and **filter out samples with `overall < 8.5`** to guarantee a perfectly clean background before the I2V generation that follows.

**`foreground_grounding_r1.jsonl`** records the **first-round VLM grounding** result that compares the source first frame and the Stage 2 edited first frame to identify foreground objects to preserve. This is the labeling step described in Stage 3 of the pipeline. One JSON object per line:

```json
{
  "id": "Sparkle_location_000000",
  "prompt": "Shift the background to a rooftop overlooking a modern city skyline at dusk, ...",
  "edit_type": "location",
  "round1_labels": [
    "woman in brown hat and coat",
    "clasped hands with ring",
    "striped shirt under coat",
    "brown wide-brimmed hat"
  ],
  "round1_objects": [
    {"bbox_2d": [447, 27, 765, 998], "label": "woman in brown hat and coat"},
    {"bbox_2d": [515, 800, 615, 980], "label": "clasped hands with ring"},
    {"bbox_2d": [490, 398, 615, 800], "label": "striped shirt under coat"},
    {"bbox_2d": [505, 27, 710, 258], "label": "brown wide-brimmed hat"}
  ]
}
```

| Field            | Description                                                                                          |
|------------------|------------------------------------------------------------------------------------------------------|
| `id`             | Sample id, matches the `task` column in the CSV.                                                     |
| `prompt`         | The editing instruction.                                                                             |
| `edit_type`      | The theme this sample belongs to (`location` / `season` / `time` / `style` / `openve3m`).            |
| `round1_labels`  | List of foreground-object labels detected by the VLM.                                                |
| `round1_objects` | Per-object detection records; each item has a `bbox_2d` and a `label`.                               |

The bounding boxes are detected on the **source first frame** (`source_frame0/{task}.png`). Since our pipeline preserves the foreground identity and pose during background replacement, these boxes apply equally to the corresponding edited first frame (`edited_frame0/{task}.png`).

<a id="normalize-bbox"></a>

The `bbox_2d` field follows Qwen3-VL's **normalized coordinate format** with values in the range `[0, 1000]`, representing `[x1, y1, x2, y2]` (top-left and bottom-right corners). Convert them to absolute pixel coordinates of the real frame as follows:

```python
def normalize_bbox(bbox, video_width: int, video_height: int):
    """Convert a Qwen3-VL [0, 1000]-normalized bbox to absolute pixel coordinates."""
    x1 = int(bbox[0] / 1000.0 * video_width)
    y1 = int(bbox[1] / 1000.0 * video_height)
    x2 = int(bbox[2] / 1000.0 * video_width)
    y2 = int(bbox[3] / 1000.0 * video_height)

    # Clamp to frame bounds and ensure x1 <= x2, y1 <= y2.
    x1 = max(0, min(min(x1, x2), video_width - 1))
    y1 = max(0, min(min(y1, y2), video_height - 1))
    x2 = max(0, min(max(x1, x2), video_width - 1))
    y2 = max(0, min(max(y1, y2), video_height - 1))
    return x1, y1, x2, y2
```

**`foreground_grounding_r2.jsonl`** records the **second-round VLM grounding** result that produces the temporal anchors for Stage 4 (BAIT Foreground Tracking). Building on the labels from `foreground_grounding_r1.jsonl`, Qwen3-VL is asked to re-locate every Round 1 label on frames sampled at 2 FPS from the source video, yielding per-frame bounding boxes that anchor the subsequent SAM3 multi-pass tracking. One JSON object per line:

```json
{
  "id": "Sparkle_location_000000",
  "prompt": "Shift the background to a rooftop overlooking a modern city skyline at dusk, ...",
  "edit_type": "location",
  "round1_labels": [...],
  "round1_objects": [...],
  "frame_objects": [
    [
      {"bbox_2d": [448, 26, 765, 998], "label": "woman in brown hat and coat"},
      {"bbox_2d": [521, 795, 618, 968], "label": "clasped hands with ring"},
      {"bbox_2d": [545, 420, 625, 805], "label": "striped shirt under coat"},
      {"bbox_2d": [507, 26, 712, 270], "label": "brown wide-brimmed hat"}
    ],
    [
      {"bbox_2d": [452, 34, 764, 998], "label": "woman in brown hat and coat"},
      {"bbox_2d": [505, 784, 600, 955], "label": "clasped hands with ring"},
      ...
    ],
    ...
  ]
}
```

The schema extends `foreground_grounding_r1.jsonl` with a single new field:

| Field           | Description                                                                                                                                                                                                                                       |
|-----------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `frame_objects` | A 2D list of grounding results, one inner list per 2 FPS-sampled frame. Each inner list mirrors the `round1_objects` schema (a list of `{"bbox_2d": [...], "label": "..."}` items), giving the per-frame bbox of every Round 1 label on that frame. |

The other fields (`id`, `prompt`, `edit_type`, `round1_labels`, `round1_objects`) are inherited unchanged from `foreground_grounding_r1.jsonl`. Use the same [`normalize_bbox`](#normalize-bbox) helper to convert `bbox_2d` values to absolute pixel coordinates.

> **Note.** Some entries in `frame_objects` may have an empty `bbox_2d` (e.g. `{"bbox_2d": [], "label": "..."}`), indicating that the VLM failed to localize that particular label on that frame. Our BAIT algorithm handles these gracefully by relying on the remaining frames' anchors and a pixel-wise majority vote across SAM3 tracking passes.

**`edited_video_score.jsonl`** records per-sample [EditScore](https://github.com/VectorSpaceLab/EditScore) evaluation of the **Stage 5 final synthesized video**. Following the protocol in our paper, we uniformly sample four non-first frames from each video and score them independently. One JSON object per line:

```json
{
  "id": "Sparkle_location_000000",
  "prompt": "Shift the background to a rooftop overlooking a modern city skyline at dusk, ...",
  "frame_indices": [1, 26, 51, 76],
  "editscore": [
    {
      "SC_score": 9.0,
      "PQ_score": 8.5,
      "O_score": 8.719958110896453,
      "SC_score_reasoning": "The editing successfully changed the background to a rooftop overlooking a modern city skyline at dusk, ...",
      "PQ_score_reasoning": "The image has a mostly natural cityscape and lighting, but the person's hands appear slightly distorted ...",
      "SC_raw_output": "...",
      "PQ_raw_output": "..."
    },
    { "SC_score": 8.3, "PQ_score": 8.5, "O_score": 8.388302424289282, "...": "..." },
    { "SC_score": 9.1, "PQ_score": 7.4, "O_score": 8.143194240945185, "...": "..." },
    { "SC_score": 8.9, "PQ_score": 7.8, "O_score": 8.318623075017307, "...": "..." }
  ]
}
```

| Field                          | Description                                                                                                           |
|--------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| `id`                           | Sample id, matches the `task` column in the CSV.                                                                      |
| `prompt`                       | The editing instruction.                                                                                              |
| `frame_indices`                | The 4 frame indices (0-based) sampled from the synthesized video for evaluation, e.g. `[1, 26, 51, 76]`.              |
| `editscore`                    | A length-4 list, one entry per sampled frame, in the same order as `frame_indices`.                                   |
| `editscore[i].SC_score`        | Sub-score (0–10) for instruction-following / consistency on frame `i`.                                                |
| `editscore[i].PQ_score`        | Sub-score (0–10) for perceptual quality on frame `i`.                                                                 |
| `editscore[i].O_score`         | Aggregated overall score on frame `i`.                                                                                |
| `editscore[i].SC_score_reasoning` | Free-text rationale behind `SC_score`.                                                                             |
| `editscore[i].PQ_score_reasoning` | Free-text rationale behind `PQ_score`.                                                                             |
| `editscore[i].SC_raw_output`   | Raw JSON string returned by the EditScore SC head (contains `reasoning` and per-criterion `score` array).             |
| `editscore[i].PQ_raw_output`   | Raw JSON string returned by the EditScore PQ head.                                                                    |

The final filtering rule is: **average `O_score` across all four sampled frames; discard the sample if the mean is below `8`.**

</details>

### 📜 Dataset License

The Sparkle dataset is released under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license.

Source videos in the `openve3m` theme are derived from [OpenVE-3M](https://arxiv.org/abs/2512.07826) and retain their original licenses; please consult the upstream source before redistribution.

## 🎯 Benchmark

**Sparkle-Bench** is the largest evaluation benchmark tailored for instruction-guided video background replacement, comprising **458 carefully curated videos across 4 themes, 21 subthemes, and 97 distinct scenes**. It is fully open-sourced at [🤗stdKonjac/Sparkle-Bench](https://huggingface.co/datasets/stdKonjac/Sparkle-Bench). For evaluation methodology and our six-dimensional scoring protocol, please refer to [our paper](https://arxiv.org/abs/2605.06535).

**All source videos in the benchmark are uncompressed and previewable directly in the browser**, so users can inspect any sample without downloading anything.

The benchmark is organized into **four themes**:

| Theme      | Description                                                                              |
| ---------- |------------------------------------------------------------------------------------------|
| `location` | Background replaced with a different physical environment (rural, nature, landmark, ...).|
| `season`   | Background changed across seasons (spring, summer, autumn, winter).                      |
| `time`     | Background changed across times of day (dawn, dusk, night, ...).                         |
| `style`    | Background restyled (era, mood, cinematic, ...).                                         |

### 🗂️ Repository Structure

```
Sparkle-Bench/
├── README.md
├── location_bench.csv                              # 3 columns: edited_type, prompt, original_video
├── location_metadata.jsonl                         # per-task metadata (edit_type, subtheme, original scene)
├── season_bench.csv
├── season_metadata.jsonl
├── time_bench.csv
├── time_metadata.jsonl
├── style_bench.csv
├── style_metadata.jsonl
├── source_videos/                                  # all 458 source videos, browsable online
│   ├── location/
│   │   ├── Sparkle_location_000011.mp4
│   │   └── ...
│   ├── season/
│   ├── time/
│   └── style/
└── ref_images/                                     # optional reference background images (see below)
    ├── location/
    ├── season/
    ├── time/
    └── style/
```

### 🧾 Benchmark Format

We follow the format of [OpenVE-Bench](https://huggingface.co/datasets/Lewandofski/OpenVE-Bench) for direct compatibility with existing evaluation pipelines.

Each theme's evaluation prompts live in `{edit_type}_bench.csv`, a three-column table:

| Column           | Description                                                                                       |
|------------------|---------------------------------------------------------------------------------------------------|
| `edited_type`    | The theme of this sample, one of `location` / `season` / `time` / `style`.                       |
| `prompt`         | The natural-language editing instruction.                                                         |
| `original_video` | Path to the source video, e.g. `source_videos/location/Sparkle_location_010913.mp4`.              |

Per-task auxiliary metadata is stored alongside in `{edit_type}_metadata.jsonl`. Each line is one sample:

```json
{
  "id": "Sparkle_location_004302",
  "prompt": "Put the subject against ancient stone ruins overgrown with wind-swept grass, ...",
  "metadata": {
    "edit_type": "location",
    "chosen_keyword": "landmark: ancient stone ruins with wind-swept grass",
    "original_scene": "A dimly lit indoor bar or restaurant with brick walls, framed artwork, and warm overhead lighting."
  }
}
```

| Field                      | Description                                                                                                |
|----------------------------|------------------------------------------------------------------------------------------------------------|
| `id`                       | Sample id, e.g. `Sparkle_location_004302`. Matches the basename of the corresponding `original_video` path. |
| `prompt`                   | Same as the `prompt` column in the CSV.                                                                    |
| `metadata.edit_type`       | The theme this sample belongs to (`location` / `season` / `time` / `style`).                               |
| `metadata.chosen_keyword`  | The `subtheme: scene` label (e.g. `"landmark: ancient stone ruins with wind-swept grass"`).               |
| `metadata.original_scene`  | A description of the source video's first frame.                                                           |

### 👀 Online Preview

All 458 source videos are stored as uncompressed `.mp4` files under `source_videos/{edit_type}/`, and can be played directly in the browser without any download.

For example, the source video of task `Sparkle_location_000011` (the first row in the **location** theme of the dataset viewer) is browsable at: [Sparkle_location_000011](https://huggingface.co/datasets/stdKonjac/Sparkle-Bench/blob/main/source_videos/location/Sparkle_location_000011.mp4).

The dataset viewer at the top of the HF page lets you scroll through all four themes and read the corresponding prompts inline.

### ⬇️ Downloading the Benchmark

Sparkle-Bench is small enough to download in one command. Pull the entire repo:

```bash
hf download stdKonjac/Sparkle-Bench --repo-type=dataset --local-dir ./Sparkle-Bench
```

Or download only a single theme (e.g. `location`):

```bash
hf download stdKonjac/Sparkle-Bench \
  --repo-type=dataset \
  --local-dir ./Sparkle-Bench \
  --include "location_*" "source_videos/location/*"
```

After downloading, the relative paths in `{edit_type}_bench.csv` (e.g. `source_videos/location/Sparkle_location_010913.mp4`) will resolve directly.

### 📊 Evaluation

We provide an end-to-end evaluation script, [`eval_sparkle_bench_gemini.py`](https://github.com/showlab/Sparkle/blob/main/eval_sparkle_bench_gemini.py), that scores edited videos using Gemini-2.5-Pro under our six-dimensional rubric (see *Section 3.7* in [our paper](https://arxiv.org/abs/2605.06535)). The six dimensions are: **Instruction Compliance**, **Overall Visual Quality**, **Foreground Integrity**, **Foreground Motion Consistency**, **Background Dynamics**, and **Background Visual Quality**, each scored on a 1–5 scale.

#### 1. Prepare your inference outputs

The script expects edited videos to be organized in a specific directory tree. For every sample in Sparkle-Bench, the inference output should be saved as:

```
{save_dir}/{edit_type}/{subtheme}---{scene_key}/{id}_edited.mp4
```

where:

- `{save_dir}` is your inference root (free to choose).
- `{edit_type}` is one of `location` / `season` / `time` / `style`.
- `{subtheme}---{scene_key}` is derived from the sample's `chosen_keyword` field in `{edit_type}_metadata.jsonl`. Specifically, splitting `chosen_keyword` on `": "` yields `subtheme: scene`, then `scene_key = scene.replace(" ", "_")`. The triple-dash `---` is the separator between the two parts.
- `{id}` is the sample id, e.g. `Sparkle_location_000172`.

For example, the inference outputs across the four themes should look like:

```
{save_dir}/
├── location/
│   └── landmark---ancient_stone_ruins_with_wind-swept_grass/
│       └── Sparkle_location_000172_edited.mp4
├── season/
│   └── {subtheme}---{scene_key}/
│       └── Sparkle_season_xxxxxx_edited.mp4
├── time/
│   └── {subtheme}---{scene_key}/
│       └── Sparkle_time_xxxxxx_edited.mp4
└── style/
    └── {subtheme}---{scene_key}/
        └── Sparkle_style_xxxxxx_edited.mp4
```

#### 2. Configure the Gemini API

By default the script uses **Azure-hosted Gemini via the OpenAI-compatible API** for convenient concurrency. Export two environment variables before running:

```bash
export AZURE_ENDPOINT="https://your-azure-endpoint"
export GEMINI_API_KEY="your-api-key"
```

If you have direct access to the Gemini API, you can swap the `GEMINI_API` client at the top of the script for the native [`google-genai`](https://github.com/googleapis/python-genai) SDK. The request payload only needs `(system prompt, source video, edited video)`, so the adaptation is straightforward. Just keep the `temperature=0` / `seed=42` settings for reproducibility.

#### 3. Run the evaluation

Assuming Sparkle-Bench has been downloaded to `data/Sparkle-Bench/` (the default `--bench_root`):

```bash
python3 eval_sparkle_bench_gemini.py \
    --video_paths /path/to/sparkle_bench_results/
```

For multiple checkpoints in one run:

```bash
python3 eval_sparkle_bench_gemini.py \
    --video_paths /path/to/ckpt_a/sparkle_bench/ \
                  /path/to/ckpt_b/sparkle_bench/ \
                  /path/to/ckpt_c/sparkle_bench/
```

By default the script evaluates all four themes (`location`, `season`, `time`, `style`); pass `--edit_types` to restrict to a subset. Concurrency is controlled inside the script (default 20 workers).

#### 4. Read the output

For each `(save_dir, edit_type)` pair, the script writes:

```
{save_dir}/{edit_type}_gemini-2.5-pro_sparkle_score.jsonl
```

Each line is a per-sample record containing the six-dim scores plus the original Gemini reasoning:

```json
{
  "id": "Sparkle_location_000172",
  "prompt": "Put the subject against ancient stone ruins overgrown with wind-swept grass, ...",
  "edit_type": "location",
  "subtheme": "landmark",
  "scene": "ancient stone ruins with wind-swept grass",
  "scores": [5, 5, 5, 5, 5, 5],
  "result": "Brief reasoning: The edited background perfectly matches every detail of the prompt, ...\nInstruction Compliance: 5\nOverall Visual Quality: 5\nForeground Integrity: 5\nForeground Motion Consistency: 5\nBackground Dynamics: 5\nBackground Visual Quality: 5"
}
```

The `scores` array follows this fixed order: `[Instruction Compliance, Overall Visual Quality, Foreground Integrity, Foreground Motion Consistency, Background Dynamics, Background Visual Quality]`. Following the OpenVE-Bench protocol, the script automatically caps dimensions 2–6 at the Instruction Compliance score to prevent score hacking.

After scoring, the script aggregates per-theme and macro averages and prints a summary table to stdout. The evaluation is **deterministic** by design (`temperature=0`, fixed `seed=42`) for reproducibility.

### 🖼️ Reference Images (Optional, Use with Caution)

By construction, every Sparkle-Bench sample is a video that **passed the first four stages of our pipeline but failed the final synthesis quality check in Stage 5** (see Section 3.7 of [our paper](https://arxiv.org/abs/2605.06535)). As a free byproduct, this means each sample comes with a **pure background image** generated by Stage 3 (Individual Background Generation), where the foreground has been removed from the preliminarily edited first frame.

We release these images under `ref_images/{edit_type}/{id}.png`, alongside the CSV/JSONL annotations. These images may be useful for **reference-based** background-replacement experiments (e.g., feeding the clean background as an extra visual condition to the editing model).

> **⚠️ Disclaimer.** Our paper neither trains any reference-based model nor includes any reference-image-based evaluation. We release `ref_images/` purely to **facilitate future research** in this direction. The images are **not curated** and may contain noise such as low-quality edits or imperfect foreground removal. Please **use them with caution**. We make no quality guarantees about this auxiliary asset.

### 📜 Benchmark License

The Sparkle-Bench is released under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license.

Source videos are derived from [OpenVE-3M](https://arxiv.org/abs/2512.07826) and retain their original licenses; please consult the upstream source before redistribution.

## 🧠 Model

We release **Kiwi-Sparkle**, a video background-replacement model fine-tuned on the **Sparkle** dataset for **10K steps** with a batch size of 128, starting from a [Kiwi-Edit](https://github.com/showlab/Kiwi-Edit) base. Since we apply no architectural modifications to Kiwi-Edit, **Kiwi-Sparkle's weights are fully compatible with the Kiwi-Edit weights structure**. Any inference, training, or deployment pipeline that runs Kiwi-Edit can run Kiwi-Sparkle as a drop-in replacement.

The model is open-sourced at [🤗stdKonjac/Kiwi-Sparkle-720P-81F](https://huggingface.co/stdKonjac/Kiwi-Sparkle-720P-81F) and supports **720P resolution** with up to **81-frame outputs**.

| Setting               | Value                                                                                                                   |
|-----------------------|-------------------------------------------------------------------------------------------------------------------------|
| Foundation model      | [Kiwi-Edit-Stage2 (Image + Video)](https://huggingface.co/linyq/wan2.2_ti2v_5b_qwen25vl_3b_stage2_img_vid_720x1280_81f) |
| Resolution            | 720 × 1280                                                                                                              |
| Max output frames     | 81                                                                                                                      |
| Fine-tuning steps     | 10,000                                                                                                                  |
| Batch size            | 128                                                                                                                     |
| Architectural changes | None. Drop-in compatible with Kiwi-Edit.                                                                                |

### 🚀 Training

Kiwi-Sparkle is trained using the official Kiwi-Edit recipe in [this script](https://github.com/showlab/Kiwi-Edit/blob/main/scripts/run_wan2.2_ti2v_5b_qwen25vl_3b_stage2_img_vid_720x1280_81f.sh) with no modifications. Two common entry points are supported:

**Train from the Kiwi-Edit base on a Sparkle theme.** Point `--vid_dataset_metadata_path` to the corresponding Sparkle training CSV, and load the foundation [Kiwi-Edit-Stage2](https://huggingface.co/linyq/wan2.2_ti2v_5b_qwen25vl_3b_stage2_img_vid_720x1280_81f) checkpoint:

```bash
--vid_dataset_metadata_path /path/to/Sparkle/prompts/{edit_type}_train.csv
--checkpoint /path/to/Kiwi-Edit-Stage2/model.safetensors
```

where `{edit_type}` is one of `location` / `season` / `time` / `style` / `openve3m`. The five training CSVs are hosted [here](https://huggingface.co/datasets/stdKonjac/Sparkle/tree/main/prompts).

**Continue training from our Kiwi-Sparkle checkpoint.** Replace the `--checkpoint` argument:

```bash
--checkpoint /path/to/Kiwi-Sparkle-720P-81F/model.safetensors
```

The rest of the script stays exactly as in the official Kiwi-Edit setup.

### 🎬 Inference

#### OpenVE-Bench

Since Kiwi-Sparkle is architecturally identical to Kiwi-Edit, you can simply follow the official OpenVE-Bench evaluation pipeline of Kiwi-Edit and swap the checkpoint to Kiwi-Sparkle. For example:

```bash
python3 test_benchmark.py \
  --ckpt_path /path/to/Kiwi-Sparkle-720P-81F/model.safetensors \
  --bench openve \
  --max_frame 81 \
  --max_pixels 921600 \
  --save_dir ./infer_results/
```

#### Sparkle-Bench

We provide a dedicated launch pair, [`test_benchmark_sparkle_bench.py`](https://github.com/showlab/Sparkle/blob/main/test_benchmark_sparkle_bench.py) and [`test_benchmark_sparkle_bench.sh`](https://github.com/showlab/Sparkle/blob/main/test_benchmark_sparkle_bench.sh), that mirror Kiwi-Edit's existing benchmarking layout.

**Step 1.** Clone the [Kiwi-Edit repository](https://github.com/showlab/Kiwi-Edit) and copy our two scripts into the Kiwi-Edit repo root, alongside the official `test_benchmark.py`.

**Step 2.** Edit the shell script to point at your Kiwi-Sparkle checkpoint, then launch (defaults to 8 GPUs):

```bash
bash test_benchmark_sparkle_bench.sh
```

The script writes inference outputs to `infer_results/Kiwi-Sparkle-720P-81F/sparkle_bench/{edit_type}/{subtheme}---{scene_key}/{id}_edited.mp4`. Re-run it with a different `EDIT_TYPE` to cover all four themes.

**Step 3.** Score the outputs with our [Gemini-based evaluator](#-evaluation):

```bash
python3 eval_sparkle_bench_gemini.py \
    --video_paths infer_results/Kiwi-Sparkle-720P-81F/sparkle_bench/
```

See the [Evaluation section](#-evaluation) above for details on environment setup, output format, and the six-dimensional scoring rubric.

### 📜 Model License

Kiwi-Sparkle is released under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license.

## 🙏 Acknowledgements

This project is built on top of a number of excellent open-source projects. We thank the authors of [Kiwi-Edit](https://github.com/showlab/Kiwi-Edit), [FLUX.2-klein-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B), [Qwen3-VL-32B](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct), [Wan2.2-I2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B), [LightX2V](https://github.com/ModelTC/lightx2v), and [VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun) for releasing the infrastructure that made this work possible.

## 📝 Citation

If you find Sparkle useful for your research, please consider citing our paper:

```bibtex
@misc{zeng2026sparkle,
  title         = {Sparkle: Realizing Lively Instruction-Guided Video Background Replacement via Decoupled Guidance},
  author        = {Zeng, Ziyun and Lin, Yiqi and Liang, Guoqiang and Shou, Mike Zheng},
  year          = {2026},
  eprint        = {2605.06535},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2605.06535}
}
```