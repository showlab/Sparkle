# Sparkle Data Pipeline

This folder presents the complete five-stage pipeline used to construct the Sparkle dataset, as described in **Figure 2 and Section 3** of [our paper](https://arxiv.org/abs/2605.06535).

## ⚠️ Important Notice

To increase data diversity, we use **random seeds rather than a fixed seed** in the pipeline. As a result, data construction outputs may differ slightly across runs or machines. For example, UniMatch-based camera-movement detection and Qwen's judgments may vary slightly between runs, leading to minor differences in which videos pass filtering. At the prompt-generation stage, a different editing prompt may be generated, leading to different results in subsequent editing stages.

The example run documented here uses the 100 released [location preview source videos](https://huggingface.co/datasets/stdKonjac/Sparkle/tree/main/location/source_video) as its inputs. Its resulting edits are therefore **not identical** to the corresponding 100 [released edited videos](https://huggingface.co/datasets/stdKonjac/Sparkle/tree/main/location/edited_video), even though they use the same source videos.

## Prepare Dummy Source Videos

For convenient debugging, we use the 100 preview source videos from the `location` subset of our [released Hugging Face dataset](https://huggingface.co/datasets/stdKonjac/Sparkle) as a small source video pool. Place these videos under `data_pipeline/location_train_preview100/source_video/`.

You can [preview and download the source videos](https://huggingface.co/datasets/stdKonjac/Sparkle/tree/main/location/source_video) directly in your browser. Alternatively, run the following commands **from the Sparkle repository root** to download only these source videos and organize them into the local source video pool:

```bash
python -m pip install -U huggingface_hub

hf download stdKonjac/Sparkle \
  --repo-type dataset \
  --include "location/source_video/*.mp4" \
  --local-dir data_pipeline/location_train_preview100

# The download preserves the Hugging Face repository's location/source_video/ path.
# Move the videos into the local source video pool.
mkdir -p data_pipeline/location_train_preview100/source_video
mv data_pipeline/location_train_preview100/location/source_video/*.mp4 \
  data_pipeline/location_train_preview100/source_video/
rmdir data_pipeline/location_train_preview100/location/source_video \
  data_pipeline/location_train_preview100/location
```

We also provide two CSV files in `location_train_preview100/` to accompany this source video pool:

| File | Contents | Columns |
| --- | --- | --- |
| [`source_video.csv`](location_train_preview100/source_video.csv) | **Used in this practice.** The source video pool for data construction, containing the 100 source video paths and their sample IDs. | `src_video`, `task` |
| [`location_train_preview100.csv`](location_train_preview100/location_train_preview100.csv) | **Reference only.** The final records of these 100 samples in the released Sparkle dataset, including their released editing prompts and source/target video paths. | `prompt`, `src_video`, `tgt_video`, `task` |

For this walkthrough, we only need `source_video.csv` and the corresponding source videos in `source_video/`; `location_train_preview100.csv` is provided for reference only. Video paths in both CSV files are relative to `data_pipeline/location_train_preview100/`. For example, `src_video` is `source_video/Sparkle_location_000000.mp4`, and `task` is `Sparkle_location_000000`.

The released edited videos referenced by the `tgt_video` column in `location_train_preview100.csv` can be [previewed on Hugging Face](https://huggingface.co/datasets/stdKonjac/Sparkle/tree/main/location/edited_video). They are excluded from the download above because they are not needed for this walkthrough.

After downloading and organizing the source videos, the complete directory layout, including the two provided CSV files, should be:

```text
data_pipeline/
└── location_train_preview100/
    ├── location_train_preview100.csv
    ├── source_video.csv
    └── source_video/
        ├── Sparkle_location_000000.mp4
        ├── Sparkle_location_000003.mp4
        └── ... (100 source videos in total)
```

## Stage 1: Fixed-Camera Source Video Filtering

### Coarse Camera Movement Filtering

We first perform coarse fixed-camera filtering using [UniMatch](https://github.com/autonomousvision/unimatch) optical flow and RANSAC homography estimation. For details of the filtering algorithm, see **Appendix A** of [our paper](https://arxiv.org/abs/2605.06535).

Starting from the **Sparkle repository root**, enter the UniMatch directory:

```bash
cd data_pipeline/stage1/unimatch
```

In this directory, [`scripts/gmflow_sparkle.sh`](stage1/unimatch/scripts/gmflow_sparkle.sh) launches optical-flow inference across multiple GPUs. It defaults to **8 GPUs and 8 shards**; adjust `NUM_GPUs`, `INDEX_BEGIN`, and `NUM_SHARDS` at the top of the script to match your setup. For UniMatch installation and pretrained weights, refer to its [README](https://github.com/autonomousvision/unimatch/blob/master/README.md); the checkpoint path is specified by `--resume` in the launch script.

Then run:

```bash
bash scripts/gmflow_sparkle.sh
```

Each GPU processes a shard of `source_video.csv` and saves its optical-flow motion statistics and camera-movement flags to a JSONL file under `location_train_preview100/stage1_unimatch_motion/` (e.g., `shards_8_0.jsonl`). After inference finishes, enter the preview directory and gather all shards:

```bash
cd ../../location_train_preview100
cat stage1_unimatch_motion/*.jsonl > stage1_unimatch_motion_gathered.jsonl
```

Next, return to `stage1/` and export the fixed-camera candidates:

```bash
cd ../stage1
python export_fixed_camera_video_unimatch.py
```

This creates `data_pipeline/location_train_preview100/stage1_fixed_camera_unimatch.csv`, containing the `src_video` and `task` fields for videos retained by the UniMatch-based coarse filter. A video is retained only when no camera movement is detected across all sampled frame pairs.

### Fine-grained Camera Movement Filtering

Next, we use [Qwen3-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct) to detect residual camera movement in the candidates retained by UniMatch. Continue in `data_pipeline/stage1/`, where the preceding export command finishes. If starting from the **Sparkle repository root**, enter it with:

```bash
cd data_pipeline/stage1
```

In this directory, [`infer_camera_qwen3vl.sh`](stage1/infer_camera_qwen3vl.sh) launches Qwen3-VL inference using **vLLM** as the backend. Each machine processes one shard using tensor parallelism across its GPUs. The underlying [`infer_camera_qwen3vl.py`](stage1/infer_camera_qwen3vl.py) defaults to `tensor_parallel_size=8`; adjust this value to match the number of GPUs used on each machine.

Set the shard parameters at the top of `infer_camera_qwen3vl.sh` before running:

- **Single machine:** set `NUM_SHARDS=1` and `SHARD_INDEX=0`.
- **Multiple machines:** for `N` machines, set `NUM_SHARDS=N` on every machine and assign each a distinct `SHARD_INDEX` from `0` to `N-1`.

Then run the following on each machine:

```bash
bash infer_camera_qwen3vl.sh
```

The script reads `location_train_preview100/stage1_fixed_camera_unimatch.csv` and saves each machine's results under `location_train_preview100/stage1_qwen3vl_camera/`, with filenames such as `shards_1_0.jsonl` for a single machine or `shards_N_i.jsonl` for shard `i` of `N`.

Once inference has finished on all machines and all shard files are collected in the same `stage1_qwen3vl_camera/` directory, enter `location_train_preview100/` and gather the results:

```bash
cd ../location_train_preview100
cat stage1_qwen3vl_camera/*.jsonl > stage1_qwen3vl_camera_gathered.jsonl
```

Return to `stage1/` and export the final Stage 1 source video pool:

```bash
cd ../stage1
python export_fixed_camera_video_qwen3vl.py
```

This creates `data_pipeline/location_train_preview100/stage1_fixed_camera_unimatch_qwen3vl.csv`, containing the `src_video` and `task` fields for videos that pass **both UniMatch coarse filtering and Qwen3-VL fine-grained filtering**. These form the strictly filtered fixed-camera source video pool for the subsequent stages.

## Stage 2: Preliminary Background Replacement

### Edit Prompt Generation

Starting from the **Sparkle repository root**, enter the Stage 2 directory:

```bash
cd data_pipeline/stage2
```

In this directory, [`gen_background_change_prompt.sh`](stage2/gen_background_change_prompt.sh) uses [Qwen3-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct) with **vLLM** to generate background editing prompts for a specified `--edit_type`. It reads the fixed-camera source video pool from `location_train_preview100/stage1_fixed_camera_unimatch_qwen3vl.csv` and generates an instruction based on each source video's first frame. For this walkthrough, we set **`--edit_type location`** in the launch script.

The script also supports sharded inference across multiple machines. Set `NUM_SHARDS` and `SHARD_INDEX` at the top of `gen_background_change_prompt.sh`:

- **Single machine:** set `NUM_SHARDS=1` and `SHARD_INDEX=0`.
- **Multiple machines:** for `N` machines, set `NUM_SHARDS=N` on every machine and assign each a distinct `SHARD_INDEX` from `0` to `N-1`.

As in Stage 1, each machine handles one shard. The underlying [`gen_background_change_prompt.py`](stage2/gen_background_change_prompt.py) defaults to `tensor_parallel_size=8`; adjust this value to match the number of GPUs used on each machine. Then run:

```bash
bash gen_background_change_prompt.sh
```

The generated prompts are saved under `location_train_preview100/stage2_generated_prompts/` as `location_{NUM_SHARDS}_{SHARD_INDEX}.jsonl` (e.g., `location_1_0.jsonl` for a single machine). Once inference has finished and all shard files are collected in this directory, enter `location_train_preview100/` and gather the `location` results:

```bash
cd ../location_train_preview100
cat stage2_generated_prompts/location_*.jsonl > stage2_generated_prompts_location_gathered.jsonl
```

Finally, return to `stage2/` and convert the gathered JSONL records to a standard CSV:

```bash
cd ../stage2
python export_prompt_jsonl2csv.py
```

This creates `data_pipeline/location_train_preview100/stage2_generated_prompts_location.csv` with three columns: **`prompt`**, **`src_video`**, and **`task`**. **The prompts are newly generated for this walkthrough rather than copied from the released annotations.** As a result, these newly generated prompts do not exactly match the released prompts, and these differences may affect subsequent editing and filtering results.

### First Frame Background Replacement

Continue in `data_pipeline/stage2/`. If starting from the **Sparkle repository root**, enter it with:

```bash
cd data_pipeline/stage2
```

First, extract the first frame of each source video listed in the generated prompt CSV:

```bash
python export_frame0.py
```

[`export_frame0.py`](stage2/export_frame0.py) reads `location_train_preview100/stage2_generated_prompts_{edit_type}.csv` and saves the extracted frames as `{task}.png` under `location_train_preview100/stage2_source_frame0_{edit_type}/`. Set `edit_type` in the script to match the desired edit type. In this walkthrough, we use `location`, so the output directory is `location_train_preview100/stage2_source_frame0_location/`.

Next, [`replace_background_sparkle.sh`](stage2/replace_background_sparkle.sh) uses **vLLM-Omni** as the inference backend to run [FLUX.2-klein-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B). It loads the extracted first frames and replaces their backgrounds according to the editing prompts generated in the preceding subsection. The launch script defaults to **8 GPUs** (`NUM_GPUs=8`) and `--edit-type location`; adjust the GPU count at the top of the script to match your setup, then run:

```bash
bash replace_background_sparkle.sh
```

The edited images are saved as `{task}.png` under `location_train_preview100/stage2_edited_frame0_{edit_type}/`, which is `location_train_preview100/stage2_edited_frame0_location/` in this walkthrough.

### EditScore Filtering

We use [EditScore](https://github.com/VectorSpaceLab/EditScore) to evaluate the first-frame background replacements and retain high-quality editing pairs. Starting from the **Sparkle repository root**, enter the EditScore directory:

```bash
cd data_pipeline/stage2/EditScore
```

In this directory, [`data/process_editscore_sparkle_stage2.sh`](stage2/EditScore/data/process_editscore_sparkle_stage2.sh) launches `NUM_GPUs` subprocesses per machine, each using one GPU with **vLLM** as the inference backend. It scores the source and edited first frames against their generated editing prompts. The defaults are `NUM_GPUs=8`, `INDEX_BEGIN=0`, and `NUM_SHARDS=8`; adjust these values at the top of the script to match your setup. For multiple machines, use the same global `NUM_SHARDS` and assign non-overlapping shard ranges through `INDEX_BEGIN`. In this walkthrough, keep **`--edit_type location`**, then run:

```bash
bash data/process_editscore_sparkle_stage2.sh
```

Each subprocess writes its results to `location_train_preview100/stage2_editscore/{edit_type}_{NUM_SHARDS}_{SHARD_INDEX}.jsonl`, such as `location_8_0.jsonl` for the first shard with the default configuration.

After inference finishes on all machines and all shard files are collected in the same `stage2_editscore/` directory, enter `location_train_preview100/` and gather the results. Here, we use `edit_type=location`:

```bash
cd ../../location_train_preview100
cat stage2_editscore/location_*.jsonl > stage2_editscore_location_gathered.jsonl
```

Finally, return to `stage2/` and filter the first-frame editing pairs:

```bash
cd ../stage2
python filter_stage2_edited_frame0.py
```

[`filter_stage2_edited_frame0.py`](stage2/filter_stage2_edited_frame0.py) retains samples with **`editscore.overall >= 8`** by default and exports the corresponding subset of the generated prompt CSV to `location_train_preview100/stage2_generated_prompts_{edit_type}_filtered.csv`. The output preserves the **`prompt`**, **`src_video`**, and **`task`** columns. Set `edit_type` and `threshold` in the script as needed; for this walkthrough, the output is `location_train_preview100/stage2_generated_prompts_location_filtered.csv`.

## Stage 3: Individual Background Generation

### Edit-driven First-frame Foreground Grounding

Starting from the **Sparkle repository root**, enter the Stage 3 directory:

```bash
cd data_pipeline/stage3
```

In this directory, [`infer_foreground_sparkle_stage3.sh`](stage3/infer_foreground_sparkle_stage3.sh) uses [Qwen3-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct) with **vLLM** to process the qualified samples in `location_train_preview100/stage2_generated_prompts_{edit_type}_filtered.csv`. For each sample, it compares the source and edited first frames together with the editing instruction to identify foreground entities that remain unchanged and locate their bounding boxes in the source frame. The paired images are read from `stage2_source_frame0_{edit_type}/` and `stage2_edited_frame0_{edit_type}/` under `location_train_preview100/`.

The script defaults to **8 GPUs per machine** through `tensor_parallel_size=8` in [`infer_foreground_sparkle_stage3.py`](stage3/infer_foreground_sparkle_stage3.py); adjust this value to match your setup. Each machine handles one shard. Set `NUM_SHARDS=1` and `SHARD_INDEX=0` for a single machine, or set `NUM_SHARDS=N` and assign distinct `SHARD_INDEX` values from `0` to `N-1` across `N` machines. For this walkthrough, keep **`--edit_type location`** in the launch script, then run:

```bash
bash infer_foreground_sparkle_stage3.sh
```

The grounding results are saved to `location_train_preview100/stage3_foreground_grounding/{edit_type}_{NUM_SHARDS}_{SHARD_INDEX}.jsonl`, such as `location_1_0.jsonl` for a single machine. Each record includes the foreground labels (`round1_labels`) and objects with bounding boxes (`round1_objects`).

Once inference finishes on all machines and all shard files are collected in the same `stage3_foreground_grounding/` directory, enter `location_train_preview100/` and gather the results. Here, we use `edit_type=location`:

```bash
cd ../location_train_preview100
cat stage3_foreground_grounding/location_*.jsonl > stage3_foreground_grounding_location_gathered.jsonl
```

### Label-driven Foreground Removal

From `location_train_preview100/`, where the preceding gather command finishes, return to the Stage 3 directory:

```bash
cd ../stage3
```

In this directory, [`remove_foreground_sparkle.sh`](stage3/remove_foreground_sparkle.sh) uses **vLLM-Omni** as the inference backend to run [FLUX.2-klein-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B). It reads the foreground labels (`round1_labels`) from `location_train_preview100/stage3_foreground_grounding_{edit_type}_gathered.jsonl` and constructs removal instructions such as `Remove the bald man`. These instructions remove the identified foreground entities from the background-replaced first frames in `location_train_preview100/stage2_edited_frame0_{edit_type}/`.

The launch script defaults to **8 GPUs** (`NUM_GPUs=8`). Adjust the GPU count to match your setup and keep **`--edit-type location`** for this walkthrough, then run:

```bash
bash remove_foreground_sparkle.sh
```

The resulting background images are saved as `{task}.png` under `location_train_preview100/stage3_edited_frame0_{edit_type}/`, which is `location_train_preview100/stage3_edited_frame0_location/` in this walkthrough.

### EditScore Filtering

We again use EditScore to evaluate the editing results, this time measuring the quality of foreground removal. Starting from the **Sparkle repository root**, enter the Stage 3 EditScore directory:

```bash
cd data_pipeline/stage3/EditScore
```

In this directory, [`data/process_editscore_sparkle_stage3.sh`](stage3/EditScore/data/process_editscore_sparkle_stage3.sh) launches **8 subprocesses on 8 GPUs** by default, with each subprocess running EditScore on one GPU using **vLLM**. As in Stage 2, configure `NUM_GPUs`, `INDEX_BEGIN`, and `NUM_SHARDS` at the top of the script to match your setup; the defaults are `8`, `0`, and `8`, respectively. For multiple machines, use the same global `NUM_SHARDS` and assign non-overlapping shard ranges through `INDEX_BEGIN`.

The script reads the qualified samples from `location_train_preview100/stage2_generated_prompts_{edit_type}_filtered.csv` and compares their foreground-intact frames in `location_train_preview100/stage2_edited_frame0_{edit_type}/` against the foreground-removed frames in `location_train_preview100/stage3_edited_frame0_{edit_type}/`. Each pair is scored against a removal instruction built from the foreground labels in `stage3_foreground_grounding_{edit_type}_gathered.jsonl`. For this walkthrough, keep **`--edit_type location`**, then run:

```bash
bash data/process_editscore_sparkle_stage3.sh
```

Each shard's results are saved to `location_train_preview100/stage3_editscore/{edit_type}_{NUM_SHARDS}_{SHARD_INDEX}.jsonl`. After inference finishes on all machines and all shard files are collected in the same `stage3_editscore/` directory, enter `location_train_preview100/` and gather the results. Here, we use `edit_type=location`:

```bash
cd ../../location_train_preview100
cat stage3_editscore/location_*.jsonl > stage3_editscore_location_gathered.jsonl
```

Finally, return to `stage3/` and filter the foreground-removal results:

```bash
cd ../stage3
python filter_stage3_edited_frame0.py
```

[`filter_stage3_edited_frame0.py`](stage3/filter_stage3_edited_frame0.py) retains samples with **`editscore.overall >= 8.5`** by default and exports the corresponding subset to `location_train_preview100/stage3_generated_prompts_{edit_type}_filtered.csv`. The output preserves the original background editing **`prompt`**, **`src_video`**, and **`task`** columns from the Stage 2 filtered CSV. Set `edit_type` and `threshold` in the script as needed; for this walkthrough, the output is `location_train_preview100/stage3_generated_prompts_location_filtered.csv`.

### Synthesize Background Video

As described in our paper, we use [LightX2V's four-step distilled model](https://huggingface.co/lightx2v/Wan2.2-Distill-Models) to animate the clean background images into foreground-free background videos.

To run LightX2V's four-step accelerated generation, first install **SageAttention2** following the [official installation instructions](https://github.com/thu-ml/sageattention).

Next, prepare the Wan base model and LightX2V's low- and high-noise four-step distilled weights. Starting from the **Sparkle repository root**, enter the Stage 3 directory:

```bash
cd data_pipeline/stage3
```

Download [Wan2.2-I2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B) to the default Hugging Face cache:

```bash
hf download Wan-AI/Wan2.2-I2V-A14B
```

The inference script loads this model from the Hugging Face cache with `local_files_only=True`. Then download the two [LightX2V distilled checkpoints](https://huggingface.co/lightx2v/Wan2.2-Distill-Models) into the `LightX2V/` root directory, still running from `stage3/`:

```bash
hf download lightx2v/Wan2.2-Distill-Models \
  --local-dir ./LightX2V \
  --include "wan2.2_i2v_A14b_low_noise_lightx2v_4step.safetensors" \
            "wan2.2_i2v_A14b_high_noise_lightx2v_4step_1030.safetensors"
```

With these preparations complete, enter the LightX2V directory:

```bash
cd LightX2V
```

In this directory, [`examples/wan/wan_i2v_distilled_sparkle.sh`](stage3/LightX2V/examples/wan/wan_i2v_distilled_sparkle.sh) launches background video generation using Wan2.2-I2V-A14B with the low- and high-noise four-step distilled weights. It defaults to **8 GPUs** and **4 inference steps**; keep **`--edit_type=location`** for this walkthrough.

The script reads the samples that passed Stage 3 EditScore filtering from `location_train_preview100/stage3_generated_prompts_{edit_type}_filtered.csv`. It animates their foreground-removed first frames in `location_train_preview100/stage3_edited_frame0_{edit_type}/` into background videos, using the generated prompts as text guidance. Run:

```bash
bash examples/wan/wan_i2v_distilled_sparkle.sh
```

The generated background videos are saved as `{task}.mp4` under `location_train_preview100/stage3_edited_background_video_{edit_type}/`, which is `location_train_preview100/stage3_edited_background_video_location/` in this walkthrough.

## Stage 4: BAIT Foreground Tracking

### Label-driven 2FPS Foreground Bounding Boxes

To implement the Bbox-Anchor-In-Temporal (BAIT) algorithm described in [our paper](https://arxiv.org/abs/2605.06535), we first locate the foreground entities identified in Stage 3 across different source-video frames. These bounding boxes serve as temporal anchors for the subsequent tracking step.

Starting from the **Sparkle repository root**, enter the Stage 4 directory:

```bash
cd data_pipeline/stage4
```

In this directory, [`infer_foreground_sparkle_stage4.sh`](stage4/infer_foreground_sparkle_stage4.sh) uses [Qwen3-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct) with **vLLM** as the inference backend. It reads the foreground labels from `location_train_preview100/stage3_foreground_grounding_{edit_type}_gathered.jsonl` and processes the samples retained in `stage3_generated_prompts_{edit_type}_filtered.csv`. Each source video is sampled at **2 FPS**, and the model grounds the labeled foreground entities in each sampled frame.

The script defaults to **8 GPUs** through `tensor_parallel_size=8` in [`infer_foreground_sparkle_stage4.py`](stage4/infer_foreground_sparkle_stage4.py); adjust this value to match your setup. Each machine handles one shard. Set `NUM_SHARDS=1` and `SHARD_INDEX=0` for a single machine, or set `NUM_SHARDS=N` and assign distinct `SHARD_INDEX` values from `0` to `N-1` across `N` machines. Keep **`--edit_type location`** for this walkthrough, then run:

```bash
bash infer_foreground_sparkle_stage4.sh
```

The results are saved to `location_train_preview100/stage4_foreground_grounding/{edit_type}_{num_shards}_{shard_index}.jsonl`, such as `location_1_0.jsonl` for a single machine. Each record includes the Stage 3 foreground labels and a `frame_objects` list containing the grounding results for each sampled frame.

After inference finishes on all machines and all shard files are collected in the same `stage4_foreground_grounding/` directory, enter `location_train_preview100/` and gather the results. Here, we use `edit_type=location`:

```bash
cd ../location_train_preview100
cat stage4_foreground_grounding/location_*.jsonl > stage4_foreground_grounding_location_gathered.jsonl
```

### BAIT Inference

From `location_train_preview100/`, where the preceding gather command finishes, return to the Stage 4 directory:

```bash
cd ../stage4
```

In this directory, [`get_video_mask_sparkle.sh`](stage4/get_video_mask_sparkle.sh) launches **8 subprocesses by default**, each running [SAM3](https://huggingface.co/facebook/sam3)-based BAIT inference on one GPU. Configure `NUM_GPUs`, `INDEX_BEGIN`, and `NUM_SHARDS` at the top of the script to match your setup; the defaults are `8`, `0`, and `8`, respectively. Keep **`--edit_type location`** for this walkthrough, then run:

```bash
bash get_video_mask_sparkle.sh
```

The script reads the temporal anchors from `location_train_preview100/stage4_foreground_grounding_{edit_type}_gathered.jsonl` and matches them to the source videos in `stage3_generated_prompts_{edit_type}_filtered.csv`. For each video, SAM3 performs forward and backward tracking from the temporal anchors, and the resulting masks are aggregated through **pixel-wise voting** into a final foreground mask.

Each video's final mask is saved as `{task}.npz` under `location_train_preview100/stage4_source_video_mask_{edit_type}/`, which is `location_train_preview100/stage4_source_video_mask_location/` in this walkthrough. The compressed NPZ contains `mask` (the bit-packed foreground mask) and `shape` (its original `(T, H, W)` dimensions).

## Stage 5: Edited Video Generation with Decoupled Fore/Background Guidance

### Mix Decoupled Foreground and Background Guidance

We use **Canny edge videos** as video guidance in Sparkle, combining the source video's foreground motion with the independently generated background dynamics.

Starting from the **Sparkle repository root**, enter the Stage 5 directory:

```bash
cd data_pipeline/stage5
```

In this directory, [`gen_canny_video_sparkle.sh`](stage5/gen_canny_video_sparkle.sh) launches **8 subprocesses by default**, each processing a shard on one GPU. Configure `NUM_GPUs`, `INDEX_BEGIN`, and `NUM_SHARDS` at the top of the script to match your setup; the defaults are `8`, `0`, and `8`, respectively. Keep **`--edit_type location`** for this walkthrough.

For the samples retained in `location_train_preview100/stage3_generated_prompts_{edit_type}_filtered.csv`, the script reads the source videos, the generated background videos in `location_train_preview100/stage3_edited_background_video_{edit_type}/`, and the BAIT foreground masks in `location_train_preview100/stage4_source_video_mask_{edit_type}/`.

It extracts edge maps from the source and generated background videos using `LineartDetector`, then combines them according to the BAIT mask: **source-video edges inside the foreground mask, and generated-background edges outside it**. The resulting `edited_video_canny` provides combined foreground and background guidance for the final background-replaced video. Run:

```bash
bash gen_canny_video_sparkle.sh
```

The mixed guidance videos are saved as `{task}.mp4` under `location_train_preview100/stage5_edited_video_canny_{edit_type}/`, which is `location_train_preview100/stage5_edited_video_canny_location/` in this walkthrough.

### Synthesize Final Video

We use [Wan2.2-Fun-A14B-Control](https://huggingface.co/alibaba-pai/Wan2.2-Fun-A14B-Control) to synthesize the final background-replaced video, conditioned on the edited first frame and the mixed foreground/background edge guidance.

From `stage5/`, enter the VideoX-Fun directory:

```bash
cd VideoX-Fun
```

First, download the model to the default Hugging Face cache, where the inference script loads it:

```bash
hf download alibaba-pai/Wan2.2-Fun-A14B-Control
```

In this directory, [`examples/wan2.2_fun/predict_v2v_control_batch_sparkle.sh`](stage5/VideoX-Fun/examples/wan2.2_fun/predict_v2v_control_batch_sparkle.sh) launches **8-GPU FSDP inference** with **10 inference steps** by default. For the samples retained after Stage 3 filtering, it uses two conditions:

- **Edited first frame:** `location_train_preview100/stage2_edited_frame0_{edit_type}/{task}.png`, produced by Stage 2 with the foreground preserved.
- **Mixed Canny guidance:** `location_train_preview100/stage5_edited_video_canny_{edit_type}/{task}.mp4`, produced in the preceding subsection.

Keep **`--edit_type=location`** for this walkthrough, then run:

```bash
bash examples/wan2.2_fun/predict_v2v_control_batch_sparkle.sh
```

The final edited videos are saved as `{task}.mp4` under `location_train_preview100/stage5_edited_video_{edit_type}/`, which is `location_train_preview100/stage5_edited_video_location/` in this walkthrough.

> **Runtime:** This step is relatively time-consuming. On our machine with **8 H100 GPUs**, generating one **720P edited video** takes approximately **60–70 seconds** with our **10-step** configuration.

### EditScore Final Filtering

Finally, we evaluate the synthesized videos with EditScore and filter them by their average frame-level editing quality. Starting from the **Sparkle repository root**, enter the Stage 5 EditScore directory:

```bash
cd data_pipeline/stage5/EditScore
```

In this directory, [`data/process_editscore_sparkle_stage5.sh`](stage5/EditScore/data/process_editscore_sparkle_stage5.sh) launches **8 subprocesses on 8 GPUs** by default, each running EditScore on one GPU with **vLLM**. Configure `NUM_GPUs`, `INDEX_BEGIN`, and `NUM_SHARDS` at the top of the script to match your setup; the defaults are `8`, `0`, and `8`, respectively. Keep **`--edit_type location`** for this walkthrough.

For each sample in `location_train_preview100/stage3_generated_prompts_{edit_type}_filtered.csv`, the script compares the source video with the final edited video in `location_train_preview100/stage5_edited_video_{edit_type}/`. It uniformly samples **four corresponding frame pairs, excluding the first frame**, and scores each pair against the editing instruction. The video's overall score is the mean of the four frame-level `O_score` values, computed during the final filtering step. Run:

```bash
bash data/process_editscore_sparkle_stage5.sh
```

Each shard writes its frame-level results to `location_train_preview100/stage5_editscore/{edit_type}_{NUM_SHARDS}_{SHARD_INDEX}.jsonl`. Once inference finishes on all machines and all shard files are collected in the same `stage5_editscore/` directory, enter `location_train_preview100/` and gather the results. Here, we use `edit_type=location`:

```bash
cd ../../location_train_preview100
cat stage5_editscore/location_*.jsonl > stage5_editscore_location_gathered.jsonl
```

Return to `stage5/` and apply the final quality filter:

```bash
cd ../stage5
python filter_stage5_edited_video.py
```

[`filter_stage5_edited_video.py`](stage5/filter_stage5_edited_video.py) averages the four frame-level `O_score` values and retains videos with a **mean score of at least 8**, discarding those below 8 as described in our paper. The retained samples are exported to `location_train_preview100/stage5_generated_prompts_{edit_type}.csv`, preserving the **`prompt`**, **`src_video`**, and **`task`** columns from the Stage 3 filtered CSV. Set `edit_type` and `threshold` in the script as needed; for this walkthrough, the final CSV is `location_train_preview100/stage5_generated_prompts_location.csv`.

### Pack Dataset to Sparkle Format

We package the final retained video pairs and editing prompts into the same annotation format as the [Sparkle release](https://huggingface.co/datasets/stdKonjac/Sparkle).

Continue in `data_pipeline/stage5/`, where the preceding filtering command finishes. If starting from the **Sparkle repository root**, enter it with:

```bash
cd data_pipeline/stage5
```

In this directory, [`pack_dataset.py`](stage5/pack_dataset.py) reads `location_train_preview100/stage5_generated_prompts_{edit_type}.csv` and collects the corresponding source videos and final Stage 5 edited videos. Set `edit_type` in the script as needed; for this walkthrough, we use `location`. Run:

```bash
python pack_dataset.py
```

The script writes the packed dataset under `location_train_preview100/`:

- **`source_video_sparkle/`** contains copies of the source videos for the final retained samples.
- **`edited_video_sparkle/`** contains copies of their corresponding final edited videos.
- **`dataset_sparkle.csv`** contains the final annotations with four columns: **`prompt`**, **`src_video`**, **`tgt_video`**, and **`task`**, matching the released Sparkle format.

Video paths in `dataset_sparkle.csv` are relative to `location_train_preview100/`: `src_video` points to `source_video_sparkle/{task}.mp4`, and `tgt_video` points to `edited_video_sparkle/{task}.mp4`.

After completing all five stages and packing the dataset, the final working directory looks approximately as follows. We use **`Sparkle_location_000003`**, the first sample in this walkthrough's `dataset_sparkle.csv`, to illustrate the per-sample artifacts. For sharded metadata and EditScore outputs, one representative JSONL file is shown in each directory; `...` indicates additional files.

```text
data_pipeline/location_train_preview100/
├── source_video.csv
├── location_train_preview100.csv
├── source_video/
│   ├── Sparkle_location_000003.mp4
│   └── ...
├── stage1_unimatch_motion/
│   ├── shards_8_0.jsonl
│   └── ...
├── stage1_unimatch_motion_gathered.jsonl
├── stage1_fixed_camera_unimatch.csv
├── stage1_qwen3vl_camera/
│   └── shards_1_0.jsonl
├── stage1_qwen3vl_camera_gathered.jsonl
├── stage1_fixed_camera_unimatch_qwen3vl.csv
├── stage2_generated_prompts/
│   └── location_1_0.jsonl
├── stage2_generated_prompts_location_gathered.jsonl
├── stage2_generated_prompts_location.csv
├── stage2_source_frame0_location/
│   ├── Sparkle_location_000003.png
│   └── ...
├── stage2_edited_frame0_location/
│   ├── Sparkle_location_000003.png
│   └── ...
├── stage2_editscore/
│   ├── location_8_0.jsonl
│   └── ...
├── stage2_editscore_location_gathered.jsonl
├── stage2_generated_prompts_location_filtered.csv
├── stage3_foreground_grounding/
│   └── location_1_0.jsonl
├── stage3_foreground_grounding_location_gathered.jsonl
├── stage3_edited_frame0_location/
│   ├── Sparkle_location_000003.png
│   └── ...
├── stage3_editscore/
│   ├── location_8_0.jsonl
│   └── ...
├── stage3_editscore_location_gathered.jsonl
├── stage3_generated_prompts_location_filtered.csv
├── stage3_edited_background_video_location/
│   ├── Sparkle_location_000003.mp4
│   └── ...
├── stage4_foreground_grounding/
│   └── location_1_0.jsonl
├── stage4_foreground_grounding_location_gathered.jsonl
├── stage4_source_video_mask_location/
│   ├── Sparkle_location_000003.npz
│   └── ...
├── stage5_edited_video_canny_location/
│   ├── Sparkle_location_000003.mp4
│   └── ...
├── stage5_edited_video_location/
│   ├── Sparkle_location_000003.mp4
│   └── ...
├── stage5_editscore/
│   ├── location_8_0.jsonl
│   └── ...
├── stage5_editscore_location_gathered.jsonl
├── stage5_generated_prompts_location.csv
├── source_video_sparkle/
│   ├── Sparkle_location_000003.mp4
│   └── ...
├── edited_video_sparkle/
│   ├── Sparkle_location_000003.mp4
│   └── ...
└── dataset_sparkle.csv
```
