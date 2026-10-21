import time
import base64

from torch.nn.parallel import data_parallel
from vllm import LLM, SamplingParams
import os
from io import BytesIO
import json
from tqdm import tqdm
from PIL import Image
import decord
import argparse
from datasets import load_dataset, Features, Value, Sequence

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_shards", type=int, default=None)
    parser.add_argument("--shard_index", type=int, default=None)
    args = parser.parse_args()

    dset_path = "../location_train_preview100/stage1_fixed_camera_unimatch.csv"

    print("Loading dataset with huggingface datasets...")

    dataset = load_dataset("csv", data_files=dset_path, split="train")

    assert (args.num_shards is None and args.shard_index is None) or (
        args.num_shards is not None and args.shard_index is not None
    )

    if args.num_shards is not None:
        dataset = dataset.shard(num_shards=args.num_shards, index=args.shard_index, contiguous=True)
        print(f"Current eval shard: {args.shard_index} / {args.num_shards}, it has {len(dataset)} samples.")

    # 1. 初始化模型引擎
    model_name = "Qwen/Qwen3-VL-32B-Instruct"
    print(f"Loading {model_name}...")
    llm = LLM(
        model=model_name,
        tensor_parallel_size=8,  # 对应 --tensor-parallel-size 8
        mm_encoder_tp_mode="data",  # 对应 --mm-encoder-tp-mode data
        enable_expert_parallel=False,  # 对应 --enable-expert-parallel
        async_scheduling=True,  # 对应 --async-scheduling
        # limit_mm_per_prompt={"image": 0, "video": 0},  # disable images and videos
        limit_mm_per_prompt={"image": 0},  # video-only mode, disable image
        # video args
        media_io_kwargs={"video": {"num_frames": -1}},
        disable_log_stats=True,  # 👈 关闭周期性的状态统计打印
    )

    save_path = f"../location_train_preview100/stage1_qwen3vl_camera/shards_{args.num_shards}_{args.shard_index}.jsonl"

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w") as fout:
        for idx, item in tqdm(enumerate(dataset), total=len(dataset)):
            try:
                video_path = os.path.join("../location_train_preview100", item["src_video"])

                with open(video_path, "rb") as f:
                    base64_encoded = base64.b64encode(f.read()).decode("utf-8")

                base64_video_url = f"data:video/mp4;base64,{base64_encoded}"

            except Exception as e:
                print(f"Video loading error on item: {item}, error: {e}")
                continue

            prompt = """
            Analyze the camera movement in the provided video clip. Determine whether the camera shot is strictly "Static" or "Dynamic".
            Definition of a Static Shot:
            The camera is completely fixed and stationary (e.g., mounted on a tripod). There is absolutely NO panning, tilting, zooming, tracking, dollying, or rotation. 
            CRITICAL: Subjects or objects within the frame may move, but the camera itself and the boundaries of the frame must remain perfectly still.

            Definition of a Dynamic Shot:
            The camera exhibits any form of movement, including panning, tilting, zooming, handheld shaking, or physical displacement.

            Output your response strictly in the following JSON format without any markdown blocks or additional text:
            {
                "reason": "Briefly explain what camera movement was detected, or confirm that the frame boundaries remained fixed.",
                "is_static": true/false,
                "confidence_score": 1-10
            }
            """

            # 2. 准备与 OpenAI 格式完全兼容的 messages 列表
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video_url", "video_url": {"url": base64_video_url}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            # 3. 设置生成参数
            sampling_params = SamplingParams(max_tokens=8192, temperature=0.0)

            # start = time.time()
            # print("Start inference...")
            try:
                # 4. 执行推理：较新版本的 vLLM 支持直接传入 messages 列表，它会自动处理 Chat Template
                outputs = llm.chat(
                    messages=messages,
                    sampling_params=sampling_params,
                    chat_template_content_format="openai",  # 👈 明确告诉 vLLM 解析 OpenAI 的列表格式
                    # video args
                    mm_processor_kwargs={"fps": 2, "do_sample_frames": True},
                    use_tqdm=False,  # 👈 关掉 vLLM 当前 batch 的进度条
                )

                resp = outputs[0].outputs[0].text
                resp = resp.strip()
            except:
                print(f"Inference error happens on item: {item}")
                continue

            # 2. 将纯文本解析为 Python 字典
            try:
                resp_json = json.loads(resp)
            except json.JSONDecodeError as e:
                print(f"Json parse failed，error output: {e}")
                continue

            # print(f"Response costs: {time.time() - start:.2f}s")
            #
            # # outputs 返回的是一个列表（为了支持批处理），我们取第一个请求的输出文本
            # print(f"Generated text:\n\n{resp}\n\n")

            # exit(0)

            ret = {"video_path": item["src_video"], "reasoning": resp_json}

            fout.write(json.dumps(ret) + "\n")
            fout.flush()

    print("Done.")
