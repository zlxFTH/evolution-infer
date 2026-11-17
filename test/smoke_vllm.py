import sys
from pathlib import Path
from vllm import LLM, SamplingParams

def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else "你好，你是什么模型？"

    root = Path(__file__).resolve().parents[1]
    model_path = root / "model" / "qwen"

    llm = LLM(
        model=str(model_path),
        dtype="bfloat16", # 精度 bf16
        tensor_parallel_size=1, # 张量并行度，大矩阵拆几个 gpu 
        max_model_len=10240, # 单条序列总长度上限
        max_num_seqs=1, # 每次处理 1 个序列
        gpu_memory_utilization=0.9, # 显存预算为每张 GPU 总显存的 90%
        enforce_eager=True, # 强制即时执行模式，关闭 torch.compile 和 CUDA Graph 优化
        limit_mm_per_prompt={"image": 0, "video": 0}, # 文本输入
    )

    outputs = llm.chat(
        messages=[{"role": "user", "content": prompt}],
        sampling_params=SamplingParams(
            temperature=0.0,
            max_tokens=1024,
        ),
        chat_template_kwargs={"enable_thinking": False},
    )

    print("\n")
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
    print(f"Stop reason: {}")

if __name__ == "__main__":
    main()