import sys
from pathlib import Path
from vllm import LLM, SamplingParams

def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else "你好，你是什么模型？"

    root = Path(__file__).resolve().parents[1]
    model_path = root / "model" / "qwen"

    llm = LLM(
        model=str(model_path),
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=2048,
        max_num_seqs=1,
        gpu_memory_utilization=0.9,
        enforce_eager=True,
        limit_mm_per_prompt={"image": 0, "video": 0},
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