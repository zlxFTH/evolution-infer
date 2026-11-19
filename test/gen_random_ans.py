from transformers import Qwen3_5ForConditionalGeneration

model = Qwen3_5ForConditionalGeneration.from_pretrained(
    str(model_path),
    dtype=torch.bfloat16,
    device_map="cuda",
).eval()