from pathlib import Path
import torch
from tqdm import tqdm
from transformers import Qwen3_5ForConditionalGeneration

def main():
  root = Path(__file__).resolve().parents[1]
  model_path = root / "model" / "qwen"
  data_dir = root / "data" / "random"

  inputs = torch.load(data_dir / "inputs.pt", map_location="cpu", weights_only=True)

  model = Qwen3_5ForConditionalGeneration.from_pretrained(
    str(model_path),
    dtype=torch.bfloat16,
    device_map="cuda",
  ).eval()

  outputs = []
  with torch.inference_mode():
    for case in tqdm(inputs[:100], desc="Running ref"):
      input_ids = case["token"].unsqueeze(0).cuda()
      logits = model(input_ids=input_ids, use_cache=False).logits
      outputs.append({
        "id": case["id"],
        "logits": logits[0, -1].cpu(),
      })

  out_path = data_dir / "answers.pt"
  torch.save(outputs, out_path)

if __name__ == "__main__":
  main()