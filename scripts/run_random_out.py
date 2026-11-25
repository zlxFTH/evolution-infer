import sys
from pathlib import Path
import torch
from tqdm import tqdm # 进度条

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src")) # 0 的意思是插入在最前面避免重名

from qwen import LLM
from load import load_model
from constant_random import *

def main():
  data_path = root / "data" / "random"
  model_path = root / "model" / "qwen"

  inputs = torch.load(data_path / "inputs.pt", map_location="cpu", weights_only=True)
  outputs = []

  torch.set_default_dtype(torch.bfloat16)
  with torch.device("cuda"): # 用 with 临时切换到 gpu
    model = LLM()
  load_model(model, model_path)

  with torch.inference_mode():
    for i in tqdm(range(case), desc="Running"):
      input_ids = inputs[i]["token"].cuda()
      seq_len = inputs[i]["token"].shape[0]
      positions = torch.arange(seq_len, dtype=torch.long, device="cuda")
      logits = model(positions, input_ids, None)
      outputs.append({
        "id": i,
        "logits": logits[-1].cpu()
      })

  torch.save(outputs, data_path / "outputs.pt")

if __name__ == "__main__":
  main()
