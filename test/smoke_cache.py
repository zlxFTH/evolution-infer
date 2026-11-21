import sys
from pathlib import Path
import torch
from tqdm import tqdm # 进度条

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src")) # 0 的意思是插入在最前面避免重名

from qwen import LLM
from load import load_model
from cache import Cache
from constant_random import *

def check_correct():
  max_diff = 0
  pass_ = True
  wa = []
  for i in range(case):
    assert inputs[i]["id"] == outputs[i]["id"] == answers[i]["id"]

    out = outputs[i]["logits"]
    ans = answers[i]["logits"]

    max_diff = max(max_diff, (out - ans).abs().max().item())
    # ok = (ans.argmax() == out.argmax()).item()
    ok = ((out - ans).abs().max() <= 0.2).item()
    if not ok:
      wa.append(i)
      pass_ = False

  if pass_:
    print(f"ac, max_diff {max_diff}")
  else:
    print(f"wa on {wa}")

def main():
  data_path = root / "data" / "random"
  model_path = root / "model" / "qwen"

  inputs = torch.load(data_path / "inputs.pt", map_location="cpu", weights_only=True)
  outputs_cache_off = []
  outputs_cache_on  = []

  assert case >= 10
  test_case = 10

  torch.set_default_dtype(torch.bfloat16)
  with torch.device("cuda"): # 用 with 临时切换到 gpu
    model = LLM()
  load_model(model, model_path)

  cache = Cache()
  
  with torch.inference_mode():
    for i in tqdm(range(test_case), desc="Running"):
      input_ids = inputs[i]["token"].cuda()
      seq_len = inputs[i]["token"].shape[0]
      positions = torch.arange(seq_len, dtype=torch.long, device="cuda")

      logits_cache_off = model(positions, input_ids, None)
      logits_cache_on  = model(positions, input_ids, None)
      outputs.append({
        "id": i,
        "logits": logits[-1].cpu()
      })

  torch.save(outputs, data_path / "outputs.pt")

if __name__ == "__main__":
  main()
