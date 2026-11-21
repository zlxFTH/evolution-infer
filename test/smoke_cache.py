import sys
from pathlib import Path
import torch
from tqdm import tqdm # 进度条
import time

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src")) # 0 的意思是插入在最前面避免重名

from qwen import LLM
from load import load_model
from cache import Cache
from constant_random import *

assert case >= 10
test_case = 10

def check_correct(inputs, outputs_cache_off, outputs_cache_on):
  max_diff = 0
  pass_ = True
  wa = []
  for i in range(test_case):
    seq_len = input[i]["token"].shape[0]
    assert seq_len * 2 == len(outputs_cache_off) == len(outputs_cache_on)

    for j in range(seq_len):
      assert inputs[i][j]["id"] == outputs_cache_off[i][j]["id"] == outputs_cache_on[i][j]["id"] 

      out_off = outputs_cache_off[i][j]["logits"]
      out_on  = outputs_cache_on[i][j]["logits"]

      diff = (out_off - out_on).abs().max()
      max_diff = max(max_diff, diff.item())
      ok = (diff <= max_error).item()
      if not ok:
        wa.append((i, j))
        pass_ = False

  if pass_:
    print(f"ac, max_diff {max_diff}")
  else:
    print(f"wa on {wa}")

def main():
  data_path  = root / "data" / "random"
  model_path = root / "model" / "qwen"

  inputs = torch.load(data_path / "inputs.pt", map_location="cpu", weights_only=True)

  torch.set_default_dtype(torch.bfloat16)
  with torch.device("cuda"): # 用 with 临时切换到 gpu
    model = LLM()
  load_model(model, model_path)
  
  with torch.inference_mode():
    def bench(mode: str, cache: Cache | None):
      torch.cuda.synchronize()
      beg = time.perf_counter()

      outputs = []
      for i in tqdm(range(test_case), desc=f"Running {mode}"):
        input_ids = inputs[i]["token"].cuda()
        seq_len = inputs[i]["token"].shape[0]
        positions = torch.arange(seq_len, dtype=torch.long, device="cuda")

        output_seq = False
        for j in range(seq_len):
          logits = model(positions, input_ids, cache)
          if cache is not None and j == 0:
            cache.is_prefill = False

          output_seq.append({
            "id": i,
            "step": j,
            "logits": logits[-1].cpu()
          })

          next_id = logits[-1].argmax().unsqueeze(0)
          if cache is None:
            positions = torch.cat((positions, positions.new_tensor([seq_len + j])))
            input_ids = torch.cat((input_ids, next_id))
          else:
            positions = positions.new_tensor([cache.seq_len])
            input_ids = next_id
        outputs.append(output_seq)

        if cache is not None:
          cache.reset()

      torch.cuda.synchronize()
      end = time.perf_counter()
      return outputs, end - beg

    cache = Cache()
    outputs_cache_off, t_off = bench("Cache Off", None)
    outputs_cache_on , t_on  = bench("Cache On", cache)
    check_correct(inputs, outputs_cache_off, outputs_cache_on)

    print(f"Cache off: {t_off:.2f}s")
    print(f"Cache on: {t_on:.2f}s")

if __name__ == "__main__":
  main()
