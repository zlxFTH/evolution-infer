from pathlib import Path
import torch

from constant_random import *

def main():
  dir = Path(__file__).resolve().parents[1] / "data" / "random"
  inputs = torch.load(dir / "inputs.pt", map_location="cpu", weights_only=True)
  outputs = torch.load(dir / "outputs.pt", map_location="cpu", weights_only=True)
  answers = torch.load(dir / "answers.pt", map_location="cpu", weights_only=True)

  max_diff = 0
  pass_ = True
  wa = []
  max_error = 0.1 # max_diff 0.0074
  for i in range(case):
    assert inputs[i]["id"] == outputs[i]["id"] == answers[i]["id"]

    out = outputs[i]["logits"]
    ans = answers[i]["logits"]

    p_out = torch.softmax(out.float(), dim=-1)
    p_ans = torch.softmax(ans.float(), dim=-1)

    diff = (p_out - p_ans).abs().max()
    max_diff = max(max_diff, diff.item())
    ok = (diff <= max_error).item()
    if not ok:
      wa.append(i)
      pass_ = False

  if pass_:
      print(f"ac")
  else:
    print(f"wa on {wa}")
  print(f"max_diff {max_diff}")

if __name__ == "__main__":
  main()