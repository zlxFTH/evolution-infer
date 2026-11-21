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
  for i in range(case):
    assert inputs[i]["id"] == outputs[i]["id"] == answers[i]["id"]

    out = outputs[i]["logits"]
    ans = answers[i]["logits"]

    max_diff = max(max_diff, (out - ans).abs().max().item())
    # ok = (ans.argmax() == out.argmax()).item()
    ok = abs(ans.max() - out.max()) <= 0.2
    if not ok:
      wa.append(i)
      pass_ = False

  if pass_:
    print(f"ac, max_diff {max_diff}")
  else:
    print(f"wa on {wa}")


if __name__ == "__main__":
  main()