import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load_backbone(path):
  from transformers import AutoModelForImageTextToText

  model = AutoModelForImageTextToText.from_pretrained(
    path, dtype=torch.bfloat16, device_map={"": "cuda"},
    attn_implementation="sdpa", local_files_only=True,
  )
  return model.model.language_model.eval().requires_grad_(False)


def save_checkpoint(model, path):
  tmp = path.with_suffix(".tmp")
  torch.save(model.state_dict(), tmp)
  tmp.replace(path)
  print(f"saved {path}", flush=True)
