import argparse
import json
from array import array
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from train_copygen import ROOT, load_backbone
from copy_locator.model import HIDDEN_LAYERS

DEFAULT_CACHE = ROOT / "tmp/copy_locator_hidden"


def token_ids(row):
  return row["prompt_ids"] + row["cot_ids"] + row["output_ids"]


def cache_paths(rows, model, cache):
  weights = [model / "config.json", *sorted(model.glob("*.safetensors"))]
  signature = [f"{file.name}:{file.stat().st_size}:{file.stat().st_mtime_ns}" for file in weights]
  key = sha256(repr((HIDDEN_LAYERS, signature)).encode()).hexdigest()[:16]
  directory = cache / key
  return [directory / (sha256(array("I", token_ids(row)[:-1]).tobytes()).hexdigest() + ".pt") for row in rows]


def cache_features(rows, model, cache):
  paths = cache_paths(rows, model, cache)
  missing = {}
  for row, path in zip(rows, paths):
    if not path.exists():
      missing[path] = row
  print(f"features cached={len(paths) - len(missing)} missing={len(missing)}", flush=True)
  if not missing:
    return paths

  backbone = load_backbone(model)
  states = []
  for layer in HIDDEN_LAYERS:
    backbone.layers[layer].register_forward_hook(lambda _layer, _inputs, output: states.append(output[0]))

  cache.mkdir(parents=True, exist_ok=True)
  paths[0].parent.mkdir(exist_ok=True)
  with TemporaryDirectory(dir=cache) as staging, torch.inference_mode():
    for i, (path, row) in enumerate(missing.items(), start=1):
      states.clear()
      tokens = torch.tensor([token_ids(row)[:-1]], device="cuda")
      backbone(tokens, use_cache=False)
      tmp = Path(staging) / path.name
      torch.save(tuple(hidden.cpu() for hidden in states), tmp)
      tmp.replace(path)
      if i == 1 or i % 50 == 0 or i == len(missing):
        print(f"features {i}/{len(missing)}", flush=True)
  return paths


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", type=Path, default=ROOT / "model/qwen")
  parser.add_argument("--data", type=Path, default=ROOT / "data/copygen_train_code.jsonl")
  parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
  args = parser.parse_args()
  with args.data.open() as stream:
    rows = list(map(json.loads, stream))
  cache_features(rows, args.model, args.cache)
