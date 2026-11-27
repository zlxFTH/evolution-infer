import argparse
import json
import math
import random
import sys
from itertools import islice
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from copygen import CopyGenHeads
from copygen.matcher import MIN_COPY, SuffixMatcher


def training_example(row):
  prompt = row["prompt_ids"]
  target = row["cot_ids"] + row["output_ids"]
  matcher = SuffixMatcher(prompt)
  labels = []
  for i, token in enumerate(target):
    matched, start = matcher.best()
    if start is not None and start + MIN_COPY <= len(prompt):
      length = 0
      limit = min(len(prompt) - start, len(target) - i)
      while length < limit and prompt[start + length] == target[i + length]:
        length += 1
      # 第 i 个目标 token 用它前一位置的 hidden state 预测。
      labels.append((len(prompt) + i - 1, matched, length))
    matcher.append(token)
  return prompt + target[:-1], labels


def load_backbone(path):
  from transformers import AutoModelForImageTextToText

  model = AutoModelForImageTextToText.from_pretrained(
    path, dtype=torch.bfloat16, device_map={"": "cuda"},
    attn_implementation="sdpa", local_files_only=True,
  )
  return model.model.language_model.eval().requires_grad_(False)


def train_loss(backbone, heads, example, weight):
  ids, labels = example
  tokens = torch.tensor([ids], device=weight.device)
  positions, matched, lengths = torch.tensor(labels, device=weight.device).T
  with torch.no_grad():
    hidden = backbone(tokens, use_cache=False).last_hidden_state[0, positions]

  features = heads.features(hidden, matched)
  copy = lengths >= MIN_COPY
  trigger_loss = F.cross_entropy(heads.trigger_head(features), copy.long(), weight=weight)
  errors = F.smooth_l1_loss(
    heads.log_length(features).squeeze(-1), lengths.float().clamp_min(1).log(), reduction="none",
  )
  length_loss = (errors * copy).sum() / copy.sum().clamp_min(1)
  return trigger_loss, length_loss


def save_checkpoint(model, path):
  tmp = path.with_suffix(".tmp")
  torch.save(model.state_dict(), tmp)
  tmp.replace(path)
  print(f"saved {path}", flush=True)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", type=Path, default=ROOT / "model/qwen")
  parser.add_argument("--data", type=Path, default=ROOT / "data/copygen_train_code.jsonl")
  parser.add_argument("--output", type=Path, default=ROOT / "checkpoints/copygen_code.pt")
  parser.add_argument("--limit", type=int)
  parser.add_argument("--epochs", type=int, default=1)
  parser.add_argument("--grad-accum", type=int, default=16)
  parser.add_argument("--lr", type=float, default=1e-4)
  args = parser.parse_args()

  with args.data.open(encoding="utf-8") as stream:
    examples = [training_example(row) for row in islice(map(json.loads, stream), args.limit)]
  examples = [example for example in examples if example[1]]
  if not examples:
    raise ValueError("No examples with eligible source matches")
  positive = sum(length >= MIN_COPY for _, labels in examples for _, _, length in labels)
  negative = sum(len(labels) for _, labels in examples) - positive
  print(f"rows={len(examples)} copy={positive} gen={negative}", flush=True)

  torch.manual_seed(42)
  rng = random.Random(42)
  backbone = load_backbone(args.model)
  heads = CopyGenHeads(backbone.config.hidden_size).to("cuda")
  # 无损模式仅训练 trigger 和 length。将可选校准头初始化为零偏置。
  heads.calib_head.requires_grad_(False)
  torch.nn.init.zeros_(heads.calib_head[-1].weight)
  torch.nn.init.zeros_(heads.calib_head[-1].bias)
  params = [p for p in heads.parameters() if p.requires_grad]
  optimizer = torch.optim.AdamW(params, lr=args.lr)
  steps = args.epochs * math.ceil(len(examples) / args.grad_accum)
  schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
  weight = torch.tensor([negative, positive], dtype=torch.float32, device="cuda").clamp_min(1).reciprocal()
  args.output.parent.mkdir(parents=True, exist_ok=True)
  step = 0
  for _ in range(args.epochs):
    rng.shuffle(examples)
    for start in range(0, len(examples), args.grad_accum):
      batch = examples[start : start + args.grad_accum]
      optimizer.zero_grad(set_to_none=True)
      losses = weight.new_zeros(2)
      for example in batch:
        trigger, length = train_loss(backbone, heads, example, weight)
        ((trigger + length) / len(batch)).backward()
        losses += torch.stack((trigger.detach(), length.detach()))
      torch.nn.utils.clip_grad_norm_(params, 1.0)
      optimizer.step()
      schedule.step()
      step += 1
      trigger, length = (losses / len(batch)).tolist()
      print(f"step {step}/{steps} trigger={trigger:.4f} length={length:.4f}", flush=True)
      if step == 1 or step % 10 == 0 or step == steps:
        save_checkpoint(heads, args.output)


if __name__ == "__main__":
  main()
