import argparse
import json
import math
import random
from itertools import islice
from pathlib import Path

import torch
import torch.nn.functional as F

from common import ROOT, load_backbone, save_checkpoint
from copygen import CopyGenHeads, MIN_COPY, MIN_MATCH, SuffixMatcher


def training_example(row):
  prompt = row["prompt_ids"]
  target = row["cot_ids"] + row["output_ids"]
  if not target:
    return prompt, len(prompt), [], []

  ends_by_tok = {}
  for idx, tok in enumerate(prompt):
    ends_by_tok.setdefault(tok, []).append(idx)

  lengths = [0] * len(target)
  next_row = {}
  for i in range(len(target) - 1, -1, -1):
    starts = ends_by_tok.get(target[i], ())
    if not starts:
      next_row = {}
      continue
    row_map = {}
    best = 0
    for s in starts:
      m = 1 + next_row.get(s + 1, 0)
      row_map[s] = m
      if m > best:
        best = m
    lengths[i] = best
    next_row = row_map

  matcher = SuffixMatcher(prompt)
  matched = []
  for token in target:
    matched.append(matcher.longest)
    matcher.append(token)
  return prompt + target[:-1], len(prompt), matched, lengths


def train_loss(backbone, heads, example, device="cuda"):
  ids, prompt_len, matched, lengths = example
  tokens = torch.tensor([ids], device=device)
  with torch.no_grad():
    hidden = backbone(tokens, use_cache=False).last_hidden_state[0, prompt_len - 1 :]

  matched_t = torch.tensor(matched, device=device)
  lengths_t = torch.tensor(lengths, device=device)
  features = heads.features(hidden, matched_t)
  copy = (matched_t >= MIN_MATCH) & (lengths_t >= MIN_COPY)
  trigger_loss = F.cross_entropy(heads.trigger_head(features).float(), copy.long())
  if copy.any():
    length_loss = F.smooth_l1_loss(
      heads.log_length(features[copy]).squeeze(-1),
      lengths_t[copy].float().log(),
      beta=1.0,
    )
  else:
    length_loss = trigger_loss.new_zeros(())
  return trigger_loss, length_loss


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
  examples = [example for example in examples if example[2]]
  if not examples:
    raise ValueError("No training examples found")
  positive = sum(m >= MIN_MATCH and l >= MIN_COPY for _, _, matched, lengths in examples for m, l in zip(matched, lengths))
  negative = sum(len(matched) for _, _, matched, _ in examples) - positive
  print(f"rows={len(examples)} copy={positive} gen={negative}", flush=True)

  torch.manual_seed(42)
  rng = random.Random(42)
  backbone = load_backbone(args.model)
  heads = CopyGenHeads(backbone.config.hidden_size).to("cuda").train()
  params = list(heads.parameters())
  optimizer = torch.optim.AdamW(params, lr=args.lr)
  steps = args.epochs * math.ceil(len(examples) / args.grad_accum)
  schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  step = 0
  for _ in range(args.epochs):
    rng.shuffle(examples)
    for start in range(0, len(examples), args.grad_accum):
      batch = examples[start : start + args.grad_accum]
      optimizer.zero_grad(set_to_none=True)
      losses = torch.zeros(2, device="cuda")
      for example in batch:
        trigger, length = train_loss(backbone, heads, example)
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
