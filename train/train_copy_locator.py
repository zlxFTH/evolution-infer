import argparse
import json
import math
import random
from collections import Counter, defaultdict
from itertools import islice
from pathlib import Path

import torch
import torch.nn.functional as F

from cache_copy_locator import DEFAULT_CACHE, cache_features, token_ids
from train_copygen import ROOT, save_checkpoint
from copy_locator.candidates import MAX_CANDIDATES, StartIndex
from copy_locator.model import BLOCK_SIZE, CopyLocator, summarize_blocks


def sample_anchors(row, rng, count, min_copy):
  prompt, cot, output = (row[key] for key in ("prompt_ids", "cot_ids", "output_ids"))
  tokens = token_ids(row)
  segments = [0] * len(prompt) + [1] * len(cot) + [2] * len(output)
  same, cross, early, covered, boundaries = [], [], [], set(), set()
  for offset, spans in ((len(prompt), row["copy_cot"]), (len(prompt) + len(cot), row["copy_output"])):
    for i, l, r in spans:
      start = offset + i
      size = r - l
      for d in range(size - min_copy + 1):
        anchor = (start + d, l + d, r)
        (same if (l + d) // BLOCK_SIZE == (r - 1) // BLOCK_SIZE else cross).append(anchor)
        if d < min_copy:
          early.append(anchor)
      covered.update(range(start, start + size))
      boundaries.add(start + size)

  hard, ordinary = [], []
  for t in range(max(len(prompt), min_copy), len(tokens)):
    if t not in covered:
      (hard if t in boundaries else ordinary).append((t, -1, -1))

  def take(pool, n):
    if not pool:
      return []
    return rng.sample(pool, min(n, len(pool))) + rng.choices(pool, k=max(0, n - len(pool)))

  n_pos, n_hard = round(count * 0.45), round(count * 0.35)
  early_sample = rng.sample(early, min(len(early), n_pos // 4))
  n_rest = n_pos - len(early_sample)
  n_same = 0
  if same:
    n_same = min(n_rest, max(1, n_pos // 4)) if cross else n_rest
  anchors = early_sample + take(same, n_same) + take(cross, n_rest - n_same)
  anchors += take(hard, n_hard) + take(ordinary, count - n_pos - n_hard)
  anchors += take(same + cross + hard + ordinary, count - len(anchors))
  return tokens, segments[:-1], anchors


def start_supervision(tokens, prompt_len, anchors, min_copy):
  index = StartIndex(tokens[:prompt_len], min_copy)
  by_time = {}
  for t, left, right in sorted(set(anchors)):
    index.extend(tokens[index.visible:t])
    candidates = index.candidates(tokens[t])
    length = right - left
    target = 0
    if left >= 0:
      answer = tokens[t : t + length]
      for i, (start, _) in enumerate(candidates, start=1):
        if start + length <= t and tokens[start : start + length] == answer:
          target = i
          break
    pad = MAX_CANDIDATES - len(candidates)
    by_time[t] = ([start for start, _ in candidates] + [-1] * pad,
                  [overlap for _, overlap in candidates] + [0] * pad, target)
  return tuple(zip(*(by_time[t] for t, _, _ in anchors)))


def load_x(locator, path, segments):
  hidden = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
  device = locator.h_eos.device
  return locator.encode_step(
    [state.to(device, non_blocking=True) for state in hidden],
    torch.tensor(segments, device=device),
  )


def anchor_losses(locator, x, anchors, candidates, overlaps, targets):
  visible, left, right = anchors.T
  start_logits = locator.start_logits(x, visible, candidates, overlaps)
  start_loss = F.cross_entropy(start_logits, targets, reduction="sum")
  positive = left >= 0
  visible, left, last = visible[positive], left[positive], right[positive] - 1
  if not len(left):
    return start_loss, *start_loss.new_zeros(2).unbind()

  h = locator.mlp_span(x)
  h_sums, h_maxes = summarize_blocks(h)
  m_r = last // BLOCK_SIZE
  block_scores, prefix = locator.end_blocks(x, h, visible, left, h_sums, h_maxes)
  r_scores = locator.end_tokens(h, visible, left, m_r, prefix)
  return (
    start_loss,
    F.cross_entropy(block_scores, m_r, reduction="sum"),
    F.cross_entropy(r_scores, last % BLOCK_SIZE, reduction="sum"),
  )


@torch.inference_mode()
def validate(locator, validation):
  rng = random.Random(17)
  totals = defaultdict(Counter)
  for row, path in validation:
    tokens, segments, anchors = sample_anchors(row, rng, 32, locator.min_copy)
    if not anchors:
      continue
    counts = totals[row.get("domain", "data")]
    positions, overlaps, targets = start_supervision(tokens, len(row["prompt_ids"]), anchors, locator.min_copy)
    visible, left, right = torch.tensor(anchors, device="cuda").T
    x = load_x(locator, path, segments)
    logits = locator.start_logits(x, visible, torch.tensor(positions, device="cuda"), torch.tensor(overlaps, device="cuda"))
    probability, picks = logits.softmax(1).max(1)
    picks = picks.masked_fill(probability < 0.6, 0).tolist()
    for (t, anchor_left, _), candidates, pick, target in zip(anchors, positions, picks, targets):
      counts["positive"] += anchor_left >= 0
      counts["covered"] += target > 0
      if pick:
        counts["proposed"] += 1
        start = candidates[pick - 1]
        counts["matched"] += tokens[t : t + locator.min_copy] == tokens[start : start + locator.min_copy]
    positive = right >= 0
    if positive.any():
      visible, left, right = (v[positive] for v in (visible, left, right))
      h = locator.mlp_span(x)
      scores, prefix = locator.end_blocks(x, h, visible, left, *summarize_blocks(h))
      blocks = scores.argmax(1)
      ends = blocks * BLOCK_SIZE + locator.end_tokens(h, visible, left, blocks, prefix).argmax(1) + 1
      counts["end_exact"] += (ends == right).sum().item()
      counts["end_long"] += (ends > right).sum().item()
  for domain, c in sorted(totals.items()):
    print(f"val {domain} top32={c['covered']}/{c['positive']} prefix5={c['matched']}/{c['proposed']} "
          f"oracle_l_end_exact={c['end_exact']}/{c['positive']} end_long={c['end_long']}/{c['positive']}", flush=True)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", type=Path, default=ROOT / "model/qwen")
  parser.add_argument("--data", type=Path, default=ROOT / "data/copygen_train_code.jsonl")
  parser.add_argument("--validation-data", type=Path)
  parser.add_argument("--init", type=Path)
  parser.add_argument("--output", type=Path, default=ROOT / "checkpoints/copy_locator_candidates.pt")
  parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
  parser.add_argument("--limit", type=int)
  parser.add_argument("--epochs", type=int, default=12)
  parser.add_argument("--val-size", type=int, default=41)
  parser.add_argument("--grad-accum", type=int, default=8)
  parser.add_argument("--anchors", type=int, default=32)
  parser.add_argument("--lr", type=float, default=3e-4)
  args = parser.parse_args()

  with args.data.open(encoding="utf-8") as stream:
    rows = list(islice(map(json.loads, stream), args.limit))
  val_rows = []
  if args.validation_data is not None:
    with args.validation_data.open(encoding="utf-8") as stream:
      val_rows = list(map(json.loads, stream))
  print(f"rows={len(rows)} cot={sum(bool(row['cot_ids']) for row in rows)}", flush=True)
  torch.manual_seed(42)
  rng = random.Random(42)
  paths = cache_features(rows + val_rows, args.model, args.cache)
  d_model = torch.load(paths[0], weights_only=True, mmap=True)[0].shape[-1]
  locator = CopyLocator(d_model).cuda()
  train = list(zip(rows, paths))
  if args.validation_data is None:
    rng.shuffle(train)
    val_size = min(args.val_size, len(train) - 1)
    validation, train = train[:val_size], train[val_size:]
  else:
    validation = list(zip(val_rows, paths[len(rows):]))
  print(f"train={len(train)} validation_from_train={len(validation)}", flush=True)
  if args.init is not None:
    locator.load_state_dict(torch.load(args.init, map_location="cpu", weights_only=True))
    print(f"initialized from {args.init}", flush=True)
    validate(locator, validation)

  optimizer = torch.optim.AdamW(locator.parameters(), lr=args.lr)
  steps = args.epochs * math.ceil(len(train) / args.grad_accum)
  schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  step = 0
  for epoch in range(args.epochs):
    rng.shuffle(train)
    for start in range(0, len(train), args.grad_accum):
      batch = [
        (len(row["prompt_ids"]), path, *sample_anchors(row, rng, args.anchors, locator.min_copy))
        for row, path in train[start : start + args.grad_accum]
      ]
      n = sum(len(anchors) for *_, anchors in batch)
      if not n:
        continue
      n_pos = sum(l >= 0 for *_, anchors in batch for _, l, _ in anchors)
      scale = torch.tensor([1 / n, *([1 / max(n_pos, 1)] * 2)], device="cuda")
      losses = torch.zeros_like(scale)
      optimizer.zero_grad(set_to_none=True)
      for prompt_len, path, tokens, segments, anchors in batch:
        if not anchors:
          continue
        supervision = start_supervision(tokens, prompt_len, anchors, locator.min_copy)
        args_gpu = [torch.tensor(values, device="cuda") for values in (anchors, *supervision)]
        loss = torch.stack(anchor_losses(locator, load_x(locator, path, segments), *args_gpu)) * scale
        loss.sum().backward()
        losses += loss.detach()
      torch.nn.utils.clip_grad_norm_(locator.parameters(), 1.0)
      optimizer.step()
      schedule.step()
      step += 1
      if step == 1 or step % 10 == 0 or step == steps:
        start_loss, r_block, r_token = losses.tolist()
        print(f"step {step}/{steps} start={start_loss:.4f} r_block={r_block:.4f} r_token={r_token:.4f}", flush=True)
    print(f"epoch {epoch + 1}/{args.epochs}", flush=True)
    validate(locator, validation)
    save_checkpoint(locator, args.output)


if __name__ == "__main__":
  main()
