from collections import deque

import torch

from spec_decode import generate_with_copy
from .candidates import StartIndex
from .model import BLOCK_SIZE, HIDDEN_LAYERS, summarize_blocks


@torch.inference_mode()
def generate_copy_locator(
  model, cache, locator, prompt_ids, max_new_tokens, eos_token_id=None,
  *, cot_end_ids=(), min_start_prob=0.6, stats=None,
):
  """返回新增 token。开启 CoT 时传入结束 CoT 的分隔 token 序列。"""
  if max_new_tokens <= 0:
    return []
  capacity = len(prompt_ids) + max_new_tokens
  x = locator.seg_emb.weight.new_empty((capacity, locator.projs[0].out_features))
  h = locator.h_eos.new_empty((capacity, locator.h_eos.numel()))
  blocks = (capacity + BLOCK_SIZE - 1) // BLOCK_SIZE
  h_sums = h.new_empty((blocks, h.shape[-1]))
  summaries = (h_sums, torch.empty_like(h_sums))
  visible = 0
  starts = None
  cot_end_ids = tuple(cot_end_ids)
  tail = deque(maxlen=len(cot_end_ids))
  stage = 1 if cot_end_ids else 2

  def commit(ids, hidden):
    nonlocal visible, stage, starts
    if visible == 0:
      starts = StartIndex(ids, locator.min_copy)
      segments = [0] * len(ids)
    else:
      starts.extend(ids)
      segments = []
      for token in ids:
        segments.append(stage)
        if stage == 1:
          tail.append(token)
          if tuple(tail) == cot_end_ids:
            stage = 2

    end = visible + len(ids)
    encoded = locator.encode_step(hidden, torch.tensor(segments, device=x.device))
    x[visible:end] = encoded
    h[visible:end] = locator.mlp_span(encoded)
    block = visible // BLOCK_SIZE
    updated = summarize_blocks(h[block * BLOCK_SIZE : end])
    for cache_blocks, new_blocks in zip(summaries, updated):
      cache_blocks[block : block + len(new_blocks)] = new_blocks
    visible = end

  def propose(history, _hidden, next_id, budget):
    count = (visible + BLOCK_SIZE - 1) // BLOCK_SIZE
    span = locator.locate(
      x[:visible], starts.candidates(next_id), h[:visible], tuple(s[:count] for s in summaries),
      min_start_prob=min_start_prob,
    )
    if span is None:
      return []
    l, r = span
    return history[l : min(r, l + budget)]

  return generate_with_copy(
    model, cache, prompt_ids, max_new_tokens, eos_token_id,
    propose=propose, commit=commit, hidden_layers=HIDDEN_LAYERS,
    min_copy_len=locator.min_copy, stats=stats,
  )
