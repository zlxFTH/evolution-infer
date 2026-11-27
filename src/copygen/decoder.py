import math

import torch

from spec_decode import generate_with_copy
from .matcher import MIN_COPY, SuffixMatcher


@torch.inference_mode()
def generate_copygen(
  model, cache, heads, prompt_ids, max_new_tokens,
  calibrate=False, eos_token_id=None, max_copy_len=64, *, stats=None,
):
  """返回新增 token。heads=None 时走普通贪心解码。"""
  ctx_ids = prompt_ids.tolist() if torch.is_tensor(prompt_ids) else list(prompt_ids)
  if heads is None or max_copy_len < MIN_COPY:
    return generate_with_copy(model, cache, ctx_ids, max_new_tokens, eos_token_id, 0, stats=stats)
  matcher = SuffixMatcher(ctx_ids)
  matched_to, match_length = len(ctx_ids), 0

  def propose(history, hidden, next_id, budget):
    nonlocal matched_to, match_length
    for token in history[matched_to:]:
      matcher.append(token)
    matched_to = len(history)
    match_length, start = matcher.best()
    if start is None or start + MIN_COPY > len(ctx_ids):
      return []
    if not calibrate and ctx_ids[start] != next_id:
      return []
    features = heads.features(hidden[-1], match_length)
    if heads.trigger_head(features).argmax().item() != 1:
      return []
    limit = min(budget, len(ctx_ids) - start)
    log_length = heads.log_length(features).item()
    length = limit if log_length >= math.log(limit) else math.floor(math.exp(log_length))
    return ctx_ids[start : start + length]

  def calibrate_logits(draft, last_logits, hidden, trial_logits, trial_hidden):
    logits = torch.cat((last_logits[None], trial_logits[:-1])).float()
    states = torch.cat((hidden[-1:], trial_hidden[:-1]))
    lengths = torch.arange(match_length, match_length + len(draft), device=logits.device)
    bias = heads.calib_head(heads.features(states, lengths)).to(logits.dtype)
    logits.scatter_add_(1, lengths.new_tensor(draft)[:, None], bias)
    return logits.argmax(-1).tolist()

  return generate_with_copy(
    model, cache, ctx_ids, max_new_tokens, eos_token_id, max_copy_len,
    propose=propose, calibrate=calibrate_logits if calibrate else None, stats=stats,
  )
