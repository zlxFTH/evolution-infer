import math
import torch

from spec_decode import generate_with_copy
from .matcher import MIN_COPY, SuffixMatcher

@torch.inference_mode()
def generate_copygen(
  model, cache, heads, prompt_ids, max_new_tokens,
  eos_token_id=None, max_copy_len=64, *, stats=None,
):
  ctx_ids = prompt_ids.tolist() if torch.is_tensor(prompt_ids) else list(prompt_ids)
  if heads is None or max_copy_len < MIN_COPY:
    return generate_with_copy(model, cache, ctx_ids, max_new_tokens, eos_token_id, 0, stats=stats)

  matcher = SuffixMatcher(ctx_ids)
  pos = len(ctx_ids)

  def propose(history, hidden, next_id, budget):
    nonlocal pos
    for tok in history[pos:]:
      matcher.append(tok)
    pos = len(history)

    match_len, start = matcher.best()
    if start is None or start + MIN_COPY > len(ctx_ids) or ctx_ids[start] != next_id:
      return []

    feat = heads.features(hidden[-1], match_len)
    if heads.trigger_head(feat).argmax().item() != 1:
      return []

    limit = min(budget, len(ctx_ids) - start)
    log_len = heads.log_length(feat).item()
    length = limit if log_len >= math.log(limit) else math.floor(math.exp(log_len))
    return ctx_ids[start : start + length]

  return generate_with_copy(
    model, cache, ctx_ids, max_new_tokens, eos_token_id, max_copy_len,
    propose=propose, stats=stats,
  )
