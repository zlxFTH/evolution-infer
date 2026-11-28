import math
import torch

from spec_decode import generate_with_copy
from .matcher import MIN_COPY, MIN_MATCH, SuffixMatcher

@torch.inference_mode()
def generate_copygen(
  model, cache, heads, prompt_ids, max_new_tokens,
  eos_token_id=None, *, stats=None,
):
  ctx_ids = prompt_ids.tolist() if torch.is_tensor(prompt_ids) else list(prompt_ids)
  if heads is None:
    return generate_with_copy(model, cache, ctx_ids, max_new_tokens, eos_token_id, stats=stats)

  matcher = SuffixMatcher(ctx_ids)
  pos = len(ctx_ids)

  def propose(history, hidden, next_id, budget):
    nonlocal pos
    for tok in history[pos:]:
      matcher.append(tok)
    pos = len(history)

    if matcher.longest < MIN_MATCH:
      return []
    starts = matcher.candidates() or matcher.fallback(next_id)
    starts = [s for s in starts if ctx_ids[s] == next_id]
    if not starts:
      return []

    feat = heads.features(hidden[-1], matcher.longest)
    if heads.trigger_head(feat).argmax().item() != 1:
      return []

    log_len = heads.log_length(feat).item()
    prop_len = budget if log_len >= math.log(budget) else max(1, math.floor(math.exp(log_len)))
    return [ctx_ids[s : s + min(prop_len, budget, len(ctx_ids) - s)] for s in starts]

  return generate_with_copy(
    model, cache, ctx_ids, max_new_tokens, eos_token_id,
    propose=propose, min_copy_len=MIN_COPY, stats=stats,
  )
