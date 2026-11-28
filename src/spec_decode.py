from dataclasses import dataclass
from time import perf_counter

import torch


@dataclass
class DecodeStats:
  steps: int = 0
  proposals: int = 0
  drafted_tokens: int = 0
  accepted_tokens: int = 0
  prefill_seconds: float = 0.0
  accepted_spans: list[tuple[int, int]] | None = None


@torch.inference_mode()
def generate_with_copy(
  model, cache, prompt_ids, max_new_tokens, eos_token_id, max_copy_len,
  *, propose=None, commit=None, hidden_layers=(), min_copy_len=5, stats=None,
):
  if max_new_tokens <= 0:
    return []
  history = prompt_ids.tolist() if torch.is_tensor(prompt_ids) else list(prompt_ids)
  if not history:
    raise ValueError("prompt_ids must not be empty")
  prompt_len = len(history)
  stop_ids = {eos_token_id} if isinstance(eos_token_id, int) else set(eos_token_id or ())
  device = next(model.parameters()).device

  def forward(ids, verify=False):
    ids = torch.as_tensor(ids, dtype=torch.long, device=device)
    positions = torch.arange(cache.seq_len, cache.seq_len + len(ids), device=device)
    if stats is not None and not cache.is_prefill:
      stats.steps += 1
    return model(positions, ids, cache, return_hidden=True, hidden_layers=hidden_layers,
                 logits_to_keep=0 if verify else 1)

  cache.reset()
  prefill_start = perf_counter() if stats is not None else None
  logits, hidden = forward(history)
  cache.is_prefill = False
  if commit is not None:
    commit(history, hidden)
  if stats is not None:
    if device.type == "cuda":
      torch.cuda.synchronize(device)
    stats.prefill_seconds += perf_counter() - prefill_start

  while len(history) - prompt_len < max_new_tokens:
    next_id = logits[-1].argmax().item()
    budget = min(max_copy_len, max_new_tokens - (len(history) - prompt_len))
    draft = propose(history, hidden, next_id, budget) if propose is not None and budget >= min_copy_len else []
    draft = draft[:budget]
    stop = next((i + 1 for i, token in enumerate(draft) if token in stop_ids), len(draft))
    draft = draft[:stop]
    can_verify = len(draft) >= min_copy_len and draft[0] == next_id

    if can_verify:
      snapshot = cache.snapshot()
      trial_logits, trial_hidden = forward(draft, verify=True)
      predicted = [next_id, *trial_logits[:-1].argmax(-1).tolist()]
      accepted = next((i for i, pair in enumerate(zip(draft, predicted)) if pair[0] != pair[1]), len(draft))
      if stats is not None:
        stats.proposals += 1
        stats.drafted_tokens += len(draft)
        stats.accepted_tokens += accepted
        if stats.accepted_spans is not None:
          stats.accepted_spans.append((len(history) - prompt_len, accepted))

      if accepted == len(draft):
        emitted, logits, hidden = draft, trial_logits, trial_hidden
      else:
        cache.restore(snapshot)
        emitted = draft[:accepted] + [predicted[accepted]]
        logits, hidden = forward(emitted)
    else:
      emitted = [next_id]
      logits, hidden = forward(emitted)

    history.extend(emitted)
    if emitted[-1] in stop_ids:
      break
    if commit is not None and len(history) - prompt_len < max_new_tokens:
      commit(emitted, hidden)

  return history[prompt_len:]
