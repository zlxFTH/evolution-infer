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
  model, cache, prompt_ids, max_new_tokens, eos_token_id=None,
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
    budget = max_new_tokens - (len(history) - prompt_len)
    raw = propose(history, hidden, next_id, budget) if propose is not None and budget >= min_copy_len else []
    raw_cands = raw if raw and isinstance(raw[0], list) else ([raw] if raw else [])
    cands = []
    for cand in raw_cands:
      cand = cand[:budget]
      stop = next((i + 1 for i, tok in enumerate(cand) if tok in stop_ids), len(cand))
      cand = cand[:stop]
      if len(cand) >= min_copy_len and cand[0] == next_id and cand not in cands:
        cands.append(cand)

    if cands:
      snapshot, owner = None, None
      best_score = best = None
      for idx, draft in enumerate(cands):
        if len(draft) == 1:
          ver_logits = logits[-1:]
          predicted, accepted, trial = [next_id], 1, None
        else:
          if snapshot is None:
            snapshot = cache.snapshot()
          elif owner is not None:
            cache.restore(snapshot)
          trial_logits, trial_hidden = forward(draft, verify=True)
          owner = idx
          predicted = [next_id, *trial_logits[:-1].argmax(-1).tolist()]
          accepted = next((i for i, pair in enumerate(zip(draft, predicted)) if pair[0] != pair[1]), len(draft))
          ver_logits = torch.cat((logits[-1:], trial_logits[:max(0, accepted - 1)]))
          trial = (trial_logits, trial_hidden)

        top2 = ver_logits[:accepted].float().topk(min(2, ver_logits.size(-1)), dim=-1).values
        margin_sum = (top2[:, 0] - top2[:, 1]).sum().item() if top2.size(-1) > 1 else top2[:, 0].sum().item()
        lse = torch.logsumexp(ver_logits[:accepted].float(), dim=-1)
        logprob_sum = (top2[:, 0] - lse).sum().item()
        score = (accepted, margin_sum, logprob_sum, -idx)
        if best_score is None or score > best_score:
          best_score, best = score, (idx, draft, predicted, accepted, trial)

      best_idx, draft, predicted, accepted, trial = best
      if stats is not None:
        stats.proposals += 1
        stats.drafted_tokens += len(draft)
        stats.accepted_tokens += accepted
        if stats.accepted_spans is not None:
          stats.accepted_spans.append((len(history) - prompt_len, accepted))

      if accepted == len(draft) and owner == best_idx:
        emitted, logits, hidden = draft, *trial
      else:
        if owner is not None:
          cache.restore(snapshot)
        emitted = draft if accepted == len(draft) else draft[:accepted] + [predicted[accepted]]
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
