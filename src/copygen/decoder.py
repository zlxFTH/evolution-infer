import math
from collections.abc import Sequence

import torch
import torch.nn as nn

from .heads import CopyGenHeads
from .matcher import match

def _propose_draft(
  ctx_ids: list[int],
  gen_ids: list[int],
  heads: CopyGenHeads,
  hidden: torch.Tensor,
  limit: int,
) -> tuple[int, list[int]]:
  if limit == 0:
    return 0, []

  match_length, copy_start = match(ctx_ids, gen_ids)
  if copy_start is None or copy_start == len(ctx_ids):
    return 0, []

  features = heads.features(hidden, match_length)
  if heads.trigger_head(features).argmax().item() != 1:
    return 0, []

  limit = min(limit, len(ctx_ids) - copy_start)
  log_length = heads.log_length(features).item()
  length = limit if log_length >= math.log(limit) else math.floor(math.exp(log_length))
  return match_length, ctx_ids[copy_start : copy_start + length]

def _predict_draft(
  prefix_output: tuple[torch.Tensor, torch.Tensor],
  draft_output: tuple[torch.Tensor, torch.Tensor],
  draft_ids: torch.Tensor,
  heads: CopyGenHeads,
  match_length: int,
  calibrate: bool,
) -> list[int]:
  next_logits, last_hidden = prefix_output
  draft_logits, draft_hidden = draft_output

  # 首个草稿由已确认前缀的最后一行预测，后续草稿使用块内左移一位的输出。
  if not calibrate:
    first = next_logits.argmax().unsqueeze(0)
    rest = draft_logits[:-1].argmax(dim=-1)
    return torch.cat((first, rest)).tolist()

  logits = torch.cat((next_logits.unsqueeze(0), draft_logits[:-1]), dim=0).float()
  hidden = torch.cat((last_hidden.unsqueeze(0), draft_hidden[:-1]), dim=0)
  # 沿同一处上下文继续拷贝时，匹配后缀每步增长一个 token。
  lengths = torch.arange(match_length, match_length + draft_ids.numel(), device=hidden.device)
  bias = heads.calib_head(heads.features(hidden, lengths)).to(logits.dtype)
  logits.scatter_add_(1, draft_ids.unsqueeze(-1), bias)
  return logits.argmax(dim=-1).tolist()


def _last_position(
  output: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
  logits, hidden = output
  # 只保留下一轮需要的一行，释放 Prefill 或验证块的完整输出。
  return logits[-1].clone(), hidden[-1].clone()


@torch.inference_mode()
def generate_copygen(
  model: nn.Module,
  cache,
  heads: CopyGenHeads,
  prompt_ids: Sequence[int] | torch.Tensor,
  max_new_tokens: int,
  calibrate: bool = False,
  *,
  eos_token_id: int | None = None,
  max_copy_len: int = 64,
) -> list[int]:
  """贪心生成并返回新增 token。max_copy_len 为 0 时走纯 GEN。

  主模型需要支持 return_hidden=True 和带缓存的多 token 因果前向。
  Cache 需要支持 snapshot/restore，且容量须覆盖 prompt 和生成预算。
  model 和 heads 应位于同一设备。
  """
  if max_new_tokens < 0 or max_copy_len < 0:
    raise ValueError("max_new_tokens and max_copy_len must be non-negative")
  if max_new_tokens == 0:
    return []

  device = next(model.parameters(), heads.ln.weight).device
  prompt = torch.as_tensor(prompt_ids, dtype=torch.long, device=device)
  if prompt.ndim != 1 or prompt.numel() == 0:
    raise ValueError("prompt_ids must be a non-empty one-dimensional sequence")
  ctx_ids = prompt_ids.tolist() if isinstance(prompt_ids, torch.Tensor) else list(prompt_ids)

  def forward(input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    start = cache.seq_len
    positions = torch.arange(start, start + input_ids.numel(), device=device)
    return model(positions, input_ids, cache, return_hidden=True)

  cache.reset()
  cache.is_prefill = True
  prefix_output = _last_position(forward(prompt))
  cache.is_prefill = False

  gen_ids: list[int] = []
  while len(gen_ids) < max_new_tokens:
    limit = min(max_copy_len, max_new_tokens - len(gen_ids))
    next_logits, last_hidden = prefix_output
    match_length, draft = _propose_draft(ctx_ids, gen_ids, heads, last_hidden, limit)

    if not draft:
      emitted = [next_logits.argmax().item()]
      output = forward(prompt.new_tensor(emitted))
    else:
      if eos_token_id in draft:
        draft = draft[: draft.index(eos_token_id) + 1]
      draft_ids = prompt.new_tensor(draft)
      snapshot = cache.snapshot()
      output = forward(draft_ids)
      predicted = _predict_draft(prefix_output, output, draft_ids, heads, match_length, calibrate)

      emitted = draft
      for index, (candidate, target) in enumerate(zip(draft, predicted, strict=True)):
        if candidate != target:
          cache.restore(snapshot)
          emitted = draft[:index] + [target]
          del output
          output = forward(prompt.new_tensor(emitted))
          break
      del snapshot

    gen_ids.extend(emitted)
    if emitted[-1] == eos_token_id or len(gen_ids) == max_new_tokens:
      break
    prefix_output = _last_position(output)
    del output

  return gen_ids
