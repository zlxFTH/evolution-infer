import math
import torch

from .matcher import match

@torch.inference_mode()
def generate_copygen(model, cache, heads, prompt_ids, max_new_tokens, calibrate=False, eos_token_id=None, max_copy_len=64):
  if max_new_tokens == 0:
    return []

  device = next(model.parameters()).device
  input_ids = torch.as_tensor(prompt_ids, dtype=torch.long, device=device)
  ctx_ids = input_ids.tolist()

  # 1. Prefill
  cache.reset()
  cache.is_prefill = True
  positions = torch.arange(input_ids.numel(), dtype=torch.long, device=device)
  logits, hidden = model(positions, input_ids, cache, return_hidden=True)
  last_logits = logits[-1].clone()
  last_hidden = hidden[-1].clone()
  cache.is_prefill = False

  gen_ids = []
  while len(gen_ids) < max_new_tokens:
    # 2. 检查后缀匹配 + TriggerHead 判断是否拷贝
    m_len, copy_start = match(ctx_ids, gen_ids)
    draft = []
    if max_copy_len > 0 and copy_start is not None and copy_start < len(ctx_ids):
      feat = heads.features(last_hidden, m_len)
      if heads.trigger_head(feat).argmax().item() == 1:
        limit = min(max_copy_len, max_new_tokens - len(gen_ids), len(ctx_ids) - copy_start)
        log_len = heads.log_length(feat).item()
        copy_len = limit if log_len >= math.log(limit) else math.floor(math.exp(log_len))
        draft = ctx_ids[copy_start : copy_start + copy_len]
        if eos_token_id in draft:
          draft = draft[: draft.index(eos_token_id) + 1]

    # 3A. 不拷贝：普通单步 Decode
    if not draft:
      next_id = last_logits.argmax().item()
      step_ids = input_ids.new_tensor([next_id])
      step_pos = input_ids.new_tensor([cache.seq_len])
      logits, hidden = model(step_pos, step_ids, cache, return_hidden=True)
      gen_ids.append(next_id)
      last_logits = logits[-1].clone()
      last_hidden = hidden[-1].clone()
      if next_id == eos_token_id:
        break
      continue

    # 3B. 拷贝：一次喂整段草稿并行验证
    snap = cache.snapshot()
    draft_ids = input_ids.new_tensor(draft)
    draft_pos = torch.arange(cache.seq_len, cache.seq_len + len(draft), dtype=torch.long, device=device)
    logits, hidden = model(draft_pos, draft_ids, cache, return_hidden=True)

    # 拼出主模型对 draft 每一位的预测（第 0 位来自 last_logits，其余来自 logits[:-1]）
    verify_logits = torch.cat((last_logits.unsqueeze(0), logits[:-1]), dim=0)
    if calibrate:
      verify_logits = verify_logits.float()
      verify_hidden = torch.cat((last_hidden.unsqueeze(0), hidden[:-1]), dim=0)
      lengths = torch.arange(m_len, m_len + len(draft), device=device)
      bias = heads.calib_head(heads.features(verify_hidden, lengths)).to(verify_logits.dtype)
      verify_logits.scatter_add_(1, draft_ids.unsqueeze(-1), bias)
    pred = verify_logits.argmax(dim=-1).tolist()

    # 4. 逐位核对，全对就直接收下；有错就回滚重放已对前缀 + 修正词
    accept = len(draft)
    for i in range(len(draft)):
      if draft[i] != pred[i]:
        accept = i
        break

    if accept == len(draft):
      gen_ids.extend(draft)
      last_logits = logits[-1].clone()
      last_hidden = hidden[-1].clone()
    else:
      cache.restore(snap)
      emitted = draft[:accept] + [pred[accept]]
      fix_ids = input_ids.new_tensor(emitted)
      fix_pos = torch.arange(cache.seq_len, cache.seq_len + len(emitted), dtype=torch.long, device=device)
      logits, hidden = model(fix_pos, fix_ids, cache, return_hidden=True)
      gen_ids.extend(emitted)
      last_logits = logits[-1].clone()
      last_hidden = hidden[-1].clone()

    if gen_ids[-1] == eos_token_id:
      break

  return gen_ids
