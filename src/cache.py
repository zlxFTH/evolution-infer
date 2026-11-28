import torch

from constant import *

class Cache:
  def __init__(self, max_len=max_positions):
    self.seq_len, self.is_prefill = 0, True

    def alloc(full, shape, dtype=torch.bfloat16):
      return [
        torch.zeros(shape, dtype=dtype) if (i % block == 3) == full else None
        for i in range(num_hidden_layers)
      ]

    with torch.device("cuda"):
      self.k_cache     = alloc(True,  (max_len, *kv_shape))
      self.v_cache     = alloc(True,  (max_len, *kv_shape))
      self.conv_state  = alloc(False, conv_shape)
      self.recur_state = alloc(False, recur_shape, torch.float32)

  def upd_kv_cache(self, i, k, v):
    beg = self.seq_len
    end = beg + k.shape[0]
    self.k_cache[i][beg : end] = k
    self.v_cache[i][beg : end] = v

  def upd_conv_cache(self, i, qkv):
    k_len = linear_num_kernel - 1
    self.conv_state[i].copy_(torch.cat([self.conv_state[i], qkv], dim=1)[:, -k_len :])

  def upd_recur_cache(self, i, state):
    self.recur_state[i].copy_(state)

  def snapshot(self):
    # KV 只追加，回滚时截短 seq_len 即可。
    conv = [s.clone() if s is not None else None for s in self.conv_state]
    recur = [s.clone() if s is not None else None for s in self.recur_state]
    return self.seq_len, self.is_prefill, conv, recur

  def restore(self, snapshot):
    self.seq_len, self.is_prefill, conv, recur = snapshot
    for i, state in enumerate(self.conv_state):
      if state is not None:
        state.copy_(conv[i])
        self.recur_state[i].copy_(recur[i])

  def reset(self):
    self.seq_len    = 0
    self.is_prefill = True
    for u in self.conv_state:
      if u is not None:
        u.zero_()
    for u in self.recur_state:
      if u is not None:
        u.zero_()
