import torch

from constant import *

class Cache():
  def __init__(self, max_len: int = 4096, dtype=torch.bfloat16, device: str = "cuda"):
    self.seq_len     = 0
    self.k_cache     = [None] * num_hidden_layers
    self.v_cache     = [None] * num_hidden_layers
    self.conv_state  = [None] * num_hidden_layers  # qkv
    self.recur_state = [None] * num_hidden_layers  # state
    self.is_prefill  = True

    conv_dim = linear_num_kq_heads * linear_head_dim * 2 + linear_num_v_heads * linear_head_dim
    with torch.device(device):
      for i in range(num_hidden_layers):
        if i % block == 3:
          self.k_cache[i] = torch.zeros(max_len, num_kv_heads, head_dim, dtype=dtype)
          self.v_cache[i] = torch.zeros(max_len, num_kv_heads, head_dim, dtype=dtype)
        else:
          self.conv_state[i]  = torch.zeros(conv_dim, linear_num_kernel - 1, dtype=dtype)
          self.recur_state[i] = torch.zeros(linear_num_v_heads, linear_head_dim, linear_head_dim,
                                            dtype=torch.float32)

  def upd_kv_cache(self, i: int, k: torch.Tensor, v: torch.Tensor):
    beg = self.seq_len
    end = beg + k.shape[0]
    self.k_cache[i][beg : end] = k
    self.v_cache[i][beg : end] = v
    return self.k_cache[i][: end], self.v_cache[i][: end]

  def upd_conv_cache(self, i: int, qkv: torch.Tensor):
    k_len = linear_num_kernel - 1
    window = torch.cat([self.conv_state[i], qkv], dim=1)
    self.conv_state[i].copy_(window[:, -k_len :])
    return window # decode 的时候可以直接拿去卷积

  def upd_recur_cache(self, i: int, state: torch.Tensor):
    self.recur_state[i].copy_(state)

  def reset(self):
    self.seq_len    = 0
    self.is_prefill = True
    for u in self.conv_state:
      if u is not None:
        u.zero_()
    for u in self.recur_state:
      if u is not None:
        u.zero_()
    