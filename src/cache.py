import torch

from constant import *

class Cache:
  def __init__(self, max_len):
    self.seq_len, self.is_prefill = 0, True

    def alloc(full, shape, dtype=torch.bfloat16):
      ids = [i for i in range(num_hidden_layers) if (i % block == 3) == full]
      buf = torch.zeros((len(ids), *shape), dtype=dtype)
      out = [None] * num_hidden_layers
      for j in range(len(ids)):
        out[ids[j]] = buf[j]
      return buf, out

    with torch.device("cuda"):
      _, self.k_cache                  = alloc(True,  (max_len, *kv_shape))
      _, self.v_cache                  = alloc(True,  (max_len, *kv_shape))
      self.conv_buf,  self.conv_state  = alloc(False, conv_shape)
      self.recur_buf, self.recur_state = alloc(False, recur_shape, torch.float32)

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
    return self.seq_len, self.is_prefill, self.conv_buf.clone(), self.recur_buf.clone()

  def restore(self, snapshot):
    self.seq_len, self.is_prefill, conv, recur = snapshot
    self.conv_buf.copy_(conv)
    self.recur_buf.copy_(recur)

  def reset(self):
    self.seq_len, self.is_prefill = 0, True
    self.conv_buf.zero_()
    self.recur_buf.zero_()
