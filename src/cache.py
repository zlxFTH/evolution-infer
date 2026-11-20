import torch

from constant import *

class Cache():
    def __init__(self, max_len: int = 4096, dtype=torch.bfloat16, device: str = "cuda"):
        self.seq_len = 0
        self.k_cache = [None] * num_hidden_layers
        self.v_cache = [None] * num_hidden_layers
        self.conv_state = [None] * num_hidden_layers # qkv
        self.recur_state = [None] * num_hidden_layers # state

        conv_dim = linear_num_kq_heads * linear_head_dim * 2 + linear_num_v_heads * linear_head_dim
        for i in range(num_hidden_layers):
            if i % block == 3:
                self.k_cache[i] = torch.zeros(max_len, num_kv_heads, head_dim, dtype=dtype, device=device)
                self.v_cache[i] = torch.zeros(max_len, num_kv_heads, head_dim, dtype=dtype, device=device)
            else:
                self.conv_state[i] = torch.zeros()
                self.recur_state[i] = 