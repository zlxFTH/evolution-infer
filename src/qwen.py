import torch
from torch import nn
import torch.nn.functional as F

from constant import *

class RMSNorm(nn.Module):
    def __init__(self, dim: int = hidden_size):
        super().__init__()

        self.eps = rms_norm_eps
        self.weight = nn.Parameter(torch.zeros(dim)) # 支持多维度

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xx = x.float()
        rms = torch.rsqrt(xx.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (xx * rms * (1.0 + self.weight.float())).to(x.dtype)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: down(silu(gate(x)) * up(x))
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

class RoPE(nn.Module): 
    def __init__(self):
        super().__init__()

        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
        t = torch.arange(max_positions, dtype=torch.float32)
        freq = torch.outer(t, inv_freq)
        emb = torch.cat((freq, freq), dim=-1)

        # persistent=False 不存进权重文件
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    # positions: [seq_len]
    # q: [seq_len, num_heads, head_dim]
    # k: [seq_len, num_kv_heads, head_dim]
    def forward(self, positions: torch.Tensor, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cos = self.cos_cached[positions].unsqueeze(1).to(q.dtype)
        sin = self.sin_cached[positions].unsqueeze(1).to(q.dtype)

        q_rot, q_pass = q[..., : rotary_dim], q[..., rotary_dim :]
        k_rot, k_pass = k[..., : rotary_dim], k[..., rotary_dim :]

        q_rot = q_rot * cos + rotate_half(q_rot) * sin
        k_rot = k_rot * cos + rotate_half(k_rot) * sin

        return torch.cat((q_rot, q_pass), dim=-1), torch.cat((k_rot, k_pass), dim=-1)

class Attention(nn.Module):
    def __init__(self):
        super().__init__()

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim * 2, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)
        self.rope = RoPE()

    def forward(self, positions: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[0]

        q, gate = torch.chunk(self.q_proj(x).view(seq_len, num_heads, head_dim * 2), 2, dim=-1)
        gate = gate.reshape(seq_len, -1)
        k = self.k_proj(x).view(seq_len, num_kv_heads, head_dim)
        v = self.v_proj(x).view(seq_len, num_kv_heads, head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = self.rope(positions, q, k)

        q = torch.transpose(q, 0, 1).unsqueeze(0)
        k = torch.transpose(k, 0, 1).unsqueeze(0)
        v = torch.transpose(v, 0, 1).unsqueeze(0)

        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            enable_gqa=True
        )

        out = out.squeeze(0).transpose(0, 1).reshape(seq_len, -1)
        out = out * torch.sigmoid(gate)
        return self.o_proj(out)

class RMSNormGated(nn.Module):
    def __init__(self, dim: int):
        super().__init__()

        self.eps = rms_norm_eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        xx = x.float()
        rms = torch.rsqrt(xx.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        out = (xx * rms).to(x.dtype) * self.weight
        return (out * F.silu(gate.float())).to(x.dtype)

class LinearAttention(nn.Module):
    def __init__(self):
        super().__init__()

        self.qk_dim = linear_num_kq_heads * linear_head_dim
        self.v_dim = linear_num_v_heads * linear_head_dim
        self.conv_dim = self.qk_dim * 2 + self.v_dim

        self.in_proj_qkv = nn.Linear(hidden_size, self.conv_dim, bias=False)
        self.in_proj_a = nn.Linear(hidden_size, linear_num_v_heads, bias=False)
        self.in_proj_b = nn.Linear(hidden_size, linear_num_v_heads, bias=False)
        self.in_proj_z = nn.Linear(hidden_size, self.v_dim, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            kernel_size=linear_num_kernel,
            groups=self.conv_dim,
            padding=linear_num_kernel-1,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[0]

        qkv = self.in_proj_qkv(x)
        z = self.in_proj_z(x).view(seq_len, linear_num_v_heads, linear_head_dim)
        a = self.in_proj_a(x)
        b = self.in_proj_b(x)

        qkv = F.silu(self.conv1d(qkv.transpose(0, 1))[:, : seq_len]).transpose(0, 1)

        q, k, v = torch.split(qkv, [self.qk_dim, self.qk_dim, self.v_dim], dim=-1)
        q = q.reshape(seq_len, linear_num_kq_heads, -1)
        k = k.reshape(seq_len, linear_num_kq_heads, -1)
        v = v.reshape(seq_len, linear_num_v_heads, -1)
