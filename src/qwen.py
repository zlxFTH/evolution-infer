import torch
from torch import nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xx = x.float()
        rms = torch.rsqrt(xx.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (xx * rms).to(x.dtype) * self.weight

class MLP(nn.Module):
    def __init__(self, hidden_size: int = 5120, intermediate_size: int = 17408):
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
    def __init__(self, rotary_dim: int = 64, max_positions: int = 262144, base: float = 10000000.0):
        super().__init__()

        self.rotary_dim = rotary_dim
        inv_freq = 1.0 / (base ** torch.arrange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
        t = torch.arrange(max_positions, dtype=torch.float32)
        freq = torch.outer(t, inv_freq)
        emb = torch.cat((freq, freq), dim=-1)

        # persistent=False 不存进权重文件
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("cos_cached", emb.sin(), persistent=False)

    # positions: [seq_len]
    # q: [seq_len, num_heads, head_dim]
    # k: [seq_len, num_kv_heads, head_dim]
    def forward(self, positions: torch.Tensor, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cos = self.cos_cached[positions].unsqueeze(1).to(q.dtype)
        sin = self.sin_cached[positions].unsqueeze(1).to(q.dtype)

        q_rot, q_pass = q[..., : self.rotary_dim], q[..., self.rotary_dim :]
        k_rot, k_pass = k[..., : self.rotary_dim], k[..., self.rotary_dim :]

        q_rot = q_rot * cos + rotate_half(q_rot) * sin
        k_rot = k_rot * cos + rotate_half(k_rot) * sin

        return torch.cat((q_rot, q_pass), dim=-1), torch.cat((k_rot, k_pass), dim=-1)