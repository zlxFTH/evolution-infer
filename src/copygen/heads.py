import torch
from torch import nn
import torch.nn.functional as F

from constant import *

def mlp(in_dim, mid_dim, out_dim, dropout):
  return nn.Sequential(
    nn.Linear(in_dim, mid_dim),
    nn.GELU(),
    nn.Dropout(dropout),
    nn.Linear(mid_dim, out_dim),
  )

class CopyGenHeads(nn.Module):
  def __init__(self, dim=hidden_size, dropout=0.1):
    super().__init__()
    in_dim = dim + 1
    self.ln = nn.LayerNorm(in_dim)
    self.trigger_head = mlp(in_dim, dim, 2, dropout)
    self.length_head = mlp(in_dim, dim, 1, dropout)

  def features(self, hidden, match_len):
    hidden = hidden.to(self.ln.weight.dtype)
    m = torch.as_tensor(match_len, device=hidden.device, dtype=torch.float32).log1p()
    m = m.expand(hidden.shape[:-1]).unsqueeze(-1).to(hidden.dtype)
    return self.ln(torch.cat((hidden, m), dim=-1))

  def log_length(self, feat):
    return F.softplus(self.length_head(feat).float())

  def forward(self, hidden, match_len):
    feat = self.features(hidden, match_len)
    return self.trigger_head(feat), self.log_length(feat)
