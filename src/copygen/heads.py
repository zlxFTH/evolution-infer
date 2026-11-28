import torch
from torch import nn
import torch.nn.functional as F

from constant import *

class CopyGenHeads(nn.Module):
  def __init__(self, dim=hidden_size, mid_dim=512):
    super().__init__()
    in_dim = dim + 1
    self.ln = nn.LayerNorm(in_dim)
    self.trigger_head = nn.Linear(in_dim, 2)
    self.length_head = nn.Sequential(
      nn.Linear(in_dim, mid_dim),
      nn.SiLU(),
      nn.Linear(mid_dim, 1),
    )

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
