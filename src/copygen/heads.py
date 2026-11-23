import torch
import torch.nn as nn
import torch.nn.functional as F

class CopyGenHeads(nn.Module):
  def __init__(self, hidden_size: int = 5120, mid_dim: int = 512):
    super().__init__()

    in_dim = hidden_size + 1 # 多一维放已匹配长度 log 特征
    self.ln = nn.LayerNorm(in_dim)
    self.trigger_head = nn.Linear(in_dim, 2) # 决定是否进入 copy，[GEN 得分, COPY 得分]
    self.length_head = nn.Sequential(
      nn.Linear(in_dim, mid_dim),
      nn.SiLU(),
      nn.Linear(mid_dim, 1),
    ) # 长度预测，单层线性加权学不出来
    self.calib_head = nn.Sequential(
      nn.Linear(in_dim, mid_dim),
      nn.SiLU(),
      nn.Linear(mid_dim, 1),
    ) # 仅在 calibrate=True（论文原版模式）时开启，微调 logits

  def features(
    self,
    hidden_states: torch.Tensor,
    match_lengths: int | torch.Tensor,
  ) -> torch.Tensor:
    # hidden 和 log(1 + m_t) 拼起来，多 hidden 形状支持
    hidden_states = hidden_states.to(dtype=self.ln.weight.dtype)
    # as_tensor 统一转成 torch.Tensor
    lengths = torch.as_tensor(match_lengths, device=hidden_states.device, dtype=torch.float32)
    log_matches = lengths.log1p().expand(hidden_states.shape[:-1])
    log_matches = log_matches.to(hidden_states.dtype).unsqueeze(-1)
    return self.ln(torch.cat((hidden_states, log_matches), dim=-1))

  def log_length(self, features: torch.Tensor) -> torch.Tensor:
    return F.softplus(self.length_head(features).float())

  def forward(
    self,
    hidden_states: torch.Tensor,
    match_lengths: int | torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features = self.features(hidden_states, match_lengths)
    return self.trigger_head(features), self.log_length(features), self.calib_head(features)
