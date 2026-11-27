import torch
import torch.nn.functional as F


def chunk_gated_delta(q, k, v, g, beta, state, chunk_size=64):
  """Gated delta 递推，块内用三角求解，块间传递 recurrent state。"""
  length = len(q)
  width = min(chunk_size, length)
  pad = (-length) % width
  q, k, v = (
    F.pad(x.transpose(0, 1), (0, 0, 0, pad)).unflatten(1, (-1, width))
    for x in (q, k, v)
  )
  g, beta = (
    F.pad(x.T, (0, pad)).unflatten(1, (-1, width))
    for x in (g, beta)
  )

  cumulative = g.cumsum(-1)
  future = torch.ones(width, width, dtype=torch.bool, device=q.device).triu(1)
  decay = (cumulative[..., None] - cumulative.unsqueeze(-2)).masked_fill(future, -torch.inf).exp()
  from_start = cumulative.exp()[..., None]
  beta_k = k * beta[..., None]
  system = (beta_k @ k.transpose(-1, -2)) * decay
  updates = torch.linalg.solve_triangular(system, v * beta[..., None], upper=False, unitriangular=True)
  state_keys = torch.linalg.solve_triangular(system, beta_k * from_start, upper=False, unitriangular=True)
  within = (q @ k.transpose(-1, -2)) * decay
  q = q * from_start
  carry = from_start[:, :, -1:, :]
  k = k * (cumulative[..., -1:] - cumulative).exp()[..., None]

  outputs = []
  for i in range(q.shape[1]):
    delta = updates[:, i] - state_keys[:, i] @ state
    outputs.append(q[:, i] @ state + within[:, i] @ delta)
    state = carry[:, i] * state + k[:, i].transpose(-1, -2) @ delta
  return torch.cat(outputs, dim=1).transpose(0, 1)[:length], state
