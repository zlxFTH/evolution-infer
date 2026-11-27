import torch
from torch import nn
import torch.nn.functional as F

from qwen import RMSNorm

HIDDEN_LAYERS = (3, 19, 47, 63)
BLOCK_SIZE = 32


class GatedMLP(nn.Module):
  def __init__(self, in_dim, hidden_dim, out_dim=1):
    super().__init__()
    self.norm = RMSNorm(in_dim)
    self.gate = nn.Linear(in_dim, hidden_dim, bias=False)
    self.up = nn.Linear(in_dim, hidden_dim, bias=False)
    self.down = nn.Linear(hidden_dim, out_dim, bias=False)

  def forward(self, x):
    z = self.norm(x)
    return self.down(F.silu(self.gate(z)) * self.up(z))


def summarize_blocks(h):
  pad = (-len(h)) % BLOCK_SIZE
  h_blocks = F.pad(h, (0, 0, 0, pad)).reshape(-1, BLOCK_SIZE, h.shape[-1])
  # 尾块在打分时按每个锚点的可见范围重新聚合。
  return h_blocks.sum(1), h_blocks.amax(1)


class CopyLocator(nn.Module):
  def __init__(self, d_model=5120, d=512, d_span=256, min_copy=5):
    super().__init__()
    self.min_copy = min_copy
    self.proj_norms = nn.ModuleList(RMSNorm(d_model) for _ in HIDDEN_LAYERS)
    self.projs = nn.ModuleList(nn.Linear(d_model, d, bias=False) for _ in HIDDEN_LAYERS)
    self.seg_emb = nn.Embedding(3, d)
    self.out_norm = RMSNorm(d)
    self.register_buffer("offsets", torch.arange(BLOCK_SIZE), persistent=False)

    self.mlp_start = GatedMLP(2 * d + 1, 512)
    self.mlp_no_copy = GatedMLP(d + 2, 256)
    self.mlp_span = GatedMLP(d, 512, d_span)
    self.mlp_u = GatedMLP(2 * d, 512, d_span)
    self.phi_same = nn.Parameter(torch.zeros(2 * d_span))
    self.h_eos = nn.Parameter(torch.zeros(d_span))
    self.mlp_r_chunk = GatedMLP(7 * d_span, 512)
    self.mlp_r_tok = GatedMLP(5 * d_span, 512)

  def encode_step(self, hidden, seg_ids):
    x = sum(proj(norm(h.to(proj.weight.dtype))) for proj, norm, h in zip(self.projs, self.proj_norms, hidden))
    return self.out_norm(x + self.seg_emb(seg_ids))

  def _slice(self, features, start, stop):
    positions = start[:, None] + self.offsets
    return features[positions.minimum(stop[:, None] - 1)], positions < stop[:, None]

  def _boundary(self, h, positions, visible):
    values = h[positions.minimum(visible[:, None] - 1)]
    return torch.where((positions < visible[:, None])[..., None], values, self.h_eos)

  @staticmethod
  def _replace_block(blocks, indices, values):
    return blocks.scatter(1, indices[:, None, None].expand(-1, 1, values.shape[-1]), values[:, None])

  def start_logits(self, x, visible, positions, overlaps):
    q = x[visible - 1]
    valid = (positions >= 0) & (positions <= visible[:, None] - self.min_copy)
    values = x[positions.clamp_min(0).minimum(visible[:, None] - 1)]
    suffix = overlaps.to(x.dtype).log1p()[..., None]
    inputs = torch.cat((q[:, None].expand_as(values), values, suffix), dim=-1)
    scores = self.mlp_start(inputs).squeeze(-1).masked_fill(~valid, -torch.inf)
    best = scores.amax(1).masked_fill(~valid.any(1), 0)
    average = scores.masked_fill(~valid, 0).sum(1) / valid.sum(1).clamp_min(1)
    no_copy = self.mlp_no_copy(torch.cat((q, best[:, None], average[:, None]), dim=-1))
    return torch.cat((no_copy, scores), dim=1)

  def end_blocks(self, x, h, visible, left, h_sums, h_maxes):
    u = self.mlp_u(torch.cat((x[visible - 1], x[left]), dim=-1))
    starts = torch.arange(len(h_sums), device=x.device) * BLOCK_SIZE
    lo = torch.maximum(left[:, None], starts)
    hi = torch.minimum(visible[:, None], starts + BLOCK_SIZE)
    counts = (hi - lo).clamp_min(0)
    start_block, tail = left // BLOCK_SIZE, (visible - 1) // BLOCK_SIZE

    sums = h_sums[None].expand(len(visible), -1, -1)
    maxes = h_maxes[None].expand(len(visible), -1, -1)
    # 首块和尾块会被 l、T 截断。同块时两次得到同一个切片。
    batch = torch.arange(len(visible), device=x.device)
    for block in (start_block, tail):
      values, mask = self._slice(h, lo[batch, block], hi[batch, block])
      sums = self._replace_block(sums, block, values.masked_fill(~mask[..., None], 0).sum(1))
      maxes = self._replace_block(maxes, block, values.masked_fill(~mask[..., None], -torch.inf).amax(1))

    empty = (counts == 0)[..., None]
    sums = sums.masked_fill(empty, 0)
    maxes = maxes.masked_fill(empty, -torch.inf)
    before_n = F.pad(counts.cumsum(1)[:, :-1], (1, 0))
    before_sum = F.pad(sums.cumsum(1)[:, :-1], (0, 0, 1, 0))
    before_max = F.pad(maxes.cummax(1).values[:, :-1], (0, 0, 1, 0), value=-torch.inf)
    before = torch.cat((before_sum / before_n.clamp_min(1)[..., None], before_max), dim=-1)
    before = torch.where((before_n == 0)[..., None], self.phi_same, before)
    block_feat = torch.cat((sums / counts.clamp_min(1)[..., None], maxes.masked_fill(empty, 0)), dim=-1)
    left_edge = h[lo.minimum(visible[:, None] - 1)]
    right_edge = self._boundary(h, hi, visible)
    features = torch.cat((u[:, None].expand(-1, len(starts), -1), before, block_feat, left_edge, right_edge), dim=-1)
    valid = (counts > 0) & (hi >= left[:, None] + self.min_copy)
    scores = self.mlp_r_chunk(features).squeeze(-1).masked_fill(~valid, -torch.inf)
    return scores, (u, before_n, before_sum, before_max)

  def end_tokens(self, h, visible, left, blocks, prefix):
    u, before_n, before_sum, before_max = prefix
    batch = torch.arange(len(visible), device=h.device)
    positions = blocks[:, None] * BLOCK_SIZE + self.offsets
    values = h[positions.minimum(visible[:, None] - 1)]
    inside = (positions >= left[:, None]) & (positions < visible[:, None])
    n = before_n[batch, blocks, None] + inside.cumsum(1)
    sums = before_sum[batch, blocks, None] + values.masked_fill(~inside[..., None], 0).cumsum(1)
    maxes = values.masked_fill(~inside[..., None], -torch.inf).cummax(1).values
    maxes = torch.maximum(before_max[batch, blocks, None], maxes)
    span = torch.cat((sums / n.clamp_min(1)[..., None], maxes.masked_fill((n == 0)[..., None], 0)), dim=-1)
    features = torch.cat((u[:, None].expand_as(values), span, values, self._boundary(h, positions + 1, visible)), dim=-1)
    scores = self.mlp_r_tok(features).squeeze(-1)
    valid = inside & (positions >= left[:, None] + self.min_copy - 1)
    return scores.masked_fill(~valid, -torch.inf)

  @torch.no_grad()
  def locate(self, x, candidates, h=None, summaries=None, *, min_start_prob=0.6):
    if not candidates:
      return None
    visible = x.new_tensor([len(x)], dtype=torch.long)
    positions, overlaps = x.new_tensor(candidates, dtype=torch.long).T
    logits = self.start_logits(x, visible, positions[None], overlaps[None])
    probability, choice = logits[0].softmax(0).max(0)
    pick = choice.masked_fill(probability < min_start_prob, 0).item()
    if pick == 0:
      return None
    left = positions[pick - 1 : pick]
    if h is None:
      h = self.mlp_span(x)
    h_sums, h_maxes = summarize_blocks(h) if summaries is None else summaries
    scores, prefix = self.end_blocks(x, h, visible, left, h_sums, h_maxes)
    end_block = scores.argmax(1)
    last = BLOCK_SIZE * end_block + self.end_tokens(h, visible, left, end_block, prefix).argmax(1)
    return candidates[pick - 1][0], last.item() + 1
