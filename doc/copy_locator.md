# Copy Locator 设计说明

> 对齐数据格式：`doc/1.md`（上下文按 `prompt -> cot -> output` 拼接，单调队列 DP 切分极长不交三元组 $(i, l, r)$，强制最小复制长度 $r - l \ge k$，默认 $k = 5$）

## 核心思路

当前上下文共有 $T$ 个 token，大模型刚算完 $T - 1$，其 `lm_head` 已免费给出位置 $T$ 的目标采样分布，可直接按当前解码策略（温度采样或贪心解码）采样确定位置 $T$ 的首词 $t_0$。

Copy Locator 在这一刻介入判断：接下来从位置 $T$ 开始，能否直接从 $[0, T)$ 中整段复制某个区间 $[l, r)$（下文为了方便，改为左闭右闭区间，$[l, e = r - 1]$），满足 $0 \le l \le T - k$ 且 $l + k - 1 \le e \le T - 1$。

整体按“先找起点 $l$，再找终点 $e$”分成三步：

1. **选起点 $l$（或判 `NO_COPY`）**：先用首词 $t_0$ 在上文筛出 `tokens[l] == t_0` 的候选位置，再结合查询特征 $q$、起点正向特征 $X_l$、起点逆向特征 $Y_l$ 和左侧重合长度打分。
2. **粗选终点块 $m_r$**：把上下文按 $32$ 个 token 分块，定位终点 $e$ 所在的块 $m_r$。
3. **块内选末尾词 $e$**：在块 $m_r$ 的 $32$ 个位置里选出精确终点 $e$，输出 $[l,\, e + 1)$。

```text
27B 四层 hidden_state (Layer 3, 19, 47, 63) + 采样确定的首词 t_0
        │
        ├──► Q_n 投影位置 T-1 ──► 查询特征 q [512]
        │
        └──► P_n 投影 [0, T)  ──► 正向特征 X [T, 512]
                                      │
                  ┌───────────────────┴───────────────────┐
                  ▼                                       ▼
        逆向特征 Y_l (右看 W = 256)              区间词特征 H [T, 256]
                  │                                       │
                  ▼                                       ├──► ② 粗选终点块 m_r (按 32 词分块)
        ① 候选池选起点 l 或 NO_COPY                        └──► ③ 块 m_r 内选末尾词 e ──► [l, e + 1)
```

模型里所有特征融合和打分都统一用 `Qwen3.8` 同结构的门控网络 $\mathrm{GatedMLP}$（`RMSNorm + SwiGLU`），不做裸点乘：

$$
\mathrm{GatedMLP}(z) = W_{\text{down}}\Big(\mathrm{SiLU}\big(W_{\text{gate}}\,\mathrm{RMSNorm}(z)\big) \odot \big(W_{\text{up}}\,\mathrm{RMSNorm}(z)\big)\Big)
$$

---

## 特征

### 前向特征和语义查询特征

从冻结的 `Qwen3.8-27B`（共 $64$ 层，hidden_state 维度 $d_{\text{model}} = 5120$）抽取第 $3, 19, 47, 63$ 层 hidden_state $h^{(n)}_j \in \mathbb{R}^{5120}$。

对每个可见位置 $j \in \{0, \dots, T - 1\}$，四层特征经 Norm 后通过投影矩阵 $P_n \in \mathbb{R}^{512 \times 5120}$ 降维相加，叠加上下文阶段信息 $E_{\text{seg}}(s_j)$（$s_j \in \{0\text{: prompt},\, 1\text{: cot},\, 2\text{: output}\}$），再经 Norm 得到正向词特征 $X_j$。

同时，对当前生成位置 $T - 1$，生成一个查询特征 $q$。$q$ 不使用 $P_n$，而是用专门的查询投影矩阵 $Q_n \in \mathbb{R}^{512 \times 5120}$ 独立提取查询语义得到 $q$：

$$
X_j = \mathrm{RMSNorm}_x\!\left(\sum_{n \in \{3, 19, 47, 63\}} P_n \, \mathrm{RMSNorm}_n\!\big(h^{(n)}_j\big) + E_{\text{seg}}(s_j)\right) \in \mathbb{R}^{512}, \qquad 0 \le j < T
$$

$$
q = \mathrm{RMSNorm}_q\!\left(\sum_{n \in \{3, 19, 47, 63\}} Q_n \, \mathrm{RMSNorm}_n\!\big(h^{(n)}_{T-1}\big) + E_{q,\text{seg}}(s_{T-1})\right) \in \mathbb{R}^{512}
$$

- $X_l$ 专注于提供 $l$ 自己和左侧序列的语义信息。
- $q$ 专注于提供当前位置的检索意图和接下来准备生成的内容规划。

### 逆向特征

大模型是因果注意力，$X_l$ 看不见 $l$ 后面的 token。当上文有两个候选位置开头一大串相同、到后面才出现差异时，单看 $X_l$ 无法区分。虽然 $q$ 涵盖了整个上下文信息，维度有限，设计时希望 $q$ 尽可能专注于查询和生成意图，长上下文下无法靠它完成区分。

为此给候选位置 $l$ 增加一个可以向右看的窗口 $W = 256$ 的逆向特征 $Y_l$，并且和 $X_l$ 分离、不叠加混用。

为了保留后续 token 的顺序信息并支持 $O(1)$ 区间求和，这里不加衰减系数，直接在特征上融合 RoPE 维护前缀和。

将 $X_j$ 线性投影为 $V_j = W_{\text{rev}}\, X_j \in \mathbb{R}^{256}$，按相邻两维配成 $128$ 个二维旋转平面，频率取 $\theta_c = 10000^{-2c / 256}\ (c = 0, \dots, 127)$。位置 $t$ 的旋转算子 $\mathrm{RoPE}(v, t)$ 为：

$$
\begin{bmatrix} \mathrm{RoPE}(v, t)_{2c} \\ \mathrm{RoPE}(v, t)_{2c+1} \end{bmatrix} = \begin{bmatrix} \cos(t\theta_c) & -\sin(t\theta_c) \\ \sin(t\theta_c) & \cos(t\theta_c) \end{bmatrix} \begin{bmatrix} v_{2c} \\ v_{2c+1} \end{bmatrix}
$$

顺着生成方向维护 `float32` 前缀和 $F$：

$$
F(0) = 0 \in \mathbb{R}^{256}, \qquad F(t + 1) = F(t) + \mathrm{RoPE}(V_t,\, t), \qquad 0 \le t < T
$$

查询候选位置 $l$ 时，取可见窗口右端点 $b_l = \min(l + W,\, T)$（$W = 256$），做一次前缀和相减，再逆旋转 $-l$ 把起点相对位置归零：

$$
Y_l = \frac{1}{\sqrt{b_l - l}}\,\mathrm{RoPE}\!\Big(F(b_l) - F(l),\ -l\Big) = \frac{1}{\sqrt{b_l - l}}\sum_{j = l}^{b_l - 1} \mathrm{RoPE}\!\big(V_j,\ j - l\big) \in \mathbb{R}^{256}
$$

- 旋转不改变模长，前缀和在 `float32` 下不会下溢或丢精度。
- 窗口定为 $W = 256$（$= d / 2$）：逆向特征是 $256$ 维，叠加 $256$ 个 token 平均每词分到 $1$ 维，既能跨过常见的相同开头看清后面的分叉，又不会像 $W = 512$ 那样混入过多下文噪声；

> Q：如果构造两个位置 $l_1, l_2$ 往后 $W$ 个 token 都相同，实际希望复制 $l_2$ 但复制了 $l_1$？ 

其实无所谓，选错了也能至少吃到一次并行验证 $W = 256$ 个 token 的加速，已经非常可观了。

### 区间特征 $H_j$ 与 $\phi(a, b)$

定终点 $e$ 时，需要判断区间 $[l, e]$ 内部有没有跨过断点，以及下一个词 $e + 1$ 该不该停手。先将每个 $X_j$ 映射成 $256$ 维的区间词特征 $H_j$：

$$
H_j = \mathrm{GatedMLP}_{\text{span}}(X_j) \in \mathbb{R}^{256}, \qquad 0 \le j < T
$$

当右边界刚好落到可见末尾 $T$ 时，$H_T$ 还不可见，因此设一个可学习的末尾占位向量 $h_{\text{eos}} \in \mathbb{R}^{256}$，记扩展序列 $\tilde{H}_j = H_j\ (j < T)$，$\tilde{H}_T = h_{\text{eos}}$。

对任意闭区间 $[a, b]$（$0 \le a \le b < T$），把区间均值和逐通道最大值拼在一起作为区间特征 $\phi(a, b)$：

$$
\mu(a, b) = \frac{1}{b - a + 1}\sum_{j = a}^{b} H_j \in \mathbb{R}^{256}, \qquad \omega(a, b) = \max_{a \le j \le b} H_j \in \mathbb{R}^{256}, \qquad \phi(a, b) = \big[\mu(a, b),\, \omega(a, b)\big] \in \mathbb{R}^{512}
$$

- $\mu(a, b)$ 表示整段平均语义；$\omega(a, b)$ 保留区间内的极值信号，防止长区间求平均把单个换行、引号等断点冲淡。
- $\phi(a, b)$ 只依赖 $[a, b]$ 内部 token，可用前缀和 `cumsum` 与前缀最大值 `cummax` 快速算出。

## 流程

打分全程使用 `float32`。由最小复制长度 $k = 5$，合法下标需满足 $0 \le l \le T - k$ 且 $l + k - 1 \le e \le T - 1$。

### 2.1 选起点 $l$ 或 `NO_COPY`

用已采样出的首词 $t_0$，通过 `StartIndex` 在上文找出所有满足 $\text{tokens}[l] = t_0$ 且 $0 \le l \le T - k$ 的位置（共 $C$ 个），同时计算每个候选左侧与当前位置 $T$ 左侧的最长公共匹配长度 $\text{ov}_l$ 作为参考特征：

$$
\mathcal{C} = \big\{(l_1, \text{ov}_1),\, \dots,\, (l_C, \text{ov}_C)\big\}
$$

若 $T < k$ 或 $C = 0$，直接返回 `NO_COPY`。

当 $C \ge 1$ 时，对每个候选 $l_j$ 拼接 $[q,\, X_{l_j},\, Y_{l_j},\, \ln(1 + \text{ov}_j)]$（共 $512 + 512 + 256 + 1 = 1281$ 维）打分：

$$
s_j = \mathrm{GatedMLP}_{\text{start}}\!\Big(\big[q,\, X_{l_j},\, Y_{l_j},\, \ln(1 + \text{ov}_j)\big]\Big) \in \mathbb{R}^1, \qquad j = 1, \dots, C
$$

再取候选最高分 $s_{\max} = \max_j s_j$ 和均分 $s_{\text{mean}} = \frac{1}{C}\sum_j s_j$，与 $q$ 拼成 $514$ 维算不复制分数 $s_{\text{no}}$：

$$
s_{\text{no}} = \mathrm{GatedMLP}_{\text{no}}\!\big([q,\, s_{\max},\, s_{\text{mean}}]\big), \qquad z^{(1)} = [s_{\text{no}},\, s_1,\, \dots,\, s_C] \in \mathbb{R}^{C+1}
$$

记 $p^{(1)} = \mathrm{softmax}\big(z^{(1)}\big)$，$\hat{c} = \arg\max z^{(1)}$：
- 若 $\hat{c} = 0$ 或 $p^{(1)}_{\hat{c}} < \tau_{\text{start}}$（默认 $\tau_{\text{start}} = 0.6$），返回 `NO_COPY`；
- 否则选中起点 $l = l_{\hat{c}}$。

### 2.2 粗选终点块 $m_r$

按 $32$ 个 token 一块切分上下文，共 $M = \lceil T / 32 \rceil$ 块。起点所在块为 $m_l = \lfloor l / 32 \rfloor$，终点最小合法块为 $m_{\min} = \lfloor (l + k - 1) / 32 \rfloor$（允许 $m_r = m_l$ 同块）。

先融合查询特征 $q$ 与起点正向特征 $X_l$（不拼固定长度的 $Y_l$，避免短复制区间混入 $> e$ 的外部 token）：

$$
u = \mathrm{GatedMLP}_u\!\big([q,\, X_l]\big) \in \mathbb{R}^{256}
$$

对每个候选块 $m \in \{m_l, \dots, M - 1\}$，块内有效范围为 $a_m = \max(l,\, 32m)$、$b_m = \min(32(m + 1),\, T) - 1$，取三项特征：

- **前置区间特征 $\phi_{\text{pre}}(m) \in \mathbb{R}^{512}$**：$m = m_l$ 时填可学习向量 $\phi_{\text{same}} \in \mathbb{R}^{512}$；$m > m_l$ 时取 $\phi(l,\, 32m - 1)$。
- **本块区间特征 $\phi_{\text{blk}}(m) = \phi(a_m,\, b_m) \in \mathbb{R}^{512}$**：同块时从 $l$ 开始算，排除块内 $< l$ 的前缀。
- **块边界特征 $[H_{a_m},\, \tilde{H}_{b_m + 1}] \in \mathbb{R}^{512}$**：块内首词和块末下一词。

拼成 $1792$ 维打分（$m < m_{\min}$ 置为 $-\infty$），选出终点块 $m_r$：

$$
z^{(2)}_m = \begin{cases} \mathrm{GatedMLP}_{r,\text{chunk}}\!\Big(\big[u,\, \phi_{\text{pre}}(m),\, \phi_{\text{blk}}(m),\, H_{a_m},\, \tilde{H}_{b_m + 1}\big]\Big), & m \ge m_{\min} \\ -\infty, & m < m_{\min} \end{cases}, \qquad m_r = \arg\max_{0 \le m < M} z^{(2)}_m
$$

### 2.3 块 $m_r$ 内选末尾词 $e$

记块 $m_r$ 内有效范围为 $a_r = \max(l,\, 32 m_r)$、$b_r = \min(32(m_r + 1),\, T) - 1$。复用上一步的前置统计量，结合块内前缀 `cumsum` / `cummax`，并行算出块内每个位置 $e = 32 m_r + t$ 的区间特征 $\phi(l, e) \in \mathbb{R}^{512}$。

再拼上边界词 $[H_e,\, \tilde{H}_{e + 1}]$（最后一个复制词与紧随其后的第一个不复制词），共 $1280$ 维打分：

$$
z^{(3)}_t = \begin{cases} \mathrm{GatedMLP}_{r,\text{tok}}\!\Big(\big[u,\, \phi(l, e),\, H_e,\, \tilde{H}_{e + 1}\big]\Big), & l + k - 1 \le 32 m_r + t < T \\ -\infty, & \text{otherwise} \end{cases}
$$

$$
e = 32 m_r + \arg\max_{0 \le t < 32} z^{(3)}_t \quad \Longrightarrow \quad [l,\, r) = [l,\, e + 1)
$$

---

## 3. PyTorch 核心结构参考

总参数量约 $27.7\text{M}$（`bf16` 约 $53\text{ MB}$，其中 $P_n$ 和 $Q_n$ 各占 $10.5\text{M}$，其余打分头共 $6.7\text{M}$）：

```python
def apply_rope(x: torch.Tensor, pos: torch.Tensor, inv_freq: torch.Tensor):
  angles = pos.unsqueeze(-1).float() * inv_freq  # [..., d_rev // 2]
  cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
  x0, x1 = x[..., 0::2], x[..., 1::2]
  return torch.stack((x0 * cos - x1 * sin, x0 * sin + x1 * cos), dim=-1).flatten(-2)


class GatedMLP(nn.Module):
  def __init__(self, in_dim: int, hid_dim: int, out_dim: int = 1):
    super().__init__()
    self.norm = RMSNorm(in_dim)
    self.gate = nn.Linear(in_dim, hid_dim, bias=False)
    self.up   = nn.Linear(in_dim, hid_dim, bias=False)
    self.down = nn.Linear(hid_dim, out_dim, bias=False)

  def forward(self, x: torch.Tensor):
    z = self.norm(x)
    return self.down(F.silu(self.gate(z)) * self.up(z))


class CopyLocator(nn.Module):
  def __init__(
    self, d_model: int = 5120, d: int = 512, d_rev: int = 256,
    d_span: int = 256, rev_window: int = 256, min_copy: int = 5,
  ):
    super().__init__()
    self.min_copy   = min_copy
    self.rev_window = rev_window
    self.proj_norms = nn.ModuleList([RMSNorm(d_model) for _ in range(4)])
    self.projs      = nn.ModuleList([nn.Linear(d_model, d, bias=False) for _ in range(4)])   # P_n -> X_j
    self.seg_emb    = nn.Embedding(3, d)  # 0: prompt, 1: cot, 2: output
    self.out_norm   = RMSNorm(d)

    self.q_projs    = nn.ModuleList([nn.Linear(d_model, d, bias=False) for _ in range(4)])   # Q_n -> q
    self.q_seg_emb  = nn.Embedding(3, d)
    self.q_norm     = RMSNorm(d)

    inv_freq = 10000.0 ** (-torch.arange(0, d_rev, 2, dtype=torch.float32) / d_rev)
    self.register_buffer("inv_freq", inv_freq, persistent=False)
    self.w_rev       = nn.Linear(d, d_rev, bias=False)          # V_j = W_rev X_j [256]
    self.mlp_start   = GatedMLP(2 * d + d_rev + 1, 512, 1)      # [q, X_l, Y_l, ln(1+ov)] -> 1281
    self.mlp_no_copy = GatedMLP(d + 2, 256, 1)                  # [q, s_max, s_mean]      -> 514

    self.mlp_span    = GatedMLP(d, 512, d_span)
    self.mlp_u       = GatedMLP(2 * d, 512, d_span)             # [q, X_l]                -> 1024
    self.phi_same    = nn.Parameter(torch.zeros(2 * d_span))
    self.h_eos       = nn.Parameter(torch.zeros(d_span))

    self.mlp_r_chunk = GatedMLP(7 * d_span, 512, 1)             # 256+512+512+256+256     = 1792
    self.mlp_r_tok   = GatedMLP(5 * d_span, 512, 1)             # 256+512+256+256         = 1280

  def encode_step(self, hs_4: list[torch.Tensor], seg_ids: torch.Tensor):
    x = sum(proj(norm(h)) for norm, proj, h in zip(self.proj_norms, self.projs, hs_4))
    return self.out_norm(x + self.seg_emb(seg_ids))  # [seq_len, 512]

  def encode_query(self, hs_4_q: list[torch.Tensor], seg_ids_q: torch.Tensor):
    q = sum(q_proj(norm(h)) for norm, q_proj, h in zip(self.proj_norms, self.q_projs, hs_4_q))
    return self.q_norm(q + self.q_seg_emb(seg_ids_q))  # [..., 512]

  def build_rev_prefix(self, X: torch.Tensor):
    T = X.shape[0]
    V = self.w_rev(X).float()
    F_rev = torch.zeros(T + 1, V.shape[-1], dtype=torch.float32, device=X.device)
    F_rev[1:] = apply_rope(V, torch.arange(T, device=X.device), self.inv_freq).cumsum(dim=0)
    return F_rev  # [T + 1, 256]

  def query_rev_window(self, F_rev: torch.Tensor, pos: torch.Tensor, visible: torch.Tensor):
    b_pos = (pos + self.rev_window).minimum(visible)  # b_l = min(l + W, T)
    scale = (b_pos - pos).clamp_min(1).float().rsqrt().unsqueeze(-1)
    return apply_rope(F_rev[b_pos] - F_rev[pos], -pos, self.inv_freq) * scale  # Y_l: [..., 256]

  @torch.no_grad()
  def locate(
    self, X: torch.Tensor, q: torch.Tensor, candidates: list[tuple[int, int]],
    F_rev: torch.Tensor | None = None, H: torch.Tensor | None = None,
    min_start_prob: float = 0.6,
  ):
    T, k = X.shape[0], self.min_copy
    if T < k or not candidates:
      return None
    if F_rev is None:
      F_rev = self.build_rev_prefix(X)

    # ① 选起点 l 或 NO_COPY
    q = q.float()                                                                   # [512]
    pos = torch.tensor([p for p, _ in candidates], dtype=torch.long, device=X.device)
    ov  = torch.tensor([o for _, o in candidates], dtype=torch.float32, device=X.device).log1p().unsqueeze(-1)
    X_l = X[pos].float()                                                            # [C, 512]
    Y_l = self.query_rev_window(F_rev, pos, pos.new_full(pos.shape, T))             # [C, 256]

    s_cand = self.mlp_start(torch.cat([q.expand_as(X_l), X_l, Y_l, ov], dim=-1)).squeeze(-1)
    s_no   = self.mlp_no_copy(torch.cat([q, s_cand.max()[None], s_cand.mean()[None]]))
    prob, pick = torch.cat([s_no, s_cand], dim=0).softmax(dim=0).max(dim=0)
    if pick.item() == 0 or prob.item() < min_start_prob:
      return None
    l = int(pos[pick.item() - 1].item())

    # ② 粗选终点块 m_r（支持 m_r == m_l 同块）
    m_l, M = l // 32, (T + 31) // 32
    H = (self.mlp_span(X) if H is None else H).float()                              # [T, 256]
    H_ext = torch.cat([H, self.h_eos.float().unsqueeze(0)], dim=0)                  # [T + 1, 256]
    u = self.mlp_u(torch.cat([q, X[l].float()]))                                    # [256]

    n_cand = M - m_l
    blk_sum, blk_cnt, blk_max, h_a, h_b1 = [], [], [], [], []
    for m in range(m_l, M):
      a_m, b_m = max(l, 32 * m), min(32 * (m + 1), T) - 1
      seg = H[a_m : b_m + 1]
      blk_sum.append(seg.sum(dim=0))
      blk_cnt.append(b_m - a_m + 1)
      blk_max.append(seg.max(dim=0).values)
      h_a.append(H[a_m])
      h_b1.append(H_ext[b_m + 1])

    blk_sum = torch.stack(blk_sum)
    blk_cnt = torch.tensor(blk_cnt, dtype=torch.float32, device=X.device).unsqueeze(-1)
    blk_max = torch.stack(blk_max)
    phi_blk = torch.cat([blk_sum / blk_cnt, blk_max], dim=-1)                       # [n_cand, 512]

    if n_cand > 1:
      cum_sum = blk_sum[:-1].cumsum(dim=0)
      cum_cnt = blk_cnt[:-1].cumsum(dim=0)
      cum_max = blk_max[:-1].cummax(dim=0).values
      phi_pre = torch.cat([self.phi_same.float().unsqueeze(0), torch.cat([cum_sum / cum_cnt, cum_max], dim=-1)], dim=0)
    else:
      phi_pre = self.phi_same.float().unsqueeze(0)

    feat_2 = torch.cat([u.expand(n_cand, -1), phi_pre, phi_blk, torch.stack(h_a), torch.stack(h_b1)], dim=-1)
    m_seq = torch.arange(m_l, M, device=X.device)
    logits_2_sub = self.mlp_r_chunk(feat_2).squeeze(-1).masked_fill(m_seq < (l + k - 1) // 32, float("-inf"))
    idx_r = int(logits_2_sub.argmax().item())
    m_r = m_l + idx_r

    # ③ 终点块 m_r 内选末尾词 e
    a_r, b_r = max(l, 32 * m_r), min(32 * (m_r + 1), T) - 1
    seg_r = H[a_r : b_r + 1]
    L_r = seg_r.shape[0]
    c_sum = seg_r.cumsum(dim=0)
    c_cnt = torch.arange(1, L_r + 1, dtype=torch.float32, device=X.device).unsqueeze(-1)
    c_max = seg_r.cummax(dim=0).values

    if idx_r > 0:
      mu_e = (cum_sum[idx_r - 1] + c_sum) / (cum_cnt[idx_r - 1] + c_cnt)
      om_e = torch.maximum(cum_max[idx_r - 1], c_max)
    else:
      mu_e, om_e = c_sum / c_cnt, c_max

    e_pos = torch.arange(a_r, b_r + 1, device=X.device)
    feat_3 = torch.cat([u.expand(L_r, -1), mu_e, om_e, H[e_pos], H_ext[e_pos + 1]], dim=-1)
    logits_3_sub = self.mlp_r_tok(feat_3).squeeze(-1).masked_fill(e_pos < l + k - 1, float("-inf"))
    e = a_r + int(logits_3_sub.argmax().item())
    return l, e + 1
```

> 在线推理时，$X_t$、`F_rev[t + 1]`、$H_t$ 和满 $32$ 词块的统计量每步 $O(1)$ 追加缓存；$q$ 只对当前位置 $T - 1$ 算一次，不存历史。不复制的步只跑 2.1（约 `0.05 ms`），触发复制时再跑 2.2 和 2.3（约 `0.1 ms`），相比 27B 单步前向（约 `50 ms`）开销可忽略。

---

## 4. 训练方案

训练时冻结 `Qwen3.8-27B`，只更新 `CopyLocator` 参数。

复制终点的两类预测偏差代价不同：猜短几个词（$e < e^*$）只是少抄一点，验证仍能全过、不会回滚；猜长越界（$e > e^*$）哪怕多 1 个词也会验证失败，触发 48 层 GDN 状态回滚重算。Loss 按这个特性设计。

### 4.1 单个锚点的 Loss

设训练锚点当前可见长度为 $T$，第 1 步预测分布为 $p^{(1)} = \mathrm{softmax}\big(z^{(1)}\big) \in \mathbb{R}^{C+1}$（batch 对齐时的填充槽位在 softmax 前置为 $-\infty$）。

#### （1）负样本

当前位置不该复制，只算起点损失，目标是第 $0$ 类（`NO_COPY`）：

$$
\mathcal{L}_{\text{start}} = -\ln p^{(1)}_0
$$

#### （2）正样本起点损失

设真实极长复制区间为 $[l, r)$，长度 $L^* = r - l \ge k$，末尾词 $e^* = r - 1$。由于候选集保留了上文全部合法同首词位置，真起点 $l$ 必在 $\mathcal{C}$ 中。对每个候选 $l_j$，算出它在可见范围 $T$ 内与目标序列 $\text{tokens}[T : T + L^*]$ 的最长连续前缀匹配长度 $L_j \in [1,\, \min(L^*,\, T - l_j)]$，取满足最小复制长度的候选集合 $\mathcal{V} = \{j \mid L_j \ge k\}$（必非空），按匹配长度平方分配软目标权重 $y_j = L_j^2 / \sum_{m \in \mathcal{V}} L_m^2\ (j \in \mathcal{V})$，其余类为 $0$：

$$
\mathcal{L}_{\text{start}} = -\sum_{j \in \mathcal{V}} y_j \ln p^{(1)}_j
$$

当上文同时有“只对得上前几十词的 $l_1$”和“整段全对的 $l_2$”时，模型主要被推向最长源 $l_2$，同时不会把前缀确实相同的 $l_1$ 当成纯错样本硬罚。

#### （3）正样本终点损失

终点两步使用 Teacher Forcing，直接代入真实起点 $l$ 和真实终点块 $m_r^* = \lfloor e^* / 32 \rfloor$（块内偏移 $t^* = e^* \bmod 32$）。记两步 softmax 分布为 $p^{(2)}, p^{(3)}$，超过真终点的概率总和记为越界概率：

$$
P_{\text{over}}^{\text{blk}} = \sum_{m > m_r^*} p^{(2)}_m, \qquad P_{\text{over}}^{\text{tok}} = \sum_{t > t^*} p^{(3)}_t
$$

在主交叉熵上加一项越界惩罚（默认 $\lambda_{\text{over}} = 1.0$）：

$$
\mathcal{L}_{r,\text{block}} = -\ln p^{(2)}_{m_r^*} - \lambda_{\text{over}}\ln\!\big(1 - P_{\text{over}}^{\text{blk}}\big), \qquad \mathcal{L}_{r,\text{tok}} = -\ln p^{(3)}_{t^*} - \lambda_{\text{over}}\ln\!\big(1 - P_{\text{over}}^{\text{tok}}\big)
$$

- 精准命中极长终点 $e^*$ 时两项同时归零，全局最优解唯一落在 $e^*$，不会退化成只抄短前缀。
- 稍微猜短（$e < e^*$）只受第一项修正；一旦猜长越界（$e > e^*$）会同时受两项惩罚，拿不准边界时模型倾向先停在安全前缀上。

### 4.2 Batch 聚合

记一个 mini-batch 总样本数为 $N$，其中正样本数为 $N_{\text{pos}}$，分开求平均后相加（避免调整正负样本比例时影响终点分支的学习率）：

$$
\mathcal{L}_{\text{total}} = \frac{1}{N} \sum_{i = 1}^{N} \mathcal{L}_{\text{start}}^{(i)} + \frac{1}{N_{\text{pos}}} \sum_{i \in \text{pos}} \Big(\mathcal{L}_{r,\text{block}}^{(i)} + \mathcal{L}_{r,\text{tok}}^{(i)}\Big)
$$

### 4.3 采样与防泄漏细节

- **区间展开**：三元组 $(i, l, r)$ 在每个偏移 $d \in \{0, \dots, r - l - k\}$ 上展开为正样本 $(i + d,\, l + d,\, r)$。末尾只剩 $1 \sim k - 1$ 个词的位置词面仍与上文一致，直接跳过不采样。
- **正负配比**：把刚抄完必须停手的位置 $i + (r - l)$ 重点采为难负样本，配比取 **正样本 : 难负样本 : 普通负样本 $\approx 45 : 35 : 20$**；正样本内保证覆盖段首样本（$d < k$）和同块样本（$m_r = m_l$）。
- **防未来词泄漏**：整条训练序列一次算出全长 $X, F, H$ 后，对可见长度为 $T$ 的锚点做三处截断：
  1. 逆向窗口右端点取 $b_l = \min(l + W,\, T)$，不读 $> T$ 的 $F$；
  2. 终点尾块 $\lfloor (T - 1) / 32 \rfloor$ 只聚合 $< T$ 的词；
  3. 右边界下标等于 $T$（$b_m + 1 = T$ 或 $e + 1 = T$）时换成 $h_{\text{eos}}$，不读 $H_T$。

---

## 5. 在线验证与系统串联

### 5.1 支持温度采样的无损验证

1. 算完位置 $T - 1$ 后，先按当前解码策略（温度 $\tau$、`top_p` / `top_k`）从目标分布 $p_0(\cdot) = p_{\text{target}}(\cdot \mid x_{<T})$ 采出首词 $t_0 \sim p_0$。无论后续复不复制，位置 $T$ 输出的词都是 $t_0$，因此用 $t_0$ 筛候选池没有分布偏差，且首词 $d_0 = t_0$ 天然 $100\%$ 接受。
2. 若返回 $[l, r)$，取待验证片段 $d_{0 : L} = \text{tokens}[l : \min(r,\, l + B_{\max})]$（$L \ge k$），先调 `cache.snapshot()` 备份状态，再把 $d_{0 : L}$ 一次性送入大模型并行前向，得到后续各步目标分布 $p_i(\cdot) = p_{\text{target}}(\cdot \mid x_{<T}, d_{0 : i})\ (i = 1, \dots, L)$：
   - **温度采样（$\tau > 0$）**：后续复制词 $d_1, \dots, d_{L-1}$ 视为单点草稿分布 $q_i(v) = \mathbb{I}[v = d_i]$。依次采样 $u_i \sim \mathrm{Uniform}(0, 1)$，若 $u_i < p_i(d_i)$ 则接受 $d_i$；一旦在第 $i$ 步拒绝（已接受前 $a = i$ 个词），把 $p_i$ 中 $d_i$ 置零重归一化得到残差分布 $p'_i(v) = \mathbb{I}[v \ne d_i]\,\frac{p_i(v)}{1 - p_i(d_i)}$，采出修正词 $t_{\text{corr}} \sim p'_i$，调 `cache.restore()` 回滚并提交 $d_{0 : a} + [t_{\text{corr}}]$。若全部通过，保留缓存，下一轮从 $p_L(\cdot)$ 采新首词。输出分布与原模型逐步采样 $100\%$ 一致。
   - **贪心解码（$\tau = 0$）**：直接退化为逐位比对 $d_i == \arg\max \text{logits}$，首个不一致处取 $\arg\max$ 作为修正词。

### 5.2 与 DSpark 串联（不设独立 Router）

Copy Locator 判 `NO_COPY` 只需约 `0.05 ms` 且直接看上文精确匹配，前面加 Router 既省不了时间又容易误杀真复制；返回 `NO_COPY` 后直接进 DSpark（单次并行前向约 `1 ms`，自带动态长度裁剪 $\hat{k} \in [0, \gamma]$，$\hat{k} = 0$ 时直接输出 $t_0$ 走单步解码）。在线直接按 **`Copy Locator -> DSpark`** 串联即可，不用单独维护 Router。

### 5.3 底座改动清单

| 文件 | 改动说明 |
| --- | --- |
| `src/cache.py` | 新增 `snapshot()` / `restore()`，备份和回滚 `seq_len` 与 $48$ 层 GDN 状态；开 CoT 时 `max_len` 扩至 $16384$ 以上 |
| `src/gdn.py` | 分块三角求解 `chunk_gated_delta`，把 $48$ 层线性注意力逐词循环改成矩阵分块并行 |
| `src/qwen.py` | 多 token 验证时构造右下对齐因果掩码；前向顺带返回第 $3, 19, 47, 63$ 层 hidden_state |
| 新增文件 | `src/copy_locator/{model.py, candidates.py, decoder.py}`、`src/spec_decode.py`、`train/train_copy_locator.py`、`scripts/bench_copy.py` |

---

## 6. 实验记录

### 6.1 第一版：块均值选起点（1 轮 / 103 步）

仅训 `code` 1 轮，在 `8` 道 `code` 测试题（关 CoT）上草稿接受率仅 **`2.9%`**（解码 `10.4 tok/s`）。回放定位出三个原因：
1. 终点两步（`r_block` 降至 `0.09 ~ 0.36`）已经能停准代码块边界；
2. 起点用 $32$ 词块均值盲选彻底失效（命中率仅 `1.5%`），求平均抹掉了首词身份和局部词序，频繁选错相似块触发回滚；
3. 训练步数太少（`103` 步），且 48 层 GDN 为 Python 逐词循环（单次验证耗时 `320 ms`）。

### 6.2 第二版：同首词候选池 + 12 轮训练 + 分块 GDN

改为 `StartIndex` 同首词候选池（$[q,\, X_l,\, \ln(1 + \text{ov})]$，`min_start_prob = 0.6`），训满 `12` 轮（`1,176` 步），接入 `src/gdn.py` 分块并行并放开 `max_copy_len = 512`。仅用 `code` 单域训练权重在四域测试集（各 8 题，关 CoT）结果如下：

| 评测集 | CopyGen 纯解码 / 总耗时 / 接受率 | Locator 纯解码 / 总耗时 / 接受率 |
| --- | --- | --- |
| `code`（域内） | `201.6 tok/s` / `31.6s` / `95.2%` | **`222.0 tok/s` / `29.2s` / `100.0%`**（4,724/4,724） |
| `literature`（零样本） | `198.5 tok/s` / `11.9s` / `88.7%` | **`548.9 tok/s` / `6.9s` / `66.5%`** |
| `random_sequence`（零样本） | `123.6 tok/s` / `72.7s` / `99.5%` | **`155.0 tok/s` / `59.7s` / `93.2%`** |
| `contract`（零样本） | `88.7 tok/s` / `113.3s` / `78.7%` | **`120.7 tok/s` / `87.1s` / `11.5%`** |
| **32 题合计（EM 均 23/32）** | 总耗时 `229.6s` | **总耗时 `183.0s`**（快 `25.5%`） |

开 CoT 评测中，`code` 8/8 EM 全中；`literature` 开 CoT 阶段纯解码 `107.5 tok/s`（CopyGen `73.1 tok/s`）。

### 6.3 第三版：起点复杂度降阶对比与四域 + CoT 合训实测

在升级专用查询投影 $Q_n$、右看窗口 $W = 256$ 的逆向特征 $Y_l$、软标签起点损失与终点越界惩罚的基础上，针对起点筛选的 $O(C \cdot d^2)$ 时间复杂度做了多方案同分布对比实验。

#### （1）实验设置与对比方案

- **训练与验证数据**：从四域训练集抽取 `1,000` 条（`code / contract / literature` 各 `300` 条，`random_sequence` 压低至 `100` 条，含 `323` 条 CoT），验证集 `110` 条（共 `3,520` 个锚点：`1,784` 正 + `1,736` 负）。四组模型喂入完全相同的随机锚点同步训满 `6` 轮（`750` 步）；同时用同一份 `1,000` 条数据训满 `6` 轮得到 `CopyGen-4dom` 作为公平基线。
- **起点复杂度优化方案**：
  - **`v3_full`（全量原版，$O(C \cdot d^2)$）**：对全部 $C$ 个同首词候选实时算 $Y_l$ 并过 `GatedMLP(1281, 512, 1)`。
  - **`v3_preproj`（单阶段预封存解耦门控，$O(d^2 + C \cdot d)$，最终选定方案）**：所有 $l \le T - W$ 的位置右窗口终点固定为 $l + W$。去掉拼接后的联合 `RMSNorm`（改为输入端独立归一化），每步仅对刚满窗口的单个位置 $l = t - W$ 做 $O(1)$ 预投影并缓存 $[G_l,\, U_l] = W_{xy}[X_l,\, \mathrm{RMSNorm}_y(Y_l)] \in \mathbb{R}^{512}$（$h = 256$）。查询时只算一次 $[g_q,\, u_q] = W_q q$，对 $C$ 个候选不做矩阵乘、不重算 $Y_l$，直接逐元素相加过门控（同时把 `StartIndex` 改为仅做 $O(1)$ 的 `list.append`，查询时在 GPU 上并行算左重合度）：

$$
[g_j,\, u_j] = [g_q,\, u_q] + [G_{l_j},\, U_{l_j}] + \ln(1 + \text{ov}_j)\, w_{\text{ov}}, \qquad s_j = w_{\text{down}}^\top\big(\mathrm{SiLU}(g_j) \odot u_j\big)
$$

  - **`v3_coarse_dot`（64 维预封存内积粗筛 + Top-16 精排，$O(C \cdot r + K d^2)$）**：预封存 $K_l \in \mathbb{R}^{64}$，用 64 维内积筛出 Top-16 再过 `GatedMLP`；并额外测试在其上叠加在线球面聚类（$\lceil\sqrt{C}\rceil$ 簇探 Top-4 簇）+ 二元组精确表 `bigram` 的亚线性方案（$O(\sqrt{C} \cdot r + K d^2)$）。
  - **`v3_block_max`（修复版 32 词分块，$O(\min(C, \frac{T}{32}) d^2)$）**：加入同首词块掩码、改用 `[mean, max]` 池化 $[X_j, Y_j]$、补入块内最大 $\text{ov}$，取 Top-4 / Top-1 块再做块内精排。

#### （2）验证集精度与单步算子延迟扩展实测

| 起点方案 | 查询时间复杂度 | 候选池召回率 | 极长起点命中率 | 提议准确率 (`prec>=5`) | 负样本拒绝率 |
| --- | --- | --- | --- | --- | --- |
| `v3_full` | $O(C \cdot d^2)$ | `1784/1784` (`100%`) | `81.7%` | `99.0%` | `99.8%` |
| **`v3_preproj`（选定）** | **$O(d^2 + C \cdot d)$** | `1784/1784` (`100%`) | **`82.5%`** | **`99.6%`** | `99.8%` |
| `v3_coarse_dot` | $O(C \cdot r + K d^2)$ | `1783/1784` (`99.94%`) | `82.3%` | `99.4%` | `99.8%` |
| `v3_coarse_dot + IVF/Bigram` | $O(\sqrt{C} \cdot r + K d^2)$ | `1783/1784` (`99.94%`) | `82.4%` | `99.4%` | `99.8%` |
| `v3_block_max` | $O(\min(C, \frac{T}{32}) d^2)$ | Top-4块 `99.4%` / 单块 `96.5%` | `81.2%` | `99.2%` | `99.8%` |

在 `RTX PRO 6000 Blackwell` 上开启 `torch.compile` 测得的起点打分纯 GPU 延迟（按高频符号词频 $f \approx 4\%$ 折算上下文长度 $T$）：

| 候选数 $C$（折算上下文 $T$） | `v3_full` ($O(C d^2)$) | **`v3_preproj` ($O(C d)$)** | `v3_coarse_dot` ($O(C r + K d^2)$) |
| --- | ---: | ---: | ---: |
| $C = 64$（$\sim 1.6\text{K}$） | `87.1 μs` | **`66.0 μs`（1.32x）** | `92.1 μs` |
| $C = 1,024$（$\sim 25\text{K}$） | `149.8 μs` | **`68.1 μs`（2.20x）** | `102.9 μs`（1.46x） |
| $C = 4,096$（$\sim 100\text{K}$） | `336.9 μs` | **`66.5 μs`（5.07x）** | `101.5 μs`（3.32x） |
| $C = 16,384$（$\sim 400\text{K}$） | `1,112.2 μs` | **`67.8 μs`（16.39x）** | `194.8 μs`（5.71x） |
| $C = 32,768$（$\sim 800\text{K}$） | `2,344.4 μs` | **`103.6 μs`（22.63x）** | `309.0 μs`（7.59x） |

#### （3）32 道四域测试题端到端 `Qwen3.8-27B` 对比（各域 8 题，关 CoT）

底座纯 AR 解码速度为 `19.9 tok/s`，各方案同机实测结果如下：

| 方案 | `code` 解码 / 耗时 / 接受率 | `contract` 解码 / 耗时 / 接受率 | `literature` 解码 / 耗时 / 接受率 | `random` 解码 / 耗时 / 接受率 | **32 题总解码 / 总耗时 / 总接受率** |
| --- | --- | --- | --- | --- | --- |
| `CopyGen-code (cap=64)` | `183.8` / `31.6s` / `96.3%` | `86.3` / `88.0s` / `77.3%` | `190.5` / `10.7s` / `89.6%` | `90.5` / `59.5s` / `100%` | `108.6 tok/s` / `189.8s` / `88.6%` |
| `CopyGen-code (cap=512)` | `199.5` / `29.2s` / `93.3%` | `86.1` / `88.1s` / `77.3%` | `188.2` / `10.8s` / `89.6%` | `90.6` / `59.4s` / `100%` | `109.9 tok/s` / `187.5s` / `87.8%` |
| `CopyGen-4dom (cap=64)` | `182.7` / `31.6s` / `97.7%` | `95.3` / `80.3s` / `79.6%` | `265.7` / `8.3s` / `84.5%` | `100.8` / `53.9s` / `100%` | `119.3 tok/s` / `174.2s` / `89.6%` |
| `CopyGen-4dom (cap=512)` | `196.8` / `29.6s` / `95.2%` | `96.5` / `79.4s` / `73.0%` | `294.6` / `7.8s` / `82.4%` | `108.6` / `50.4s` / `100%` | `124.9 tok/s` / `167.2s` / `85.7%` |
| `Locator v3_full` | `214.7` / `27.5s` / `92.0%` | `166.3` / `49.5s` / `86.0%` | `1082.0` / `3.9s` / `71.3%` | `108.5` / `50.4s` / `72.7%` | `164.9 tok/s` / `131.3s` / `82.2%` |
| **`Locator v3_preproj`（选定）** | **`217.1` / `27.2s` / `95.3%`** | **`171.4` / `48.3s` / `92.6%`** | **`1102.4` / `3.9s` / `83.1%`** | **`110.5` / `49.4s` / `79.9%`** | **`168.5 tok/s` / `128.8s` / `88.9%`** |
| `Locator v3_coarse_dot` | `214.0` / `27.5s` / `88.7%` | `180.0` / `46.4s` / `84.1%` | `1087.3` / `3.9s` / `71.3%` | `107.5` / `50.9s` / `66.7%` | `168.9 tok/s` / `128.7s` / `78.9%` |
| `Locator v3_block_max` (Top-4块) | `216.5` / `27.3s` / `91.9%` | `164.8` / `49.9s` / `80.1%` | `1173.8` / `3.8s` / `83.2%` | `111.5` / `48.9s` / `88.8%` | `166.7 tok/s` / `129.9s` / `85.6%` |
| `Locator v3_block_max` (Top-1块) | `217.3` / `27.2s` / `83.5%` | `184.4` / `45.5s` / `78.6%` | `1077.3` / `3.9s` / `71.3%` | `106.6` / `51.3s` / `59.0%` | `170.2 tok/s` / `127.8s` / `73.2%` |

#### （4）结论小结

1. **选定 `PreProj-SwiGLU`**：将起点查询复杂度从 $O(C \cdot d^2)$ 降为 $O(d^2 + C \cdot d)$ 后，单阶段算子开销最小（`400K` 上下文仍稳定在 `68 μs`），且拆开联合 `RMSNorm` 消除了跨模态范数干扰，端到端接受率从 `82.2%` 提升到最高的 **`88.9%`**。
2. **不采用分块选起点**：修复版 `block_max` 虽把单块召回从 `1.5%` 救回 `96.5%`，但面对高频符号时几乎每个 32 词块都非空，块数仍为 $T / 32$，实际算子耗时在所有长度下最慢，且单块接受率掉至 `73.2%`。
3. **对比同训练集 `CopyGen-4dom`**：CopyGen 受限于只匹配静态 `prompt`（无法利用 `contract` 输出自复制）以及标量长度回归看不见真实终点边界，即使放开 `cap=512` 总耗时仍需 `167.2s`；`Locator v3_preproj` 总耗时降至 **`128.8s`**（纯解码快 `34.9%`，其中 `contract` 快 `1.78x`、`literature` 快 `3.74x`）。
