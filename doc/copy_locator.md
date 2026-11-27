# Copy Locator 设计说明

> 对齐数据格式：`doc/1.md`（上下文按 `prompt -> cot -> output` 拼接，DP 切分极长不交三元组 $(i, l, r)$，强制最小复制长度 $r - l \ge k$，默认 $k = 5$）

## 核心思路

设当前可见上下文共有 $T$ 个 token（下标 $0, 1, \dots, T - 1$），底座模型刚算完位置 $T - 1$。Copy Locator 根据当前状态判断：接下来从位置 $T$ 开始，能否直接从可见历史 $[0, T)$ 中整段复制某个区间 $[l, r)$。

为了不限制最大拷贝长度、同时避免对全部 $T$ 个词逐一精细打分，起点 $l$ 和终点 $r$（改写为预测最后一个被复制的词 $e = r - 1$，满足 $l + k - 1 \le e \le T - 1$）均采用 **“先按 $32$ 词一块选块，再在块内选词”** 的两级定位，共分四步：

```text
27B 四层隐状态 (Layer 3, 19, 47, 63)
        │
        ▼
  词特征 X [T, 512] ──┬──► 块均值 C [M, 512] ──► ① 选起点块 m_l 或 NO_COPY  (GatedMLP)
                      │                      ──► ② 块 m_l 内选精确起点 l    (GatedMLP)
                      ▼
             区间词特征 H [T, 256] (前缀和 + 前缀最大值聚合纯净区间 [l .. e])
                      │
                      ├──► ③ 选终点块 m_r：看 [前置必经段 + 候选块纯净切片 + 块边界]  (原生支持 m_r = m_l 同块)
                      └──► ④ 块 m_r 内选 e：看 [纯净区间 [l, e] + 停手边界 (H_e, H_{e+1})] ──► 输出 [l, e + 1)
```

三个核心设计原则：

- **统一用门控非线性块 $\mathrm{GatedMLP}$（`RMSNorm + SwiGLU`），彻底去掉裸点乘**：不再对未对齐的查询与候选直接做 $q \odot k$ 或内积，而是拼接异构特征 $z$ 后经 `RMSNorm` 送入与 `src/qwen.py` 同构的 `SwiGLU` 门控网络：

$$\mathrm{GatedMLP}(z) = W_{\text{down}}\Big(\mathrm{SiLU}\big(W_{\text{gate}}\,\mathrm{RMSNorm}(z)\big) \odot \big(W_{\text{up}}\,\mathrm{RMSNorm}(z)\big)\Big)$$

  其中 $W_{\text{gate}}$ 与 $W_{\text{up}}$ 先将不同来源特征独立投影到对齐子空间再做门控相乘：在找起点 $l$ 时充当跨空间特征匹配器，在定终点 $e$ 时充当“当前任务意图开关 $\times$ 区间断点检测器”。

- **定终点只看 $[l, e]$ 的纯净区间内容与停手边界，与长度/距离完全解耦**：在何处停止复制只取决于“$[l, e]$ 里抄了什么、内部有没有跨过断点、以及紧挨着的下一词 $e + 1$ 是不是转义/待改边界”，与复制长度是 $6$ 个词还是 $2000$ 个词无关，不引入任何距离偏置或标量长度回归。

- **均值 + 通道极值组成纯净区间算子 $\phi(a, b)$，原生支持同块定位（$m_r = m_l$）**：只求均值会把长区间内单个换行/围栏断点稀释为 $1/L$；同时保留均值与通道级最大值，既刻画整体语义又 $100\%$ 保留内部任意断点信号，并在同块情形下严格截掉块内 $< l$ 的无关前缀。

---

## 1. 特征构建与纯净区间算子

### 1.1 底座四层特征融合（$X$ 与起点块均值 $C$）

从冻结的 `Qwen3.8-27B`（共 $64$ 层，隐状态维度 $5120$）抽取第 $\mathcal{L} = \{3, 19, 47, 63\}$ 层隐状态 $h^{(n)}_j \in \mathbb{R}^{5120}$（分别对应浅层词面、中层局部结构、深层任务意图、最终输出前预测方向）。

对每个可见位置 $j \in \{0, \dots, T - 1\}$，四层特征各自降维到 $512$ 维后相加，叠加上下文阶段嵌入 $E_{\text{seg}}(s_j)$（$s_j \in \{0\text{: prompt},\, 1\text{: cot},\, 2\text{: output}\}$），再经 $\mathrm{RMSNorm}$ 归一化：

$$X_j = \mathrm{RMSNorm}\!\left(\sum_{n \in \{3, 19, 47, 63\}} P_n \, \mathrm{RMSNorm}_n\!\big(h^{(n)}_j\big) + E_{\text{seg}}(s_j)\right) \in \mathbb{R}^{512}$$

- **当前查询向量**：$q = X_{T-1} \in \mathbb{R}^{512}$（推理时按步追加缓存 $512$ 维的 $X_j$，$8\text{K}$ 上下文仅占约 $8\text{ MB}$）。
- **起点粗筛块均值** $C \in \mathbb{R}^{M \times 512}$（共 $M = \lceil T / 32 \rceil$ 块，尾块只对 $< T$ 的真实可见词求平均）：

$$C_m = \frac{1}{\min(32(m + 1), T) - 32m} \sum_{j = 32m}^{\min(32(m + 1),\, T) - 1} X_j \in \mathbb{R}^{512}, \qquad m = 0, \dots, M - 1$$

### 1.2 区间专用词特征 $H$ 与纯净区间算子 $\phi(a, b)$

在求区间聚合前，先通过 $\mathrm{GatedMLP}_{\text{span}}$ 将每个词映射为专用于跨度表示的 $256$ 维特征（各通道分别提取语义内容与换行、引号、括号闭合、代码缩进等边界信号）：

$$H_j = \mathrm{GatedMLP}_{\text{span}}(X_j) \in \mathbb{R}^{256}, \qquad 0 \le j < T$$

另设可学习的上下文末尾哨兵向量 $h_{\text{eos}} \in \mathbb{R}^{256}$，记扩展序列 $\tilde{H}_j = H_j\ (j < T)$，$\tilde{H}_T = h_{\text{eos}}$。

对任意闭区间 $[a, b]$（$0 \le a \le b < T$），定义其**纯净区间特征** $\phi(a, b) \in \mathbb{R}^{512}$ 为区间均值与通道级最大值的拼接：

$$\mu(a, b) = \frac{1}{b - a + 1}\sum_{j = a}^{b} H_j \in \mathbb{R}^{256}, \qquad \omega(a, b) = \max_{a \le j \le b} H_j \in \mathbb{R}^{256}, \qquad \phi(a, b) = \big[\mu(a, b),\, \omega(a, b)\big] \in \mathbb{R}^{512}$$

$\mu(a, b)$ 与 $\omega(a, b)$ 严格只依赖 $[a, b]$ 内部 token，数值尺度与区间长度完全无关，且均可借助前缀和 `cumsum` 与前缀最大值 `cummax` 在 $O(1)$ 均摊时间内并行算出。

---

## 2. 四步定位流程

打分与分类全程在 `float32` 下计算。由最小复制长度 $r - l \ge k$（$k = 5$），合法起点与末尾词下标须满足：

$$0 \le l \le T - k, \qquad l + k - 1 \le e \le T - 1$$

若 $T < k$，直接返回 $\text{NO\_COPY}$。

### ① 选起点块 $m_l$ 或 $\text{NO\_COPY}$（$M + 1$ 分类）

对所有历史块拼接 $[q,\, C_m]$ 打分，并将块起点 $32m > T - k$ 的非法块置为 $-\infty$：

$$s_m = \begin{cases} \mathrm{GatedMLP}_{l,\text{chunk}}\big([q,\, C_m]\big), & 32m \le T - k \\ -\infty, & 32m > T - k \end{cases}$$

在合法块集合 $\{m \mid 32m \le T - k\}$ 上取最高分 $s_{\max} = \max_m s_m$ 与均分 $s_{\text{mean}} = \operatorname{mean}_m s_m$，送入 $\mathrm{GatedMLP}_{\text{no}}$ 计算不复制分数 $s_{\text{no}}$，拼成第 ① 步 logits：

$$s_{\text{no}} = \mathrm{GatedMLP}_{\text{no}}\!\big([q,\, s_{\max},\, s_{\text{mean}}]\big), \qquad z^{(1)} = [s_{\text{no}},\, s_0,\, \dots,\, s_{M-1}] \in \mathbb{R}^{M+1}$$

- 若 $\arg\max z^{(1)} = 0$：当前步不复制，直接返回 $\text{NO\_COPY}$；
- 否则选中起点块 $m_l = \arg\max z^{(1)} - 1$。

### ② 在块 $m_l$ 内选精确起点 $l$（$32$ 分类）

对块 $m_l$ 内的偏移 $t \in \{0, \dots, 31\}$（对应全局下标 $p = 32 m_l + t$）打分：

$$z^{(2)}_t = \begin{cases} \mathrm{GatedMLP}_{l,\text{tok}}\big([q,\, X_p]\big), & p \le T - k \\ -\infty, & p > T - k \end{cases}$$

选出精确起点：

$$l = 32 m_l + \arg\max_{0 \le t < 32} z^{(2)}_t$$

### ③ 基于块间纯净区间特征，选终点块 $m_r$（$M$ 分类，支持同块）

确定 $l$ 后（所在块 $m_l = \lfloor l / 32 \rfloor$），先将当前状态与起点特征融合为任务起点意图向量：

$$u = \mathrm{GatedMLP}_u\!\big([q,\, X_l]\big) \in \mathbb{R}^{256}$$

终点块最小合法块号为 $m_{\min} = \lfloor (l + k - 1) / 32 \rfloor$（当 $l \bmod 32 \le 32 - k$ 时 $m_{\min} = m_l$，即起点与终点可落在同一块内）。对任意候选块 $m \in \{m_l, \dots, M - 1\}$，其从 $l$ 起算的块内合法切片范围为：

$$a_m = \max(l,\, 32m), \qquad b_m = \min\big(32(m + 1),\, T\big) - 1$$

将从 $l$ 到块 $m$ 的纯净区间拆解为 **“前置必经段 + 候选块自身切片 + 块首尾边界”** 三部分：

1. **前置必经段特征 $\phi_{\text{pre}}(m) \in \mathbb{R}^{512}$**：
   - 当 $m = m_l$（**同块情形**）：前置为空，取可学习的同块标识向量 $\phi_{\text{same}} \in \mathbb{R}^{512}$；
   - 当 $m > m_l$（**跨块情形**）：取 $\phi(l,\, 32m - 1)$。若真实终点落在块 $m$，前置区间 $[l,\, 32m - 1]$ 必须完整无断点；一旦更早的块已出现断点，$\omega(l,\, 32m - 1)$ 会直接抑制后续所有块。
2. **候选块纯净切片 $\phi_{\text{blk}}(m) = \phi(a_m,\, b_m) \in \mathbb{R}^{512}$**：同块时 $a_{m_l} = l$，只聚合 $[l,\, b_{m_l}]$，严格排除块内 $< l$ 的前缀噪声。
3. **块首尾边界特征 $[H_{a_m},\, \tilde{H}_{b_m + 1}] \in \mathbb{R}^{512}$**：兼顾刚好停在块末 $b_m$、断点词落在下一块开头 $b_m + 1$ 的边界情况。

拼接成 $1792$ 维向量计算第 ③ 步 logits（$m < m_{\min}$ 置为 $-\infty$）：

$$z^{(3)}_m = \begin{cases} \mathrm{GatedMLP}_{r,\text{chunk}}\!\Big(\big[u,\, \phi_{\text{pre}}(m),\, \phi_{\text{blk}}(m),\, H_{a_m},\, \tilde{H}_{b_m + 1}\big]\Big), & m \ge m_{\min} \\ -\infty, & m < m_{\min} \end{cases}$$

选出终点块：

$$m_r = \arg\max_{0 \le m < M} z^{(3)}_m$$

### ④ 基于精确 $[l, e]$ 纯净区间特征，在块 $m_r$ 内选末尾词 $e$（$32$ 分类）

选中终点块 $m_r$ 后，记块内起点 $a_r = \max(l,\, 32 m_r)$、终点 $b_r = \min(32(m_r + 1),\, T) - 1$。复用第 ③ 步已算出的前置聚合量（$m_r > m_l$ 时为 $[l,\, 32 m_r - 1]$ 的和与最大值）与块内切片 $[a_r,\, b_r]$ 的前缀 `cumsum` / `cummax` 合并，即可并行得到块内每个位置 $e = 32 m_r + t \in [a_r,\, b_r]$ 的精确纯净区间特征 $\phi(l, e) \in \mathbb{R}^{512}$。

再拼上停手交界处的两个词 $[H_e,\, \tilde{H}_{e + 1}]$（最后被复制的词 $H_e$ 与紧随其后第一个不复制的词 $\tilde{H}_{e + 1}$），共 $1280$ 维送入 $\mathrm{GatedMLP}_{r,\text{tok}}$：

$$z^{(4)}_t = \begin{cases} \mathrm{GatedMLP}_{r,\text{tok}}\!\Big(\big[u,\, \phi(l, e),\, H_e,\, \tilde{H}_{e + 1}\big]\Big), & l + k - 1 \le 32 m_r + t < T \\ -\infty, & \text{otherwise} \end{cases}$$

选出精确末尾词并输出左闭右开复制区间：

$$e = 32 m_r + \arg\max_{0 \le t < 32} z^{(4)}_t \quad \Longrightarrow \quad [l,\, r) = [l,\, e + 1)$$

---

## 3. PyTorch 核心结构参考

整个定位器参数量约 $17.9\text{M}$（`bf16` 下约 $36\text{ MB}$），前向逻辑与上述公式一一对应：

```python
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
  def __init__(self, d_model: int = 5120, d: int = 512, d_span: int = 256, min_copy: int = 5):
    super().__init__()
    self.min_copy   = min_copy
    self.proj_norms = nn.ModuleList([RMSNorm(d_model) for _ in range(4)])
    self.projs      = nn.ModuleList([nn.Linear(d_model, d, bias=False) for _ in range(4)])
    self.seg_emb    = nn.Embedding(3, d)  # 0: prompt, 1: cot, 2: output
    self.out_norm   = RMSNorm(d)

    self.mlp_l_chunk = GatedMLP(d * 2, 512, 1)
    self.mlp_no_copy = GatedMLP(d + 2, 256, 1)
    self.mlp_l_tok   = GatedMLP(d * 2, 512, 1)

    self.mlp_span    = GatedMLP(d, 512, d_span)
    self.mlp_u       = GatedMLP(d * 2, 512, d_span)
    self.phi_same    = nn.Parameter(torch.zeros(d_span * 2))
    self.h_eos       = nn.Parameter(torch.zeros(d_span))

    self.mlp_r_chunk = GatedMLP(d_span * 7, 512, 1)  # 256 + 512 + 512 + 256 + 256 = 1792
    self.mlp_r_tok   = GatedMLP(d_span * 5, 512, 1)  # 256 + 512 + 256 + 256 = 1280

  def encode_step(self, hs_4: list[torch.Tensor], seg_ids: torch.Tensor):
    x = sum(proj(norm(h)) for norm, proj, h in zip(self.proj_norms, self.projs, hs_4))
    return self.out_norm(x + self.seg_emb(seg_ids))  # [seq_len, 512]

  @torch.no_grad()
  def locate(self, X: torch.Tensor):  # X: [T, 512] (float32)
    T, k = X.shape[0], self.min_copy
    if T < k:
      return None

    M = (T + 31) // 32
    C = torch.stack([X[32 * m : min(32 * (m + 1), T)].mean(dim=0) for m in range(M)])
    q = X[T - 1]

    # ① 选起点块 m_l 或 NO_COPY
    s_chunk = self.mlp_l_chunk(torch.cat([q.expand_as(C), C], dim=-1)).squeeze(-1)
    valid_m = (torch.arange(M, device=X.device) * 32) <= (T - k)
    s_chunk = s_chunk.masked_fill(~valid_m, float("-inf"))
    s_valid = s_chunk[valid_m]
    s_no = self.mlp_no_copy(torch.cat([q, s_valid.max()[None], s_valid.mean()[None]]))
    pick = int(torch.cat([s_no, s_chunk]).argmax().item())
    if pick == 0:
      return None
    m_l = pick - 1

    # ② 块内选精确起点 l
    b_l_end = min(32 * (m_l + 1), T)
    B_l = X[32 * m_l : b_l_end]
    pos_l = torch.arange(32 * m_l, b_l_end, device=X.device)
    logits_2 = self.mlp_l_tok(torch.cat([q.expand_as(B_l), B_l], dim=-1)).squeeze(-1)
    logits_2 = logits_2.masked_fill(pos_l > T - k, float("-inf"))
    l = 32 * m_l + int(logits_2.argmax().item())

    # ③ 构造从 l 起算的块级纯净区间特征，选终点块 m_r（兼容 m_r == m_l 同块）
    H = self.mlp_span(X)                                     # [T, 256]
    H_ext = torch.cat([H, self.h_eos.unsqueeze(0)], dim=0)   # [T + 1, 256]
    u = self.mlp_u(torch.cat([q, X[l]]))                     # [256]

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
    blk_cnt = torch.tensor(blk_cnt, dtype=X.dtype, device=X.device).unsqueeze(-1)
    blk_max = torch.stack(blk_max)
    phi_blk = torch.cat([blk_sum / blk_cnt, blk_max], dim=-1)  # [n_cand, 512]

    if n_cand > 1:
      cum_sum = blk_sum[:-1].cumsum(dim=0)
      cum_cnt = blk_cnt[:-1].cumsum(dim=0)
      cum_max = blk_max[:-1].cummax(dim=0).values
      phi_pre = torch.cat([self.phi_same.unsqueeze(0), torch.cat([cum_sum / cum_cnt, cum_max], dim=-1)], dim=0)
    else:
      phi_pre = self.phi_same.unsqueeze(0)

    feat_3 = torch.cat([u.expand(n_cand, -1), phi_pre, phi_blk, torch.stack(h_a), torch.stack(h_b1)], dim=-1)
    m_seq = torch.arange(m_l, M, device=X.device)
    logits_3_sub = self.mlp_r_chunk(feat_3).squeeze(-1).masked_fill(m_seq < (l + k - 1) // 32, float("-inf"))
    idx_r = int(logits_3_sub.argmax().item())
    m_r = m_l + idx_r

    # ④ 在终点块 m_r 内构造精确 [l, e] 纯净区间特征，选末尾词 e
    a_r, b_r = max(l, 32 * m_r), min(32 * (m_r + 1), T) - 1
    seg_r = H[a_r : b_r + 1]
    L_r = seg_r.shape[0]
    c_sum = seg_r.cumsum(dim=0)
    c_cnt = torch.arange(1, L_r + 1, dtype=X.dtype, device=X.device).unsqueeze(-1)
    c_max = seg_r.cummax(dim=0).values

    if idx_r > 0:
      mu_e = (cum_sum[idx_r - 1] + c_sum) / (cum_cnt[idx_r - 1] + c_cnt)
      om_e = torch.maximum(cum_max[idx_r - 1], c_max)
    else:
      mu_e, om_e = c_sum / c_cnt, c_max

    e_pos = torch.arange(a_r, b_r + 1, device=X.device)
    feat_4 = torch.cat([u.expand(L_r, -1), mu_e, om_e, H[e_pos], H_ext[e_pos + 1]], dim=-1)
    logits_4_sub = self.mlp_r_tok(feat_4).squeeze(-1).masked_fill(e_pos < l + k - 1, float("-inf"))
    e = a_r + int(logits_4_sub.argmax().item())
    return l, e + 1
```

> **推理缓存加速**：在线解码时，$X_t$ 与 $H_t$ 均按步增量计算并缓存，已走完的完整 $32$ 词块的 `blk_sum` 与 `blk_max` 也一并缓存；第 ③ 步只需单独算首块残片 $[l,\, b_{m_l}]$ 和最末尾块（各不超过 $32$ 个词），其余满块直接查表。

---

## 4. 训练方案

训练时底座 `Qwen3.8-27B` 全程冻结，只更新 `CopyLocator` 参数。由于四步输出的全部是分类 logits，全程只用标准交叉熵 `F.cross_entropy`，不含任何标量回归项或手工损失权重。

### 4.1 单个锚点的四项交叉熵

设一个训练锚点的当前可见长度为 $T$：

- **负样本（当前位置不该复制）**：仅监督第 ① 步在 $M + 1$ 类中选中第 $0$ 类（$\text{NO\_COPY}$），第 ②③④ 步不计算损失：

$$\mathcal{L}_1 = \mathrm{CE}\!\big(z^{(1)},\, 0\big)$$

- **正样本（真实复制区间为 $[l, r)$，记末尾词 $e = r - 1$）**：第 ②③④ 步全部采用 **Teacher Forcing**（直接喂入真实的起点块 $m_l = \lfloor l / 32 \rfloor$、真实起点 $l$、真实终点块 $m_r = \lfloor e / 32 \rfloor$，防止训练初期前置步骤预测偏差污染后续梯度），四步各算一项交叉熵（直接对应第 3 节代码中的切片 logits）：

$$\begin{aligned}
\mathcal{L}_1 &= \mathrm{CE}\!\big(z^{(1)},\, 1 + m_l\big) && \text{（在 } M+1 \text{ 类中选中起点块，同时压低 NO\_COPY）} \\
\mathcal{L}_2 &= \mathrm{CE}\!\big(z^{(2)},\, l - 32 m_l\big) && \text{（在起点块内选准精确起点 } l \text{）} \\
\mathcal{L}_3 &= \mathrm{CE}\!\big(z^{(3)}_{\text{sub}},\, m_r - m_l\big) && \text{（在 } m \ge m_l \text{ 的候选块中选准终点块 } m_r \text{）} \\
\mathcal{L}_4 &= \mathrm{CE}\!\big(z^{(4)}_{\text{sub}},\, e - a_r\big) && \text{（在 } [a_r, b_r] \text{ 内选准末尾词 } e \text{，其中 } a_r = \max(l, 32 m_r)\text{）}
\end{aligned}$$

### 4.2 Batch 级聚合方式（分母按有效样本数分别归一化）

设一个 mini-batch 共采样 $N$ 个锚点（包含 $N_{\text{pos}}$ 个正样本与 $N_{\text{neg}}$ 个负样本），第 ① 步在全量锚点上取平均，第 ②③④ 步仅在正样本集合内取平均后再相加：

$$\mathcal{L}_{\text{total}} = \frac{1}{N} \sum_{i=1}^{N} \mathcal{L}_1^{(i)} + \frac{1}{N_{\text{pos}}} \sum_{i \in \text{pos}} \Big(\mathcal{L}_2^{(i)} + \mathcal{L}_3^{(i)} + \mathcal{L}_4^{(i)}\Big)$$

分开归一化的原因：$\mathcal{L}_1$ 需要正负样本共同校准触发阈值，分母取 $N$ 可真实反映正负配比；而 $\mathcal{L}_2, \mathcal{L}_3, \mathcal{L}_4$ 仅在正样本上存在，除以 $N_{\text{pos}}$ 能保证调整正负采样比例时，后三步定位器的有效学习率保持恒定、不被负样本数量冲淡。

### 4.3 样本展开、难负采样与防穿越细节

1. **极长区间展开与短尾忽略区（`ignore`）**：`doc/1.md` 中的一条三元组 $(i, l, r)$，在每个偏移 $d \in \{0, 1, \dots, (r - l) - k\}$ 上展开为一个正样本（对应生成下标 $i + d$，目标区间 $[l + d,\, r)$）。区间末尾只剩 $1 \sim k - 1$（即 $1 \sim 4$）个词的位置，虽不足最小复制长度 $k = 5$，但词面确实与上文一致——若强标为 `NO_COPY` 会与正样本梯度冲突，因此这 $k - 1$ 个位置**既不当正样本也不当负样本，直接跳过不采样**。
2. **难负样本与同块样本覆盖**：
   - 将复制刚结束、必须停手的交界位置 $i + (r - l)$ 高比例采为**难负样本**（逼迫 $\mathcal{L}_1$ 学会抄完立即切回自回归生成），其余普通非复制位置随机下采样，配比取 **正样本 : 难负样本 : 普通负样本 $\approx 45 : 35 : 20$**；
   - 正样本内部确保覆盖充足的**同块样本**（$\lfloor (l + d) / 32 \rfloor = \lfloor (r - 1) / 32 \rfloor$，`contract` 天然含约 $36.7\%$ 原始同块区间，各域展开至尾部 $< 32$ 词时也会自然落入同块），使 $\phi_{\text{same}}$ 与同块切片分支充分收敛。
3. **防未来词穿越**：整条序列一次前向算出完整 $X$ 与 $H$ 后，对可见长度为 $T$ 的训练锚点：
   - 第 ① 步尾块 $m_{\text{tail}} = \lfloor (T - 1) / 32 \rfloor$ 必须只对 $\{X_{32 m_{\text{tail}}}, \dots, X_{T-1}\}$ 求均值；
   - 第 ③、④ 步右边界前瞻在下标恰好等于当前可见长度 $T$（即 $b_m + 1 = T$ 或 $e + 1 = T$）时，**必须替换为哨兵向量 $h_{\text{eos}}$**，严禁读取全序列张量里的 $H_T$（真实未来词），保证训练与在线推理严格一致。

---

## 5. 无损验证与底座改动

### 在线推理闭环（$100\%$ 贪心无损）

1. 底座刚算完位置 $T - 1$ 时，自身的 `lm_head` 已免费给出位置 $T$ 的贪心预测词 $t_0 = \arg\max \text{logits}_{T-1}$。
2. 调用 `CopyLocator.locate(X[:T])`：
   - 若返回 $\text{NO\_COPY}$，或预测的复制首词 $\text{tokens}[l] \ne t_0$（首词即与底座贪心预测矛盾），**零验证开销直接放弃复制**，正常输出 $t_0$ 进入下一步；
   - 若 $\text{tokens}[l] = t_0$，先调用 `cache.snapshot()` 备份缓存状态，将候选片段 $\text{tokens}[l : r]$（长 $L = r - l \ge k$）一次性送入底座做单次并行前向，逐位核对贪心 $\arg\max$：设连续通过验证的复制词数为 $a$（$1 \le a \le L$），若 $a = L$ 直接收下整段加上末尾预测的新贪心词；若 $a < L$ 调用 `cache.restore()` 回滚并推进已确认的前 $a$ 个词与修正词。

### 底座最小改动清单

| 文件 | 改动说明 |
|---|---|
| `src/cache.py` | 新增 `snapshot()` / `restore()`：备份与回滚 `seq_len` 以及 $48$ 层 GDN 的 `conv_state`、`recur_state`（$16$ 层 Full Attention 的 KV Cache 靠 `seq_len` 截断覆盖即可，无需拷贝）；开 CoT 时 `max_len` 扩至 $16384$ 以上 |
| `src/qwen.py` | `Attention.forward`：多 token 并行验证（`not cache.is_prefill and seq_len > 1`）时构造右下对齐因果掩码；`Model.forward`：顺手返回第 $3, 19, 47, 63$ 层隐状态 |
| 新增文件 | `src/copy_locator/{model.py, decoder.py}`、`train/train_copy_locator.py`、`scripts/bench_copy.py` |

---

## 6. 实验复盘与架构演进记录

### 6.1 第一版实测与根因定位（纯块均值选起点，1 轮 / 103 步）

- **实验条件**：仅使用 `code` 训练集 `821` 条（含 `273` 条 CoT），`epochs = 1`、`grad_accum = 8`（共 `103` 步），在 `8` 道 `code` 测试题上关闭 CoT、无损验证（`max_copy_len = 128`）。
- **实测结果**：官方 EM 均为 `8/8` 且逐 token 与 AR 一致，但 Locator 纯解码仅 `10.4 tok/s`（AR `19.6 tok/s`，CopyGen `124.8 tok/s`），草稿接受率仅 **`2.9%`**（送验 `104,694` 词仅接受 `2,984` 词）。
- **逐 token 回放诊断出的核心根因**：
  1. **第 ③、④ 步（纯净区间定终点 $r$）已学出规律**：训练日志中 `r_block` 迅速降至 `0.09 ~ 0.36`，回放中能精准停在代码块右边界。
  2. **第 ① 步（32 词块均值 $C_m$ 盲选起点块）彻底崩盘**：测试题回放中起点块命中率仅 `1.5%`（`7 / 466`），把目标字典块误判为同文件另一函数的相似字典；第 ② 步在错块中仍挑出了常见符号（`\n`、`' "'`、`'":'`），骗过单 token 首词预筛 `draft[0] == next_id`（`245` 次假阳性），反复触发 `128` 词验证 + 48 层 GDN 回滚重放。
  3. **训练严重欠拟合 + GDN 引擎串行瓶颈**：`17.9M` 参数仅更新 `103` 步，结束时 `l_block` 仍高达 `1.45`；同时 `src/qwen.py` 中 48 层 GDN 为纯 Python `for t in range(seq_len)` 串行循环，验证 ~86 词耗时高达 `320 ms`（单步 AR 仅 `51 ms`）。

---

### 6.2 第二版核心改动与复测结果（首词候选池 + 12 轮训练 + 分块 GDN）

针对上述问题完成了四项核心升级：

1. **起点定位改为“首词硬筛候选池 + 语义/后缀联合打分”**：放弃用模糊的 32 词块均值盲搜起点，直接利用免费拿到的 `next_id` 在可见历史中筛出 `tokens[l] == next_id` 的候选位置（`StartIndex`），拼接 `[q,\, X_l,\, \log(1 + \text{overlap})]` 经 `GatedMLP` 与 `NO_COPY` 联合打分，并增加触发置信度门限 `min_start_prob = 0.6`；终点定位保持纯净区间 $[\mu, \omega]$ 的第 ③、④ 步不变。
2. **离线缓存四层 BF16 特征并充分训练**：留出 `41` 条验证集，其余 `780` 条训满 `12` 轮（`1,176` 步）。
3. **分块 GDN 加速（`src/gdn.py`）**：将 48 层线性注意力多 token 递推改为分块三角求解 `chunk_gated_delta`，大幅压缩 Prefill 与长草稿验证耗时。
4. **放开 `max_copy_len` 至 `512`**：释放 CopyLocator 不限长精确指终点的优势。

**复测结果（仍为仅用 `code` 单域训练出的权重，在其余三域为零样本迁移）**：

| 评测集（关 CoT，各 8 题） | CopyGen 纯解码 / 总耗时 / 接受率 | Locator 纯解码 / 总耗时 / 接受率 |
|---|---|---|
| `code`（域内） | `201.6 tok/s` / `31.6s` / `95.2%` | **`222.0 tok/s` / `29.2s` / `100.0%`**（4,724/4,724） |
| `literature`（零样本） | `198.5 tok/s` / `11.9s` / `88.7%` | **`548.9 tok/s` / `6.9s` / `66.5%`** |
| `random_sequence`（零样本） | `123.6 tok/s` / `72.7s` / `99.5%` | **`155.0 tok/s` / `59.7s` / `93.2%`** |
| `contract`（零样本） | `88.7 tok/s` / `113.3s` / `78.7%` | **`120.7 tok/s` / `87.1s` / `11.5%`** |
| **32 题合计（EM 均 23/32）** | 总耗时 `229.6s` | **总耗时 `183.0s`**（快 `25.5%`） |

- 开 CoT 评测中，`code` 8/8 EM 全部命中；`literature` 开 CoT 阶段 Locator 纯解码达 `107.5 tok/s`（CopyGen `73.1 tok/s`，快 `47%`）。

---

### 6.3 已识别的后续优化点

1. **冷启动候选池截断与错标修复**：`MAX_CANDIDATES = 32` 在冷启动（`overlap = 0/1`）时按 `(overlap, pos)` 仅保留末尾 32 个位置，导致 `code train` 的 `836` 个区间开头只有 `27` 个入池，且未入池正样本在 `start_supervision` 中被默认标成了 `target = 0`（`NO_COPY`）。后续应扩池至 `128 ~ 256`、对未入池正样本使用 `ignore_index = -100` 忽略、支持多等价正例损失，并在起点打分中补入前驱特征 $X_{l-1}$ 对齐因果相位。
2. **四域 + CoT 联合训练**：当前权重仅用 `code` 单域训练，在 `contract` 上因未学过 JSON 转义停手规则导致接受率仅 `11.5%`，且在 CoT 短引用处易偏激进地指到整段末尾。四个域的训练数据与 `copy_cot` 已全部就绪，合训后可同步拉高 `contract` 与 CoT 阶段的草稿接受率。
3. **CoT 多来源确定性标注与工程细节**：带 CoT 标注时先由 DP 确定目标切分，再按 `(in_prompt, left_overlap, tie_pos)` 确定性选唯一来源（`prompt` 源头优先，保证开/关 CoT 标签一致与右边界干净）；工程侧预分配 `Cache.snapshot()` 静态缓冲并惰性计算候选后缀重合；另需注意长 CoT 下 `bfloat16` 分块归约尾数差可能导致 top-1/top-2 极近的思考词轨迹分叉。
