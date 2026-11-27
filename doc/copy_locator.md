# Copy Locator 设计说明

> 对齐数据格式：`doc/1.md`（上下文按 `prompt -> cot -> output` 拼接，单调队列 DP 切分极长不交三元组 $(i, l, r)$，强制最小复制长度 $r - l \ge k$，默认 $k = 5$）

## 核心思路

当前上下文共有 $T$ 个 token，大模型刚算完 $T - 1$，其 `lm_head` 已免费给出位置 $T$ 的目标采样分布，可直接按当前解码策略（温度采样或贪心解码）采样确定位置 $T$ 的首词 $t_0$。

Copy Locator 在这一刻介入判断：接下来从位置 $T$ 开始，能否直接从 $[0, T)$ 中整段复制某个区间 $[l, r)$（下文为了方便，改为左闭右闭区间，$[l, e = r - 1]$），满足 $0 \le l \le T - k$ 且 $l + k - 1 \le e \le T - 1$。

【这段看不懂】

为兼顾起点冷启动消歧精度与任意长度终点定位效率，整体采用 **“同首词候选池 + 正逆向解耦特征选起点 $l$ $\to$ 按 $32$ 词分块两级纯净区间定终点 $e$”** 的三步架构：

```text
27B 四层隐状态 (Layer 3, 19, 47, 63) + 由 logits_{T-1} 采样确定的首词 t_0
        │
        ▼
  正向词特征 X [T, 512] ──┬──► 逆向 RoPE 前缀和 F [T+1, 256] ──► 右看窗口特征 Y_l (W = 256, O(1) 区间差)
                          │
                          ├──► ① 选精确起点 l 或 NO_COPY：StartIndex 筛 tokens[l] == t_0 候选池，
                          │      拼接 [q, X_l, Y_l, ln(1 + ov)] 经 GatedMLP 打分 (正逆向严格分离)
                          ▼
                 区间词特征 H [T, 256] (前缀和 + 前缀最大值聚合纯净区间 [l .. e])
                          │
                          ├──► ② 选终点块 m_r：看 [前置必经段 + 候选块纯净切片 + 块边界]  (原生支持 m_r = m_l 同块)
                          └──► ③ 块 m_r 内选 e：看 [纯净区间 [l, e] + 停手边界 (H_e, H_{e+1})] ──► 输出 [l, e + 1)
```

四个核心设计原则：

1. **统一用门控非线性块 $\mathrm{GatedMLP}$（`RMSNorm + SwiGLU`），彻底去掉裸点乘**：不再对未对齐的查询与候选直接做 $q \odot k$ 或内积，而是拼接异构特征 $z$ 后经 `RMSNorm` 送入与 `src/qwen.py` 同构的 `SwiGLU` 门控网络（$W_{\text{gate}}$ 与 $W_{\text{up}}$ 先将不同来源特征独立投影到对齐子空间再做门控相乘，在找起点 $l$ 时充当跨空间特征匹配器，在定终点 $e$ 时充当“当前任务意图开关 $\times$ 区间断点检测器”）：
2. **起点表征正逆向严格分离（正向 $X_l$ 看左上下文，逆向 $Y_l$ 看右窗口 $W = 256$）**：因果底座在位置 $l$ 的正向隐状态 $X_l$ 只能看见 $\le l$ 的历史，遇到多处共享长公共前缀、在后续位置才分叉的相似段落时会失明。为此引入无衰减 RoPE 前缀和构造右看窗口特征 $Y_l$（覆盖 $[l,\, \min(l + W, T))$，$W = 256$），且**绝不与 $X_l$ 相加混用**，而是独立拼接为 $[q,\, X_l,\, Y_l,\, \ln(1 + \text{ov}_l)]$，分别对齐“左侧来龙去脉”与“右侧即将吐出的词序”。
3. **定终点只看 $[l, e]$ 的纯净区间内容与停手边界，与长度/距离完全解耦**：一旦定下 $l$，在何处停止复制只取决于“$[l, e]$ 里抄了什么、内部有没有跨过断点、以及紧挨着的下一词 $e + 1$ 是不是转义/待改边界”，与复制长度是 $5$ 个词还是 $2500$ 个词无关，不引入距离偏置或标量长度回归（也不向终点分支注入固定窗口的 $Y_l$，防止 $e - l < W$ 时混入 $> e$ 的区间外噪声）。
4. **均值 + 通道极值组成纯净区间算子 $\phi(a, b)$，原生支持同块定位（$m_r = m_l$）**：只求均值会把长区间内单个换行/围栏断点稀释为 $1/L$；同时保留均值与通道级最大值，既刻画整体语义又 $100\%$ 保留内部任意断点信号，并在同块情形下严格截掉块内 $< l$ 的无关前缀。

全模型统一使用的基础门控算子定义为：

$$\mathrm{GatedMLP}(z) = W_{\text{down}}\Big(\mathrm{SiLU}\big(W_{\text{gate}}\,\mathrm{RMSNorm}(z)\big) \odot \big(W_{\text{up}}\,\mathrm{RMSNorm}(z)\big)\Big)$$

【这段看不懂】

---





从冻结的 `Qwen3.8-27B`（共 $64$ 层，hidden_state 维度 $d_{\text{model}} = 5120$）抽取第 $3, 19, 47, 63$ 层 hidden_state $h^{(n)}_j \in \mathbb{R}^{5120}$。

对每个可见位置 $j \in \{0, \dots, T - 1\}$，四层特征经 Norm 后通过投影矩阵 $P_n \in \mathbb{R}^{512 \times 5120}$ 降维相加，叠加上下文阶段信息 $E_{\text{seg}}(s_j)$（$s_j \in \{0\text{: prompt},\, 1\text{: cot},\, 2\text{: output}\}$），再经 $\mathrm{RMSNorm}_x$ 得到正向词特征 $X_j$。

同时，对当前生成位置 $T - 1$，生成一个查询特征 $q$。$q$ 不使用 $P_n$，而是用专门的查询投影矩阵 $Q_n \in \mathbb{R}^{512 \times 5120}$ 独立提取查询语义得到 $q$：

$$
X_j = \mathrm{RMSNorm}_x\!\left(\sum_{n \in \{3, 19, 47, 63\}} P_n \, \mathrm{RMSNorm}_n\!\big(h^{(n)}_j\big) + E_{\text{seg}}(s_j)\right) \in \mathbb{R}^{512}, \qquad 0 \le j < T
$$

$$
q = \mathrm{RMSNorm}_q\!\left(\sum_{n \in \{3, 19, 47, 63\}} Q_n \, \mathrm{RMSNorm}_n\!\big(h^{(n)}_{T-1}\big) + E_{q,\text{seg}}(s_{T-1})\right) \in \mathbb{R}^{512}
$$

- $X_l$ 专注于提供 $l$ 自己和左侧序列的语义信息。
- $q$ 专注于提供当前位置的检索意图和接下来准备生成的内容规划。

### 1.2 无衰减 RoPE 前缀和逆向注意力特征 $Y_l$（窗口 $W = 256$）

为使候选起点 $l$ 能够以严格 $O(1)$ 均摊开销“向右看清”后续生成的词序（类似保序多项式滚动哈希，同时彻底规避指数衰减系数 $\gamma^t$ 在 `float32` 前缀相减时的浮点下溢），采用**无衰减多频 RoPE 旋转前缀和**：

**（1）线性投影与旋转编码**：将每个位置的正向特征投影为逆向值向量 $V_j = W_{\text{rev}}\, X_j \in \mathbb{R}^{d_r}$（取 $d_r = 256$，无偏置），将 $256$ 个通道两两配对为 $128$ 个二维旋转平面，频率设为 $\theta_c = 10000^{-2c / d_r}\ (c = 0, \dots, 127)$。记位置 $t \in \mathbb{Z}$ 处的正交旋转算子 $\mathrm{RoPE}(v,\, t) \in \mathbb{R}^{256}$ 为：

$$\begin{bmatrix} \mathrm{RoPE}(v, t)_{2c} \\ \mathrm{RoPE}(v, t)_{2c+1} \end{bmatrix} = \begin{bmatrix} \cos(t\theta_c) & -\sin(t\theta_c) \\ \sin(t\theta_c) & \cos(t\theta_c) \end{bmatrix} \begin{bmatrix} v_{2c} \\ v_{2c+1} \end{bmatrix}$$

**（2）顺向维护 `float32` 前缀和表 $F$（每步增量 $O(1)$）**：

$$F(0) = 0 \in \mathbb{R}^{256}, \qquad F(t + 1) = F(t) + \mathrm{RoPE}(V_t,\, t), \qquad 0 \le t < T$$

由于正交旋转模长恒为 $1$（$\|\mathrm{RoPE}(V_t, t)\|_2 = \|V_t\|_2$），即使到 $T = 16384$，前缀和幅值也仅为 $\sim \sqrt{T} \approx 128$，在 `float32` 下无任何下溢或灾难性抵消（实测 $T = 16384$ 处相对误差仅 $8.3 \times 10^{-7}$）。

**（3）候选点 $O(1)$ 区间作差与相位归零得到 $Y_l$**：对任意候选起点 $l < T$，取可见边界内右窗口终点 $b_l = \min(l + W,\, T)$（固定 **$W = 256$**），做一次前缀差并施加逆旋转 $-l$ 将相对位移 $j - l$ 归零：

$$Y_l = \frac{1}{\sqrt{b_l - l}}\,\mathrm{RoPE}\!\Big(F(b_l) - F(l),\ -l\Big) = \frac{1}{\sqrt{b_l - l}}\sum_{j = l}^{b_l - 1} \mathrm{RoPE}\!\big(V_j,\ j - l\big) \in \mathbb{R}^{256}$$

> **为何根据隐藏层维度 $d = 512$ 选定 $W = 256$（$= d / 2$）**：
> - **子空间容量与信噪比定律**：在隐藏层维度 $d = 512$、逆向窗口维度 $d_r = 256$ 下，查询向量 $q$ 与逆向特征 $Y_l$ 对齐的有效匹配子空间自由度上限为 $d_{\text{eff}} = d_r = 256$。无衰减叠加 $W$ 个词时，区分两处仅相差 $\Delta$ 个词的候选信噪比满足 $z \approx \sqrt{\Delta \cdot d_{\text{eff}} / W}$：$W = 256$ 保证每个窗口词分到约 $1.0$ 个有效自由度，既能跨过真实代码/文档 $50 \sim 150$ 词的公共样板头抓出约 1 行（$\Delta \ge 9$ 词，$z \ge 3\sigma$）的后续分叉，又不会像 $W = 512$（$0.5$ 维/词）那样发生严重零空间混叠与跨段串扰。
> - **投机验证损益不对称性**：若两处候选连续 $\ge 256$ 个词完全相同、直到第 $256+$ 个词才分叉，即便挑中另一副本，底座并行验证也会单步无损收下全部 $256+$ 个相同前缀词并给出修正词，加速收益已完全吃满，无需牺牲近中距微改分辨力去盲目拉大窗口。

### 1.3 区间专用词特征 $H$ 与纯净区间算子 $\phi(a, b)$

在定终点阶段，通过 $\mathrm{GatedMLP}_{\text{span}}$ 将每个词映射为专用于跨度表示的 $d_{\text{span}} = 256$ 维特征（各通道分别提取语义内容与换行、引号、括号闭合、代码缩进、转义符等边界信号）：

$$H_j = \mathrm{GatedMLP}_{\text{span}}(X_j) \in \mathbb{R}^{256}, \qquad 0 \le j < T$$

另设可学习的上下文末尾哨兵向量 $h_{\text{eos}} \in \mathbb{R}^{256}$，记扩展序列 $\tilde{H}_j = H_j\ (j < T)$，$\tilde{H}_T = h_{\text{eos}}$。

对任意闭区间 $[a, b]$（$0 \le a \le b < T$），定义其**纯净区间特征** $\phi(a, b) \in \mathbb{R}^{512}$ 为区间均值与通道级最大值的拼接：

$$\mu(a, b) = \frac{1}{b - a + 1}\sum_{j = a}^{b} H_j \in \mathbb{R}^{256}, \qquad \omega(a, b) = \max_{a \le j \le b} H_j \in \mathbb{R}^{256}, \qquad \phi(a, b) = \big[\mu(a, b),\, \omega(a, b)\big] \in \mathbb{R}^{512}$$

$\mu(a, b)$ 与 $\omega(a, b)$ 严格只依赖 $[a, b]$ 内部 token，数值尺度与区间长度完全无关，且均可借助前缀和 `cumsum` 与前缀最大值 `cummax` 在 $O(1)$ 均摊时间内并行算出。

---

## 2. 三步定位流程

打分与分类全程在 `float32` 下计算。由最小复制长度 $r - l \ge k$（$k = 5$），合法起点与末尾词下标须满足：

$$0 \le l \le T - k, \qquad l + k - 1 \le e \le T - 1$$

### ① 同首词候选池筛选精确起点 $l$ 或 $\text{NO\_COPY}$（$C + 1$ 分类，$C \le K$）

利用刚从底座目标分布采样确定的首词 $t_0 \sim p_{\text{target}}(\cdot \mid x_{<T})$，通过倒排索引 `StartIndex` 在可见历史中直接检出满足 $\text{tokens}[l] = t_0$ 且 $0 \le l \le T - k$ 的候选位置，计算每个候选左侧与当前位置 $T$ 左侧的最长公共后缀长度 $\text{ov}_l$，按 $(\text{ov}_l,\, l)$ 降序保留前 $C$ 个候选（上限 $K = 128$）：

$$\mathcal{C} = \big\{(l_1, \text{ov}_1),\, \dots,\, (l_C, \text{ov}_C)\big\}, \qquad 0 \le C \le K$$

若 $T < k$ 或 $C = 0$（上文从未出现合法同首词），零 GPU 开销直接返回 $\text{NO\_COPY}$。

当 $C \ge 1$ 时，对每个候选 $l_j$（$j = 1, \dots, C$）拼入查询向量 $q$、正向特征 $X_{l_j}$、逆向窗口特征 $Y_{l_j}$ 与对数左重合特征 $\ln(1 + \text{ov}_j)$（共 $512 + 512 + 256 + 1 = 1281$ 维）：

$$s_j = \mathrm{GatedMLP}_{\text{start}}\!\Big(\big[q,\, X_{l_j},\, Y_{l_j},\, \ln(1 + \text{ov}_j)\big]\Big) \in \mathbb{R}^1, \qquad j = 1, \dots, C$$

取候选最高分 $s_{\max} = \max_{1 \le j \le C} s_j$ 与均分 $s_{\text{mean}} = \frac{1}{C}\sum_{j=1}^C s_j$，送入 $\mathrm{GatedMLP}_{\text{no}}$（输入 $512 + 2 = 514$ 维）计算不复制分数 $s_{\text{no}}$，拼成第 ① 步 logits：

$$s_{\text{no}} = \mathrm{GatedMLP}_{\text{no}}\!\big([q,\, s_{\max},\, s_{\text{mean}}]\big), \qquad z^{(1)} = [s_{\text{no}},\, s_1,\, \dots,\, s_C] \in \mathbb{R}^{C+1}$$

记类别概率 $p^{(1)} = \mathrm{softmax}\big(z^{(1)}\big)$，最优选 $\hat{c} = \arg\max z^{(1)}$：
- 若 $\hat{c} = 0$ 或置信度 $p^{(1)}_{\hat{c}} < \tau_{\text{start}}$（默认 $\tau_{\text{start}} = 0.6$）：当前步不复制，返回 $\text{NO\_COPY}$；
- 否则选中精确起点 $l = l_{\hat{c}}$。

### ② 基于块间纯净区间特征，选终点块 $m_r$（$M$ 分类，支持同块）

确定精确起点 $l$ 后（所在块 $m_l = \lfloor l / 32 \rfloor$，总块数 $M = \lceil T / 32 \rceil$），先将当前状态与起点正向特征融合为任务起点意图向量：

$$u = \mathrm{GatedMLP}_u\!\big([q,\, X_l]\big) \in \mathbb{R}^{256}$$

终点块最小合法块号为 $m_{\min} = \lfloor (l + k - 1) / 32 \rfloor$（当 $l \bmod 32 \le 32 - k$ 时 $m_{\min} = m_l$，即起点与终点可落在同一块内）。对任意候选块 $m \in \{m_l, \dots, M - 1\}$，其从 $l$ 起算的块内合法切片范围为：

$$a_m = \max(l,\, 32m), \qquad b_m = \min\big(32(m + 1),\, T\big) - 1$$

将从 $l$ 到块 $m$ 的纯净区间拆解为 **“前置必经段 + 候选块自身切片 + 块首尾边界”** 三部分：

1. **前置必经段特征 $\phi_{\text{pre}}(m) \in \mathbb{R}^{512}$**：
   - 当 $m = m_l$（**同块情形**）：前置为空，取可学习的同块标识向量 $\phi_{\text{same}} \in \mathbb{R}^{512}$；
   - 当 $m > m_l$（**跨块情形**）：取 $\phi(l,\, 32m - 1)$。若真实终点落在块 $m$，前置区间 $[l,\, 32m - 1]$ 必须完整无断点；一旦更早的块已出现断点，$\omega(l,\, 32m - 1)$ 会直接否决后续所有块。
2. **候选块纯净切片 $\phi_{\text{blk}}(m) = \phi(a_m,\, b_m) \in \mathbb{R}^{512}$**：同块时 $a_{m_l} = l$，只聚合 $[l,\, b_{m_l}]$，严格排除块内 $< l$ 的前缀噪声。
3. **块首尾边界特征 $[H_{a_m},\, \tilde{H}_{b_m + 1}] \in \mathbb{R}^{512}$**：兼顾刚好停在块末 $b_m$、断点词落在下一块开头 $b_m + 1$ 的边界情况。

拼接成 $1792$ 维向量计算第 ② 步 logits（$m < m_{\min}$ 置为 $-\infty$）：

$$z^{(2)}_m = \begin{cases} \mathrm{GatedMLP}_{r,\text{chunk}}\!\Big(\big[u,\, \phi_{\text{pre}}(m),\, \phi_{\text{blk}}(m),\, H_{a_m},\, \tilde{H}_{b_m + 1}\big]\Big), & m \ge m_{\min} \\ -\infty, & m < m_{\min} \end{cases}$$

选出终点块：

$$m_r = \arg\max_{0 \le m < M} z^{(2)}_m$$

### ③ 基于精确 $[l, e]$ 纯净区间特征，在块 $m_r$ 内选末尾词 $e$（$32$ 分类）

选中终点块 $m_r$ 后，记块内起点 $a_r = \max(l,\, 32 m_r)$、终点 $b_r = \min(32(m_r + 1),\, T) - 1$。复用第 ② 步已算出的前置聚合量（$m_r > m_l$ 时为 $[l,\, 32 m_r - 1]$ 的和与最大值）与块内切片 $[a_r,\, b_r]$ 的前缀 `cumsum` / `cummax` 合并，即可并行得到块内每个位置 $e = 32 m_r + t \in [a_r,\, b_r]$ 的精确纯净区间特征 $\phi(l, e) \in \mathbb{R}^{512}$。

再拼上停手交界处的两个词 $[H_e,\, \tilde{H}_{e + 1}]$（最后被复制的词 $H_e$ 与紧随其后第一个不复制的词 $\tilde{H}_{e + 1}$），共 $1280$ 维送入 $\mathrm{GatedMLP}_{r,\text{tok}}$：

$$z^{(3)}_t = \begin{cases} \mathrm{GatedMLP}_{r,\text{tok}}\!\Big(\big[u,\, \phi(l, e),\, H_e,\, \tilde{H}_{e + 1}\big]\Big), & l + k - 1 \le 32 m_r + t < T \\ -\infty, & \text{otherwise} \end{cases}$$

选出精确末尾词并输出左闭右开复制区间：

$$e = 32 m_r + \arg\max_{0 \le t < 32} z^{(3)}_t \quad \Longrightarrow \quad [l,\, r) = [l,\, e + 1)$$

---

## 3. PyTorch 核心结构参考

整个定位器参数量约 $27.7\text{M}$（`bf16` 下约 $53\text{ MB}$；其中历史投影 $P_n$ 与专用查询投影 $Q_n$ 各占 $10.5\text{M}$，打分与区间特征头仅 $6.7\text{M}$），前向逻辑与上述公式一一对应：

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

    # ① 候选池打分选精确起点 l 或 NO_COPY
    q = q.float()                                                                   # [512] 专用查询向量
    pos = torch.tensor([p for p, _ in candidates], dtype=torch.long, device=X.device)
    ov  = torch.tensor([o for _, o in candidates], dtype=torch.float32, device=X.device).log1p().unsqueeze(-1)
    X_l = X[pos].float()                                                            # [C, 512] 正向特征
    Y_l = self.query_rev_window(F_rev, pos, pos.new_full(pos.shape, T))             # [C, 256] 逆向特征 (W=256)

    s_cand = self.mlp_start(torch.cat([q.expand_as(X_l), X_l, Y_l, ov], dim=-1)).squeeze(-1)
    s_no   = self.mlp_no_copy(torch.cat([q, s_cand.max()[None], s_cand.mean()[None]]))
    prob, pick = torch.cat([s_no, s_cand], dim=0).softmax(dim=0).max(dim=0)
    if pick.item() == 0 or prob.item() < min_start_prob:
      return None
    l = int(pos[pick.item() - 1].item())

    # ② 构造从 l 起算的块级纯净区间特征，选终点块 m_r（兼容 m_r == m_l 同块）
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

    # ③ 在终点块 m_r 内构造精确 [l, e] 纯净区间特征，选末尾词 e
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

> **推理增量缓存与复杂度**：在线解码提交新词时，$X_t$、逆向 RoPE 前缀和 `F_rev[t + 1]`、区间特征 $H_t$ 以及满 $32$ 词块的 `(blk_sum, blk_max)` 均按步 $O(1)$ 追加更新（$16\text{K}$ 上下文总缓存仅约 $28\text{ MB}$）。普通不复制步仅执行第 ① 步（GPU 复杂度 $O(K) = O(1)$，耗时约 `0.05 ms`）；仅在触发复制时才执行第 ②、③ 步（$O(T / 32) + O(32)$，耗时约 `0.1 ms`），整体开销不足单次 27B 自回归前向（`~50 ms`）的千分之三。

---

## 4. 训练方案

训练时底座 `Qwen3.8-27B` 全程冻结，只更新 `CopyLocator` 参数。针对投机解码中 **“保守少猜零回滚，越界多猜必触发 48 层 GDN 状态回滚重放”** 的物理损益不对称性，设计起点长度加权软目标与终点非对称越界惩罚。

### 4.1 单个锚点的三项损失函数

设一个训练锚点的当前可见长度为 $T$，第 ① 步预测概率为 $p^{(1)} = \mathrm{softmax}\big(z^{(1)}\big) \in \mathbb{R}^{K+1}$（不足 $K$ 个候选的填充槽位在 softmax 前置为 $-\infty$）。

#### （1）负样本（当前位置不该复制）

仅监督第 ① 步选中第 $0$ 类（$\text{NO\_COPY}$），第 ②、③ 步不计算损失：

$$\mathcal{L}_{\text{start}} = -\ln p^{(1)}_0 = \mathrm{CE}\!\big(z^{(1)},\, 0\big)$$

#### （2）正样本起点损失（按可复制长度加权的软标签交叉熵 + 池外忽略）

设真实极长复制区间为 $[l, r)$，长度 $L^* = r - l \ge k$，末尾词 $e^* = r - 1$。对候选池中每个合法候选 $l_j\ (1 \le j \le C)$，计算其在可见边界 $T$ 内与当前目标序列 $\text{tokens}[T : T + L^*]$ 的最长连续前缀匹配长度 $L_j \in [1,\, \min(L^*,\, T - l_j)]$，记达到最小复制门槛的有效候选集为 $\mathcal{V} = \{j \mid L_j \ge k\}$：

- 若 $\mathcal{V} = \varnothing$（真起点因候选池截断未入池）：该锚点的 $\mathcal{L}_{\text{start}}$ 标记为 `ignore`（**严禁错标为 `0` 类 `NO_COPY`**）；
- 若 $\mathcal{V} \ne \varnothing$：按匹配长度平方分配软目标权重 $y_j = \dfrac{L_j^2}{\sum_{m \in \mathcal{V}} L_m^2}\ (j \in \mathcal{V})$，其余类目标为 $0$（唯一候选匹配时退化为标准硬交叉熵）：

$$\mathcal{L}_{\text{start}} = -\sum_{j \in \mathcal{V}} y_j \ln p^{(1)}_j$$

当候选池中同时存在“仅前几十词相同即分叉的次优源 $l_1$”与“整段完全匹配的极长源 $l_2$”时，该损失既以压倒性权重推着模型借助 $Y_l$ 选出极长源 $l_2$，又不会把前缀确实合法的 $l_1$ 当作纯错误样本造成梯度冲突。

#### （3）正样本终点损失（Teacher Forcing + 极长终点主 CE + 越界非对称重罚）

第 ②、③ 步直接喂入真实起点 $l$ 与真实终点块 $m_r^* = \lfloor e^* / 32 \rfloor$（块内真值偏移 $t^* = e^* \bmod 32$）。记第 ②、③ 步 softmax 分布分别为 $p^{(2)} = \mathrm{softmax}\big(z^{(2)}\big)$、$p^{(3)} = \mathrm{softmax}\big(z^{(3)}\big)$，定义超过真终点的**越界概率总和**：

$$P_{\text{over}}^{\text{blk}} = \sum_{m > m_r^*} p^{(2)}_m, \qquad P_{\text{over}}^{\text{tok}} = \sum_{t > t^*} p^{(3)}_t$$

在极长终点主交叉熵之上叠加越界重罚项（默认 $\lambda_{\text{over}} = 1.0$）：

$$\mathcal{L}_{r,\text{block}} = -\ln p^{(2)}_{m_r^*} - \lambda_{\text{over}}\ln\!\big(1 - P_{\text{over}}^{\text{blk}}\big)$$

$$\mathcal{L}_{r,\text{tok}} = -\ln p^{(3)}_{t^*} - \lambda_{\text{over}}\ln\!\big(1 - P_{\text{over}}^{\text{tok}}\big)$$

梯度机制分为三种情形：
- **精准命中极长终点 $e^*$（含长度 $> 512$ 的长段）**：$p(e^*) \to 1,\, P_{\text{over}} \to 0$，两项同时归零，全局唯一最优解锁死在 $e^*$，绝不会退化为“只敢抄 $k = 5$ 个词躺平”；
- **轻微保守偏短（$e < e^*$）**：仅受第一项平滑修正拉向 $e^*$，不触发第二项重罚；
- **越界多猜（$e > e^*$）**：同时承受主项与 $\frac{\lambda_{\text{over}}}{1 - P_{\text{over}}}$ 的放大重罚，使模型在边界模糊处宁可先收下安全前缀，也不冒险越界触发回滚。

### 4.2 Batch 级聚合方式（分母按有效样本数分别归一化）

设一个 mini-batch 中有效起点锚点数为 $N_{\text{start}}$（即全量采样锚点剔除 $\mathcal{V} = \varnothing$ 的池外正例后剩余数量），正样本总数为 $N_{\text{pos}}$，起点损失与两项终点损失分别归一化后相加：

$$\mathcal{L}_{\text{total}} = \frac{1}{N_{\text{start}}} \sum_{i \in \text{valid\_start}} \mathcal{L}_{\text{start}}^{(i)} + \frac{1}{N_{\text{pos}}} \sum_{i \in \text{pos}} \Big(\mathcal{L}_{r,\text{block}}^{(i)} + \mathcal{L}_{r,\text{tok}}^{(i)}\Big)$$

分开归一化的原因：$\mathcal{L}_{\text{start}}$ 需要正负样本共同校准触发门限，而 $\mathcal{L}_{r,\text{block}},\, \mathcal{L}_{r,\text{tok}}$ 仅在正样本上存在，除以 $N_{\text{pos}}$ 可保证调整正负采样比例时终点定位器的有效学习率恒定。

### 4.3 样本展开、难负采样与防穿越细节

1. **极长区间展开与短尾忽略区（`ignore`）**：`doc/1.md` 中的一条三元组 $(i, l, r)$，在每个偏移 $d \in \{0, 1, \dots, (r - l) - k\}$ 上展开为一个正样本（对应生成下标 $i + d$，目标区间 $[l + d,\, r)$）。区间末尾只剩 $1 \sim k - 1$（即 $1 \sim 4$）个词的位置虽不足最小复制长度 $k = 5$，但词面与上文一致——**既不当正样本也不当负样本，直接跳过不采样**。
2. **难负样本与同块/段首样本覆盖**：
   - 将复制刚结束、必须停手的交界位置 $i + (r - l)$ 高比例采为**难负样本**（逼迫第 ① 步抄完立即切回生成），配比取 **正样本 : 难负样本 : 普通负样本 $\approx 45 : 35 : 20$**；
   - 正样本内部确保充足覆盖**段首冷启动样本**（$d < k$，专门锤炼逆向特征 $Y_l$ 在低 `overlap` 下的消歧能力）与**同块样本**（$\lfloor (l + d) / 32 \rfloor = \lfloor (r - 1) / 32 \rfloor$，训练 $\phi_{\text{same}}$ 与同块切片分支）。
3. **严格防未来词穿越（三处硬截断）**：整条训练序列一次算出全长 $X$、前缀和表 $F$ 与 $H$ 后，对可见长度为 $T$ 的训练锚点：
   - **逆向窗口右边界截断**：计算 $Y_{l_j}$ 时右端点必须截断在 $b_{l_j} = \min(l_j + W,\, T)$，严禁读取全序列 $F$ 表中 $> T$ 的未来状态；
   - **终点尾块截断**：第 ② 步尾块 $m_{\text{tail}} = \lfloor (T - 1) / 32 \rfloor$ 仅聚合 $[a_{m_{\text{tail}}},\, T - 1]$；
   - **右边界哨兵替换**：第 ②、③ 步在右前瞻下标恰好等于 $T$（$b_m + 1 = T$ 或 $e + 1 = T$）时，**必须替换为哨兵向量 $h_{\text{eos}}$**，严禁读取 $H_T$，保证训练与在线推理逐位一致。

---

## 5. 无损验证、系统级联与底座改动

### 5.1 在线无损投机验证（兼容温度采样与贪心解码）及 `Copy Locator -> DSpark` 瀑布流

#### （1）兼容温度采样的单步无损复制验证

底座算完位置 $T - 1$ 后，先从目标分布 $p_0(\cdot) = p_{\text{target}}(\cdot \mid x_{<T})$（经温度 $\tau$、`top_p` / `top_k` 处理）直接采样得到位置 $T$ 的首词 $t_0 \sim p_0$，再调用 `StartIndex.candidates(t_0)` 与 `CopyLocator.locate(...)`：

- 由于 $t_0$ 在调用定位器前已严格按目标分布 $p_0$ 采出，且无论返回 $\text{NO\_COPY}$ 还是 $[l, r)$（满足 $\text{tokens}[l] = t_0$），位置 $T$ 最终提交的词恒为 $t_0$，因此**首词过滤零分布偏移，且 $d_0 = t_0$ 天然 $100\%$ 接受**。
- 若返回 $[l, r)$，记送验片段为 $d_{0 : L} = \text{tokens}[l : \min(r,\, l + B_{\max})]$（长 $L \ge k$），其中后续词 $d_1, \dots, d_{L-1}$ 构成确定性单点草稿分布 $q_i(v) = \mathbb{I}[v = d_i]$。先调用 `cache.snapshot()` 备份状态，将 $d_{0 : L}$ 一次性送入底座并行前向，得到各步目标分布 $p_i(\cdot) = p_{\text{target}}(\cdot \mid x_{<T}, d_{0 : i})\ (i = 1, \dots, L)$：
  - **温度采样（$\tau > 0$）**：对 $i = 1, \dots, L - 1$ 依次采样 $u_i \sim \mathrm{Uniform}(0, 1)$，按标准无损投机采样准则检验 $u_i < \min\big(1, p_i(d_i) / q_i(d_i)\big) = p_i(d_i)$。若在某步 $i$ 拒绝（已连续接受前 $a = i \in [1, L - 1]$ 个复制词），将 $p_i$ 中 $d_i$ 概率置零重归一化得到精确残差分布 $p'_i(v) = \mathbb{I}[v \ne d_i]\,\frac{p_i(v)}{1 - p_i(d_i)}$，从中采出修正词 $t_{\text{corr}} \sim p'_i$，调用 `cache.restore()` 回滚并提交 $d_{0 : a} + [t_{\text{corr}}]$；若全部通过（$a = L$），直接保留当前缓存，下一轮从末尾输出分布 $p_L(\cdot)$ 采样新首词。因 $p_i(d_i)\mathbb{I}[v = d_i] + (1 - p_i(d_i))p'_i(v) \equiv p_i(v)$，输出分布与原底座逐步温度采样**严格一致（$100\%$ 分布无损）**。
  - **贪心解码（$\tau = 0$ 特例）**：$p_i$ 退化为 $\arg\max$ 单点分布，上述检验自动简化为逐位核对 $d_i == \arg\max \text{logits}$，首个不一致处直接取 $\arg\max$ 作为修正词 $t_{\text{corr}}$。

#### （2）与 DSpark 串联无需独立 Router

- `Copy Locator` 判定 `NO_COPY` 仅需 `~0.05 ms` 且手握精确历史匹配证据，任何前置 Router 既省不下时间又会误杀真拷贝；
- 当 `Copy Locator` 返回 `NO_COPY` 时，直接串行交由轻量并行草稿器 `DSpark`（单次前向 `~1 ms`，自带置信度动态长度裁剪 $\hat{k} \in [0, \gamma]$，$\hat{k} = 0$ 时直接提交已采出的 $t_0$ 走单步解码），构成 **`Copy Locator -> DSpark`** 极简两级瀑布流，消除独立 Router 的维护成本与级联误判。

### 5.2 底座最小改动清单

| 文件 | 改动说明 |
|---|---|
| `src/cache.py` | 新增 `snapshot()` / `restore()`：备份与回滚 `seq_len` 以及 $48$ 层 GDN 的 `conv_state`、`recur_state`（$16$ 层 Full Attention 的 KV Cache 靠 `seq_len` 截断覆盖即可）；开 CoT 时 `max_len` 扩至 $16384$ 以上 |
| `src/gdn.py` | 分块三角求解 `chunk_gated_delta`：将 $48$ 层线性注意力多 token 串行递推加速为矩阵分块并行前向 |
| `src/qwen.py` | `Attention.forward`：多 token 并行验证时构造右下对齐因果掩码；`Model.forward`：顺手返回第 $3, 19, 47, 63$ 层隐状态 |
| 新增文件 | `src/copy_locator/{model.py, candidates.py, decoder.py}`、`src/spec_decode.py`、`train/train_copy_locator.py`、`scripts/bench_copy.py` |

---

## 6. 实验复盘与架构演进记录

### 6.1 第一版实测与根因定位（纯块均值选起点，1 轮 / 103 步）

- **实验条件**：仅使用 `code` 训练集 `821` 条（含 `273` 条 CoT），`epochs = 1`、`grad_accum = 8`（共 `103` 步），在 `8` 道 `code` 测试题上关闭 CoT、无损验证（`max_copy_len = 128`）。
- **实测结果**：官方 EM 均为 `8/8` 且逐 token 与 AR 一致，但 Locator 纯解码仅 `10.4 tok/s`（AR `19.6 tok/s`，CopyGen `124.8 tok/s`），草稿接受率仅 **`2.9%`**（送验 `104,694` 词仅接受 `2,984` 词）。
- **逐 token 回放诊断出的核心根因**：
  1. **纯净区间定终点分支已学出规律**：训练日志中 `r_block` 迅速降至 `0.09 ~ 0.36`，回放中能精准停在代码块右边界。
  2. **32 词块均值盲选起点块彻底崩盘**：测试题回放中起点块命中率仅 `1.5%`（`7 / 466`），32 词求平均抹平了首词身份与局部词序，把目标字典块误判为同文件另一函数的相似字典，反复触发假阳性验证与 GDN 回滚。
  3. **训练欠拟合 + GDN 串行瓶颈**：仅更新 `103` 步，且 48 层 GDN 为纯 Python 逐词循环，单次验证耗时高达 `320 ms`。

---

### 6.2 第二版核心改动与复测结果（同首词候选池 + 12 轮训练 + 分块 GDN）

针对第一版根因完成了四项升级：（1）起点改为 `StartIndex` 同首词候选池 + `[q, X_l, ln(1 + ov)]` 与 `NO_COPY` 联合打分（`min_start_prob = 0.6`）；（2）离线缓存四层 BF16 特征训满 `12` 轮（`1,176` 步）；（3）接入分块并行 `src/gdn.py`；（4）放开 `max_copy_len = 512`。

**复测结果（仅用 `code` 单域训练权重，在其余三域为零样本迁移）**：

| 评测集（关 CoT，各 8 题） | CopyGen 纯解码 / 总耗时 / 接受率 | Locator 纯解码 / 总耗时 / 接受率 |
|---|---|---|
| `code`（域内） | `201.6 tok/s` / `31.6s` / `95.2%` | **`222.0 tok/s` / `29.2s` / `100.0%`**（4,724/4,724） |
| `literature`（零样本） | `198.5 tok/s` / `11.9s` / `88.7%` | **`548.9 tok/s` / `6.9s` / `66.5%`** |
| `random_sequence`（零样本） | `123.6 tok/s` / `72.7s` / `99.5%` | **`155.0 tok/s` / `59.7s` / `93.2%`** |
| `contract`（零样本） | `88.7 tok/s` / `113.3s` / `78.7%` | **`120.7 tok/s` / `87.1s` / `11.5%`** |
| **32 题合计（EM 均 23/32）** | 总耗时 `229.6s` | **总耗时 `183.0s`**（快 `25.5%`） |

- 开 CoT 评测中，`code` 8/8 EM 全部命中；`literature` 开 CoT 阶段 Locator 纯解码达 `107.5 tok/s`（CopyGen `73.1 tok/s`，快 `47%`）。

---

### 6.3 第三版演进（当前设计：逆向 RoPE 特征 $Y_l$ + 非对称越界损失 + 全域合训）

在第二版基础上，针对冷启动长公共前缀消歧、边界风险控制与多域泛化完成本文档当前定型设计：

1. **引入无衰减 RoPE 前缀和逆向特征 $Y_l$（$W = 256$）并扩充候选池（$K = 128$）**：弥补因果正向特征 $X_l$ 看不见 $l$ 右侧下文的盲区，解决相似模板在冷启动（`overlap = 0/1`）或长公共样板头后的分叉消歧，同时消除 `MAX_CANDIDATES = 32` 的段首截断。
2. **长度加权软起点目标 + 终点越界非对称重罚**：对池外正例置 `ignore` 消除假负例错标，对多合法候选按 $L_j^2$ 软加权；在两级终点损失中加入 $-\lambda_{\text{over}} \ln(1 - P_{\text{over}})$，严惩触发 GDN 回滚的越界预测而不诱发短前缀躺平。
3. **四域 + CoT 联合训练（压低 `random_sequence` 占比）**：结合 `doc/1.md` 的三层确定性选源（`prompt > cot > output`、左重合度、最靠后位置），统一学习 JSON 转义停手边界与 CoT 短引用边界。
