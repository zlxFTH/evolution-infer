# Copy Locator 设计说明

> 对齐数据格式：`doc/1.md`（上下文按 `prompt -> cot -> output` 拼接，只存极长不交三元组 $(i, l, r)$，强制最小复制长度 $r - l \ge 3$）

## 核心思路

设当前可见上下文共有 $T$ 个 token（下标 $0, 1, \dots, T - 1$），底座模型刚算完位置 $T - 1$。Copy Locator 根据当前状态判断：接下来从位置 $T$ 开始，能否直接从可见历史 $[0, T)$ 中整段复制某个区间 $[l, r)$。

为了不限制最大拷贝长度、同时避免对全部 $T$ 个词逐一精细打分，起点 $l$ 和终点 $r$ 均采用 **“先按 $32$ 词一块选块，再在块内 $32$ 选 $1$”** 的两级定位，共分四步：

```text
27B 四层隐状态 (Layer 3, 19, 47, 63)
        │
        ▼
  特征序列 X [T, 512] ──► 每 32 词求均值 ──► 块特征 C [M, 512]
        │
        ├── ① 选起点块 m_l 或 NO_COPY   (M + 1 分类，两层 MLP)
        ├── ② 在块 m_l 内选起点词 l     (32 分类，两层 MLP)
        ├── ③ 融入 l 信息，选终点块 m_r (M 分类，两层 MLP)
        └── ④ 在块 m_r 内选末尾词 e     (32 分类，两层 MLP) ──► 输出 [l, e + 1)
```

两个核心设计原则：

- **四步统一用两层 $\mathrm{PairMLP}$ 打分**：给定查询向量 $q \in \mathbb{R}^{512}$ 与候选向量 $k \in \mathbb{R}^{512}$，拼接 $[q,\, k,\, q \odot k] \in \mathbb{R}^{1536}$（其中 $q \odot k$ 提供逐通道匹配信号，$[q, k]$ 保留各自原始语义），经两层 MLP（$W_1 \in \mathbb{R}^{256 \times 1536},\, W_2 \in \mathbb{R}^{1 \times 256}$）直接输出标量分数；对候选矩阵 $K \in \mathbb{R}^{N \times 512}$ 批量打分记为 $\mathrm{PairMLP}(q, K) \in \mathbb{R}^N$：

$$\mathrm{PairMLP}(q, k) = W_2 \, \mathrm{SiLU}\!\big(W_1 [q,\, k,\, q \odot k] + b_1\big) + b_2 \in \mathbb{R}$$

- **不引入手工相对距离偏置**：选终点时，已选起点词特征 $X_l$ 与起点块特征 $C_{m_l}$ 已作为输入送入两层 MLP，且底座隐状态自身包含位置演化信息；不再人为假设“离 $l$ 近就一定更好”，也不做对数距离分桶。

---

## 1. 特征构建（$X$ 与块均值 $C$）

从冻结的 `Qwen3.8-27B`（共 $64$ 层，隐状态维度 $5120$）抽取第 $\mathcal{L} = \{3, 19, 47, 63\}$ 层隐状态 $h^{(n)}_j \in \mathbb{R}^{5120}$（分别对应浅层词面、中层局部结构、深层任务意图、最终输出前特征）。

对每个可见位置 $j \in \{0, \dots, T - 1\}$，四层特征各自降维到 $512$ 维后相加，叠加上下文阶段嵌入 $E_{\text{seg}}(s_j)$（$s_j \in \{0\text{: prompt},\, 1\text{: cot},\, 2\text{: output}\}$），再经 $\mathrm{RMSNorm}$ 归一化：

$$X_j = \mathrm{RMSNorm}\!\left(\sum_{n \in \{3, 19, 47, 63\}} P_n \, \mathrm{RMSNorm}_n\!\big(h^{(n)}_j\big) + E_{\text{seg}}(s_j)\right) \in \mathbb{R}^{512}$$

其中 $P_n \in \mathbb{R}^{512 \times 5120}$。由此得到三个核心量：

- **词级特征矩阵** $X \in \mathbb{R}^{T \times 512}$：推理时只需按步追加缓存 $512$ 维的 $X_j$（$8\text{K}$ 上下文仅占约 $8\text{ MB}$），无需缓存底座 $5120$ 维隐状态。
- **当前查询向量** $q = X_{T-1} \in \mathbb{R}^{512}$。
- **块均值矩阵** $C \in \mathbb{R}^{M \times 512}$：每 $32$ 个词切一块，共 $M = \lceil T / 32 \rceil$ 块；第 $m$ 块只对 $< T$ 的真实可见词求均值（尾块不补零）：

$$C_m = \frac{1}{\min(32(m + 1), T) - 32m} \sum_{j = 32m}^{\min(32(m + 1),\, T) - 1} X_j \in \mathbb{R}^{512}, \qquad m = 0, \dots, M - 1$$

---

## 2. 四步定位流程

打分与分类全程在 `float32` 下计算。为避免半开区间右端点 $r$ 落到 $T$ 越界，第 ③、④ 步改为预测**最后一个被复制的词下标 $e = r - 1$**，最终输出 $[l,\, e + 1)$。

由最小复制长度 $r - l \ge 3$（即 $e - l \ge 2$），合法下标须满足：

$$0 \le l \le T - 3, \qquad l + 2 \le e \le T - 1$$

若 $T < 3$，直接返回 $\text{NO\_COPY}$。

### ① 选起点块 $m_l$ 或 $\text{NO\_COPY}$（$M + 1$ 分类）

先对所有历史块打分，并将块起点 $32m > T - 3$ 的非法块置为 $-\infty$：

$$s_m = \begin{cases} \mathrm{PairMLP}_{l,\text{chunk}}(q, C_m), & 32m \le T - 3 \\ -\infty, & 32m > T - 3 \end{cases}$$

在合法块集合 $\{m \mid 32m \le T - 3\}$ 上取最高分 $s_{\max} = \max_m s_m$ 与均分 $s_{\text{mean}} = \operatorname{mean}_m s_m$（使网络感知全局是否存在高分匹配块），经两层 $\mathrm{MLP}_{\text{no}}$（$514 \to 128 \to \mathrm{SiLU} \to 1$）计算不复制分数 $s_{\text{no}}$，拼成第 ① 步 logits：

$$s_{\text{no}} = \mathrm{MLP}_{\text{no}}\!\big([q,\, s_{\max},\, s_{\text{mean}}]\big), \qquad z^{(1)} = [s_{\text{no}},\, s_0,\, \dots,\, s_{M-1}] \in \mathbb{R}^{M+1}$$

- 若 $\arg\max z^{(1)} = 0$：判定当前步不复制，直接返回 $\text{NO\_COPY}$；
- 否则选中起点块 $m_l = \arg\max z^{(1)} - 1$。

### ② 在块 $m_l$ 内选精确起点 $l$（$32$ 分类）

对块 $m_l$ 内的偏移 $k \in \{0, \dots, 31\}$（对应全局下标 $32 m_l + k$）逐位打分：

$$z^{(2)}_k = \begin{cases} \mathrm{PairMLP}_{l,\text{tok}}(q, X_{32 m_l + k}), & 32 m_l + k \le T - 3 \\ -\infty, & 32 m_l + k > T - 3 \end{cases}$$

选出精确起点：

$$l = 32 m_l + \arg\max_{0 \le k < 32} z^{(2)}_k$$

### ③ 融合 $l$ 信息，选终点块 $m_r$（$M$ 分类）

将当前查询 $q$、已选起点词特征 $X_l$ 与起点块特征 $C_{m_l}$ 拼接为 $1536$ 维，经两层 $\mathrm{MLP}_{\text{end}}$（$1536 \to 512 \to \mathrm{SiLU} \to 512$）融合成终点查询向量：

$$q_{\text{end}} = \mathrm{MLP}_{\text{end}}\!\big([q,\, X_l,\, C_{m_l}]\big) \in \mathbb{R}^{512}$$

直接用两层 $\mathrm{PairMLP}$ 对所有块打分（不加任何距离项），并将不满足 $e \ge l + 2$ 的前缀块置为 $-\infty$：

$$z^{(3)}_m = \begin{cases} \mathrm{PairMLP}_{r,\text{chunk}}(q_{\text{end}}, C_m), & m \ge \lfloor (l + 2) / 32 \rfloor \\ -\infty, & m < \lfloor (l + 2) / 32 \rfloor \end{cases}$$

选出终点块：

$$m_r = \arg\max_{0 \le m < M} z^{(3)}_m$$

### ④ 在块 $m_r$ 内选末尾词 $e$（$32$ 分类）

对块 $m_r$ 内的偏移 $k \in \{0, \dots, 31\}$（对应全局下标 $32 m_r + k$）逐位打分：

$$z^{(4)}_k = \begin{cases} \mathrm{PairMLP}_{r,\text{tok}}(q_{\text{end}}, X_{32 m_r + k}), & l + 2 \le 32 m_r + k < T \\ -\infty, & \text{otherwise} \end{cases}$$

选出末尾词并输出半开复制区间：

$$e = 32 m_r + \arg\max_{0 \le k < 32} z^{(4)}_k \quad \Longrightarrow \quad [l,\, r) = [l,\, e + 1)$$

---

## 3. PyTorch 核心结构参考

整个定位器参数量约 $13\text{M}$（`bf16` 下约 $26\text{ MB}$），上述公式与如下代码一一对应：

```python
class PairMLP(nn.Module):
  def __init__(self, d: int = 512, h: int = 256):
    super().__init__()
    self.net = nn.Sequential(nn.Linear(d * 3, h), nn.SiLU(), nn.Linear(h, 1))

  def forward(self, q: torch.Tensor, K: torch.Tensor):  # q: [512], K: [N, 512]
    q = q.expand_as(K)
    return self.net(torch.cat([q, K, q * K], dim=-1)).squeeze(-1)  # [N]


class CopyLocator(nn.Module):
  def __init__(self, d_model: int = 5120, d: int = 512):
    super().__init__()
    self.proj_norms = nn.ModuleList([RMSNorm(d_model) for _ in range(4)])
    self.projs      = nn.ModuleList([nn.Linear(d_model, d, bias=False) for _ in range(4)])
    self.seg_emb    = nn.Embedding(3, d)  # 0: prompt, 1: cot, 2: output
    self.out_norm   = RMSNorm(d)

    self.mlp_l_chunk = PairMLP(d, 256)
    self.mlp_no_copy = nn.Sequential(nn.Linear(d + 2, 128), nn.SiLU(), nn.Linear(128, 1))
    self.mlp_l_tok   = PairMLP(d, 256)

    self.mlp_end_q   = nn.Sequential(nn.Linear(d * 3, d), nn.SiLU(), nn.Linear(d, d))
    self.mlp_r_chunk = PairMLP(d, 256)
    self.mlp_r_tok   = PairMLP(d, 256)

  def encode_step(self, hs_4: list[torch.Tensor], seg_ids: torch.Tensor):
    x = sum(proj(norm(h)) for norm, proj, h in zip(self.proj_norms, self.projs, hs_4))
    return self.out_norm(x + self.seg_emb(seg_ids))  # [seq_len, 512]

  @staticmethod
  def get_block(X: torch.Tensor, m: int):
    blk = X[32 * m : min(32 * (m + 1), X.shape[0])]
    return F.pad(blk, (0, 0, 0, 32 - blk.shape[0])) if blk.shape[0] < 32 else blk

  @torch.no_grad()
  def locate(self, X: torch.Tensor):  # X: [T, 512] (float32)
    T = X.shape[0]
    if T < 3:
      return None

    M = (T + 31) // 32
    C = torch.stack([X[32 * m : min(32 * (m + 1), T)].mean(dim=0) for m in range(M)])
    q = X[T - 1]

    # ① 选起点块 m_l 或 NO_COPY
    s_chunk = self.mlp_l_chunk(q, C)
    valid_m = (torch.arange(M, device=X.device) * 32) <= (T - 3)
    s_chunk = s_chunk.masked_fill(~valid_m, float("-inf"))
    s_valid = s_chunk[valid_m]
    s_no = self.mlp_no_copy(torch.cat([q, s_valid.max()[None], s_valid.mean()[None]]))
    pick = int(torch.cat([s_no, s_chunk]).argmax().item())
    if pick == 0:
      return None
    m_l = pick - 1

    # ② 块内选精确起点 l
    pos_l = 32 * m_l + torch.arange(32, device=X.device)
    logits_2 = self.mlp_l_tok(q, self.get_block(X, m_l)).masked_fill(pos_l > T - 3, float("-inf"))
    l = 32 * m_l + int(logits_2.argmax().item())

    # ③ 融合 l 信息，选终点块 m_r
    q_end = self.mlp_end_q(torch.cat([q, X[l], C[m_l]]))
    m_idx = torch.arange(M, device=X.device)
    logits_3 = self.mlp_r_chunk(q_end, C).masked_fill(m_idx < (l + 2) // 32, float("-inf"))
    m_r = int(logits_3.argmax().item())

    # ④ 块内选末尾词 e，返回 [l, e + 1)
    pos_r = 32 * m_r + torch.arange(32, device=X.device)
    bad_r = (pos_r < l + 2) | (pos_r >= T)
    logits_4 = self.mlp_r_tok(q_end, self.get_block(X, m_r)).masked_fill(bad_r, float("-inf"))
    e = 32 * m_r + int(logits_4.argmax().item())
    return l, e + 1
```

---

## 4. 训练方案

训练时底座 `Qwen3.8-27B` 全程冻结，只更新 `CopyLocator` 参数。

### 损失函数

- **负样本（$\text{NO\_COPY}$ 锚点）**：仅监督第 ① 步选中第 $0$ 类：

$$\mathcal{L}_{\text{neg}} = \mathrm{CE}\!\big(z^{(1)},\, 0\big)$$

- **正样本（真实区间 $[l, r)$，记 $e = r - 1$）**：第 ②、③、④ 步采用 **Teacher Forcing**（直接代入真实 $m_l = \lfloor l / 32 \rfloor$、$l$、$m_r = \lfloor e / 32 \rfloor$，避免训练初期起点预测偏差干扰后续步骤），四步交叉熵直接相加：

$$\mathcal{L}_{\text{pos}} = \mathrm{CE}\!\big(z^{(1)},\, 1 + \lfloor l / 32 \rfloor\big) + \mathrm{CE}\!\big(z^{(2)},\, l \bmod 32\big) + \mathrm{CE}\!\big(z^{(3)},\, \lfloor e / 32 \rfloor\big) + \mathrm{CE}\!\big(z^{(4)},\, e \bmod 32\big)$$

### 样本展开与防穿越细节

1. **极长区间展开**：`doc/1.md` 中的一条三元组 $(i, l, r)$，在每个偏移 $d \in \{0, 1, \dots, (r - l) - 3\}$ 上展开为一个正样本（对应生成下标 $i + d$，目标复制区间 $[l + d,\, r)$）。末尾剩余长度 $< 3$ 的最后 $1 \sim 2$ 个位置既不当正样本也不当负样本，直接跳过。
2. **正负样本配比**：重点将“复制刚结束、应当停手”的交界位置 $i + (r - l)$ 采为**难负样本**（防止推理时抄过头），其余普通非复制位置随机下采样。每条序列抽约 $32 \sim 64$ 个训练锚点，比例取 **正样本 : 难负样本 : 普通负样本 $\approx 45 : 35 : 20$**。
3. **尾块防未来词穿越**：整条序列一次前向算出完整 $X$ 后，对可见长度为 $T$ 的训练锚点，前 $M - 1$ 个满块可直接复用预计算的块均值，但尾块 $m_{\text{tail}} = \lfloor (T - 1) / 32 \rfloor$ **必须只对 $\{X_{32 m_{\text{tail}}}, \dots, X_{T-1}\}$ 求均值**，严禁混入 $\ge T$ 的未来词，以保证训练与在线推理严格一致。

---

## 5. 无损验证与底座改动

### 在线推理闭环（$100\%$ 贪心无损）

1. 底座刚算完位置 $T - 1$ 时，自身的 `lm_head` 已免费给出位置 $T$ 的贪心预测词 $t_0 = \arg\max \text{logits}_{T-1}$。
2. 调用 `CopyLocator.locate(X[:T])`：
   - 若返回 $\text{NO\_COPY}$，或预测的复制首词 $\text{tokens}[l] \ne t_0$（首词即与底座贪心预测矛盾），**零验证开销直接放弃复制**，正常输出 $t_0$ 进入下一步；
   - 若 $\text{tokens}[l] = t_0$，先调用 `cache.snapshot()` 备份缓存状态，将候选片段 $\text{tokens}[l : r]$（长 $L = r - l$）一次性送入底座做单次并行前向，逐位核对贪心 $\arg\max$：设连续通过验证的复制词数为 $k$（$1 \le k \le L$），则一次性接受这 $k$ 个复制词 **外加验证顺手算出的第 $k + 1$ 个新贪心词**，并用 `cache.restore()` 将多余步数的缓存状态回滚到实际接受长度。

### 底座最小改动清单

| 文件 | 改动说明 |
|---|---|
| `src/cache.py` | 新增 `snapshot()` / `restore()`：备份与回滚 `seq_len` 以及 $48$ 层 GDN 的 `conv_state`、`recur_state`（$16$ 层 Full Attention 的 KV Cache 靠 `seq_len` 截断覆盖即可，无需拷贝）；开 CoT 时 `max_len` 扩至 $16384$ |
| `src/qwen.py` | `Attention.forward`：多 token 并行验证（`not cache.is_prefill and seq_len > 1`）时构造右下对齐因果掩码；`Model.forward`：顺手返回第 $3, 19, 47, 63$ 层隐状态 |
| 新增文件 | `src/copy_locator/{model.py, decoder.py}`、`train/train_copy_locator.py`、`misc/bench_copy.py` |
