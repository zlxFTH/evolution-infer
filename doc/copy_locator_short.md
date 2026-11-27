# Copy Locator Pipeline（极简版）

## 0. 超参与基础算子

- 常量：$d_{\text{model}} = 5120$，$d = 512$，$d_{\text{span}} = 256$，$B = 32$，$k = 5$，$\mathcal{L} = \{3, 19, 47, 63\}$
- 序列：$T$ 为当前可见长度，$M = \lceil T / 32 \rceil$，查询下标 $T - 1$，合法范围 $0 \le l \le T - k$、$l + k - 1 \le e \le T - 1$，$r = e + 1$
- $\mathrm{GatedMLP}(d_{\text{in}}, d_{\text{hid}}, d_{\text{out}})$：参数 $\gamma \in \mathbb{R}^{d_{\text{in}}}$，$W_{\text{gate}}, W_{\text{up}} \in \mathbb{R}^{d_{\text{hid}} \times d_{\text{in}}}$，$W_{\text{down}} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{hid}}}$

$$\mathrm{GatedMLP}(z) = W_{\text{down}}\Big(\mathrm{SiLU}\big(W_{\text{gate}}\,\mathrm{RMSNorm}(z)\big) \odot \big(W_{\text{up}}\,\mathrm{RMSNorm}(z)\big)\Big)$$

---

## 1. 参数清单（总计约 $17.9\text{M}$）

| 模块 / 参数 | 形状 / 规格 | 作用 |
|---|---|---|
| `proj_norms[0..3]` | $4 \times \mathrm{RMSNorm}(5120)$ | 四层隐状态归一化 |
| `projs[0..3]` ($P_n$) | $4 \times [512, 5120]$ | 四层特征降维 |
| `seg_emb` ($E_{\text{seg}}$) | $[3, 512]$ | 阶段嵌入（`0: prompt, 1: cot, 2: output`） |
| `out_norm` | $\mathrm{RMSNorm}(512)$ | 融合后归一化 |
| `mlp_l_chunk` | $\mathrm{GatedMLP}(1024, 512, 1)$ | 起点块打分 |
| `mlp_no_copy` | $\mathrm{GatedMLP}(514, 256, 1)$ | `NO_COPY` 打分 |
| `mlp_l_tok` | $\mathrm{GatedMLP}(1024, 512, 1)$ | 起点块内词打分 |
| `mlp_span` | $\mathrm{GatedMLP}(512, 512, 256)$ | 区间专用词特征映射 |
| `mlp_u` | $\mathrm{GatedMLP}(1024, 512, 256)$ | 查询与起点融合 |
| `phi_same` ($\phi_{\text{same}}$) | $[512]$ | 同块前置占位向量 |
| `h_eos` ($h_{\text{eos}}$) | $[256]$ | 右边界越界哨兵（下标 $T$） |
| `mlp_r_chunk` | $\mathrm{GatedMLP}(1792, 512, 1)$ | 终点块打分 |
| `mlp_r_tok` | $\mathrm{GatedMLP}(1280, 512, 1)$ | 终点块内词打分 |

---

## 2. Forward Pipeline（全程 `float32`）

### Step 0：特征构建

输入：$h^{(n)} \in \mathbb{R}^{T \times 5120}\ (n \in \mathcal{L})$，$s \in \{0, 1, 2\}^T$

$$X_j = \mathrm{RMSNorm}\!\left(\sum_{n \in \mathcal{L}} P_n\,\mathrm{RMSNorm}_n\!\big(h^{(n)}_j\big) + E_{\text{seg}}(s_j)\right) \in \mathbb{R}^{512}, \qquad X \in \mathbb{R}^{T \times 512}, \quad q = X_{T-1} \in \mathbb{R}^{512}$$

$$C_m = \frac{1}{\min(32(m + 1), T) - 32m} \sum_{j = 32m}^{\min(32(m + 1),\, T) - 1} X_j \in \mathbb{R}^{512}, \qquad C \in \mathbb{R}^{M \times 512}$$

$$H = \mathrm{GatedMLP}_{\text{span}}(X) \in \mathbb{R}^{T \times 256}, \qquad \tilde{H} = [H;\, h_{\text{eos}}] \in \mathbb{R}^{(T + 1) \times 256}$$

纯净区间算子（$0 \le a \le b < T$）：

$$\mu(a, b) = \frac{1}{b - a + 1}\sum_{j = a}^{b} H_j \in \mathbb{R}^{256}, \qquad \omega(a, b) = \max_{a \le j \le b} H_j \in \mathbb{R}^{256}, \qquad \phi(a, b) = \big[\mu(a, b),\, \omega(a, b)\big] \in \mathbb{R}^{512}$$

### Step 1：选起点块 $m_l$ 或 $\text{NO\_COPY}$（$T < k$ 直接返回 $\text{NO\_COPY}$）

$$s_m = \begin{cases} \mathrm{GatedMLP}_{l,\text{chunk}}\big([q,\, C_m]\big), & 32m \le T - k \\ -\infty, & 32m > T - k \end{cases} \qquad ([q, C_m] \in \mathbb{R}^{1024},\ s \in \mathbb{R}^{M})$$

$$s_{\text{no}} = \mathrm{GatedMLP}_{\text{no}}\!\big([q,\, \max_{32m \le T-k} s_m,\, \operatorname{mean}_{32m \le T-k} s_m]\big) \in \mathbb{R}^{1} \qquad (\text{输入} \in \mathbb{R}^{514})$$

$$z^{(1)} = [s_{\text{no}},\, s_0,\, \dots,\, s_{M-1}] \in \mathbb{R}^{M+1}, \qquad \arg\max z^{(1)} = 0 \Rightarrow \text{NO\_COPY},\quad \text{否则}\ m_l = \arg\max z^{(1)} - 1$$

### Step 2：块 $m_l$ 内选精确起点 $l$

$$z^{(2)}_t = \begin{cases} \mathrm{GatedMLP}_{l,\text{tok}}\big([q,\, X_{32 m_l + t}]\big), & 32 m_l + t \le T - k \\ -\infty, & 32 m_l + t > T - k \end{cases} \qquad ([q, X_p] \in \mathbb{R}^{1024},\ z^{(2)} \in \mathbb{R}^{32})$$

$$l = 32 m_l + \arg\max_{0 \le t < 32} z^{(2)}_t$$

### Step 3：选终点块 $m_r$

$$u = \mathrm{GatedMLP}_u\big([q,\, X_l]\big) \in \mathbb{R}^{256} \qquad ([q, X_l] \in \mathbb{R}^{1024})$$

令 $m_{\min} = \lfloor (l + k - 1) / 32 \rfloor$；对每个块 $m \in \{m_l, \dots, M - 1\}$，令 $a_m = \max(l, 32m)$，$b_m = \min(32(m + 1), T) - 1$：

$$\phi_{\text{pre}}(m) = \begin{cases} \phi_{\text{same}}, & m = m_l \\ \phi(l,\, 32m - 1), & m > m_l \end{cases} \in \mathbb{R}^{512}, \qquad \phi_{\text{blk}}(m) = \phi(a_m,\, b_m) \in \mathbb{R}^{512}$$

$$z^{(3)}_m = \begin{cases} \mathrm{GatedMLP}_{r,\text{chunk}}\!\Big(\big[u,\, \phi_{\text{pre}}(m),\, \phi_{\text{blk}}(m),\, H_{a_m},\, \tilde{H}_{b_m + 1}\big]\Big), & m \ge m_{\min} \\ -\infty, & m < m_{\min} \end{cases} \qquad (\text{输入} \in \mathbb{R}^{1792},\ z^{(3)} \in \mathbb{R}^{M})$$

$$m_r = \arg\max_{0 \le m < M} z^{(3)}_m$$

### Step 4：块 $m_r$ 内选精确末尾词 $e$

对 $t \in \{0, \dots, 31\}$，令 $e = 32 m_r + t$：

$$z^{(4)}_t = \begin{cases} \mathrm{GatedMLP}_{r,\text{tok}}\!\Big(\big[u,\, \phi(l, e),\, H_e,\, \tilde{H}_{e + 1}\big]\Big), & l + k - 1 \le e < T \\ -\infty, & \text{otherwise} \end{cases} \qquad (\text{输入} \in \mathbb{R}^{1280},\ z^{(4)} \in \mathbb{R}^{32})$$

$$e = 32 m_r + \arg\max_{0 \le t < 32} z^{(4)}_t \quad \Longrightarrow \quad [l,\, r) = [l,\, e + 1)$$

---

## 3. Training Loss（Teacher Forcing）

- 负样本：$\mathcal{L}_1 = \mathrm{CE}\!\big(z^{(1)},\, 0\big)$
- 正样本（真值 $[l, r)$，$e = r - 1$，$m_l = \lfloor l / 32 \rfloor$，$m_r = \lfloor e / 32 \rfloor$）：

$$\mathcal{L}_1 = \mathrm{CE}\!\big(z^{(1)},\, 1 + m_l\big), \quad \mathcal{L}_2 = \mathrm{CE}\!\big(z^{(2)},\, l \bmod 32\big), \quad \mathcal{L}_3 = \mathrm{CE}\!\big(z^{(3)},\, m_r\big), \quad \mathcal{L}_4 = \mathrm{CE}\!\big(z^{(4)},\, e \bmod 32\big)$$

- Batch 聚合（共 $N$ 个锚点，其中 $N_{\text{pos}}$ 个正样本）：

$$\mathcal{L}_{\text{total}} = \frac{1}{N}\sum_{i = 1}^{N} \mathcal{L}_1^{(i)} + \frac{1}{N_{\text{pos}}}\sum_{i \in \text{pos}} \Big(\mathcal{L}_2^{(i)} + \mathcal{L}_3^{(i)} + \mathcal{L}_4^{(i)}\Big)$$
