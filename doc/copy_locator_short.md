# Copy Locator Pipeline（极简版）

## 0. 超参与基础算子

- 常量：$d_{\text{model}} = 5120$，$d = 512$，$d_r = 256$，$W = 256$，$d_{\text{span}} = 256$，$B = 32$，$k = 5$，$K = 128$，$\tau_{\text{start}} = 0.6$，$\lambda_{\text{over}} = 1.0$，$\mathcal{L} = \{3, 19, 47, 63\}$
- 序列：$T$ 为当前可见长度，$M = \lceil T / 32 \rceil$，查询下标 $T - 1$，由 $\text{logits}_{T-1}$ 按目标采样分布采出首词 $t_0 \sim p_{\text{target}}(\cdot \mid x_{<T})$（$\tau = 0$ 时退化为 $\arg\max \text{logits}_{T-1}$），合法范围 $0 \le l \le T - k$、$l + k - 1 \le e \le T - 1$，$r = e + 1$
- $\mathrm{RoPE}(v,\, t) \in \mathbb{R}^{d_r}$：对 $c \in \{0, \dots, d_r / 2 - 1\}$，以频率 $\theta_c = 10000^{-2c / d_r}$ 对通道对 $(v_{2c}, v_{2c+1})$ 施加旋转角 $t\theta_c$
- $\mathrm{GatedMLP}(d_{\text{in}}, d_{\text{hid}}, d_{\text{out}})$：参数 $\gamma \in \mathbb{R}^{d_{\text{in}}}$，$W_{\text{gate}}, W_{\text{up}} \in \mathbb{R}^{d_{\text{hid}} \times d_{\text{in}}}$，$W_{\text{down}} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{hid}}}$

$$\mathrm{GatedMLP}(z) = W_{\text{down}}\Big(\mathrm{SiLU}\big(W_{\text{gate}}\,\mathrm{RMSNorm}(z)\big) \odot \big(W_{\text{up}}\,\mathrm{RMSNorm}(z)\big)\Big)$$

---

## 1. 参数清单（总计约 $27.7\text{M}$）

| 模块 / 参数 | 形状 / 规格 | 作用 |
|---|---|---|
| `proj_norms[0..3]` | $4 \times \mathrm{RMSNorm}(5120)$ | 四层隐状态归一化 |
| `projs[0..3]` ($P_n$) | $4 \times [512, 5120]$ | 四层历史正向特征降维（生成 $X_j$） |
| `seg_emb` ($E_{\text{seg}}$) | $[3, 512]$ | 历史阶段嵌入（`0: prompt, 1: cot, 2: output`） |
| `out_norm` ($\mathrm{RMSNorm}_x$) | $\mathrm{RMSNorm}(512)$ | 正向特征归一化 |
| `q_projs[0..3]` ($Q_n$) | $4 \times [512, 5120]$ | 四层专用查询语义降维（生成 $q$） |
| `q_seg_emb` ($E_{q,\text{seg}}$) | $[3, 512]$ | 查询端阶段嵌入 |
| `q_norm` ($\mathrm{RMSNorm}_q$) | $\mathrm{RMSNorm}(512)$ | 查询向量归一化 |
| `w_rev` ($W_{\text{rev}}$) | $[256, 512]$ | 逆向 RoPE 值投影 |
| `mlp_start` | $\mathrm{GatedMLP}(1281, 512, 1)$ | 候选起点打分（输入 $[q, X_l, Y_l, \ln(1 + \text{ov})]$） |
| `mlp_no_copy` | $\mathrm{GatedMLP}(514, 256, 1)$ | `NO_COPY` 打分 |
| `mlp_span` | $\mathrm{GatedMLP}(512, 512, 256)$ | 区间专用词特征映射 |
| `mlp_u` | $\mathrm{GatedMLP}(1024, 512, 256)$ | 查询与起点正向特征融合 |
| `phi_same` ($\phi_{\text{same}}$) | $[512]$ | 同块前置占位向量 |
| `h_eos` ($h_{\text{eos}}$) | $[256]$ | 右边界越界哨兵（下标 $T$） |
| `mlp_r_chunk` | $\mathrm{GatedMLP}(1792, 512, 1)$ | 终点块打分 |
| `mlp_r_tok` | $\mathrm{GatedMLP}(1280, 512, 1)$ | 终点块内词打分 |

---

## 2. Forward Pipeline（全程 `float32`）

### Step 0：正向特征 $X_l$、专用查询向量 $q$、逆向 RoPE 特征 $Y_l$ 与纯净区间算子

输入：$h^{(n)} \in \mathbb{R}^{T \times 5120}\ (n \in \mathcal{L})$，$s \in \{0, 1, 2\}^T$

$$X_j = \mathrm{RMSNorm}_x\!\left(\sum_{n \in \mathcal{L}} P_n\,\mathrm{RMSNorm}_n\!\big(h^{(n)}_j\big) + E_{\text{seg}}(s_j)\right) \in \mathbb{R}^{512}, \qquad X \in \mathbb{R}^{T \times 512}$$

$$q = \mathrm{RMSNorm}_q\!\left(\sum_{n \in \mathcal{L}} Q_n\,\mathrm{RMSNorm}_n\!\big(h^{(n)}_{T-1}\big) + E_{q,\text{seg}}(s_{T-1})\right) \in \mathbb{R}^{512}$$

无衰减逆向 RoPE 前缀和与右窗口特征 $Y_l$（$W = 256$，$b_l = \min(l + W,\, T)$，与 $X_l$ 严格分离）：

$$V_j = W_{\text{rev}}\, X_j \in \mathbb{R}^{256}, \qquad F(0) = 0,\quad F(t + 1) = F(t) + \mathrm{RoPE}(V_t,\, t) \in \mathbb{R}^{256}$$

$$Y_l = \frac{1}{\sqrt{b_l - l}}\,\mathrm{RoPE}\!\Big(F(b_l) - F(l),\ -l\Big) = \frac{1}{\sqrt{b_l - l}}\sum_{j = l}^{b_l - 1} \mathrm{RoPE}(V_j,\ j - l) \in \mathbb{R}^{256}$$

区间词特征与纯净区间算子（$0 \le a \le b < T$）：

$$H = \mathrm{GatedMLP}_{\text{span}}(X) \in \mathbb{R}^{T \times 256}, \qquad \tilde{H} = [H;\, h_{\text{eos}}] \in \mathbb{R}^{(T + 1) \times 256}$$

$$\mu(a, b) = \frac{1}{b - a + 1}\sum_{j = a}^{b} H_j \in \mathbb{R}^{256}, \qquad \omega(a, b) = \max_{a \le j \le b} H_j \in \mathbb{R}^{256}, \qquad \phi(a, b) = \big[\mu(a, b),\, \omega(a, b)\big] \in \mathbb{R}^{512}$$

### Step 1：候选池选精确起点 $l$ 或 $\text{NO\_COPY}$

`StartIndex` 检出满足 $\text{tokens}[l] = t_0$ 且 $0 \le l \le T - k$ 的候选，按左上下文最长公共匹配 $\text{ov}_l$ 与位置 $l$ 降序取前 $C \le K$ 个（若 $T < k$ 或 $C = 0$ 直接返回 $\text{NO\_COPY}$）：

$$s_j = \mathrm{GatedMLP}_{\text{start}}\!\Big(\big[q,\, X_{l_j},\, Y_{l_j},\, \ln(1 + \text{ov}_j)\big]\Big) \in \mathbb{R}^1, \qquad j = 1, \dots, C \qquad (\text{输入} \in \mathbb{R}^{1281})$$

$$s_{\text{no}} = \mathrm{GatedMLP}_{\text{no}}\!\Big(\big[q,\, \max_{1 \le j \le C} s_j,\, \operatorname{mean}_{1 \le j \le C} s_j\big]\Big) \in \mathbb{R}^1 \qquad (\text{输入} \in \mathbb{R}^{514})$$

$$z^{(1)} = [s_{\text{no}},\, s_1,\, \dots,\, s_C] \in \mathbb{R}^{C+1}, \qquad p^{(1)} = \mathrm{softmax}\big(z^{(1)}\big), \quad \hat{c} = \arg\max z^{(1)}$$

$$\hat{c} = 0 \ \lor\ p^{(1)}_{\hat{c}} < \tau_{\text{start}} \implies \text{NO\_COPY}, \qquad \text{否则}\ l = l_{\hat{c}}$$

### Step 2：选终点块 $m_r$（支持 $m_r = m_l$ 同块）

$$u = \mathrm{GatedMLP}_u\big([q,\, X_l]\big) \in \mathbb{R}^{256} \qquad ([q, X_l] \in \mathbb{R}^{1024})$$

令 $m_l = \lfloor l / 32 \rfloor$，$m_{\min} = \lfloor (l + k - 1) / 32 \rfloor$；对每个块 $m \in \{m_l, \dots, M - 1\}$，令 $a_m = \max(l, 32m)$，$b_m = \min(32(m + 1), T) - 1$：

$$\phi_{\text{pre}}(m) = \begin{cases} \phi_{\text{same}}, & m = m_l \\ \phi(l,\, 32m - 1), & m > m_l \end{cases} \in \mathbb{R}^{512}, \qquad \phi_{\text{blk}}(m) = \phi(a_m,\, b_m) \in \mathbb{R}^{512}$$

$$z^{(2)}_m = \begin{cases} \mathrm{GatedMLP}_{r,\text{chunk}}\!\Big(\big[u,\, \phi_{\text{pre}}(m),\, \phi_{\text{blk}}(m),\, H_{a_m},\, \tilde{H}_{b_m + 1}\big]\Big), & m \ge m_{\min} \\ -\infty, & m < m_{\min} \end{cases} \qquad (\text{输入} \in \mathbb{R}^{1792},\ z^{(2)} \in \mathbb{R}^{M})$$

$$m_r = \arg\max_{0 \le m < M} z^{(2)}_m$$

### Step 3：块 $m_r$ 内选精确末尾词 $e$

对 $t \in \{0, \dots, 31\}$，令 $e = 32 m_r + t$：

$$z^{(3)}_t = \begin{cases} \mathrm{GatedMLP}_{r,\text{tok}}\!\Big(\big[u,\, \phi(l, e),\, H_e,\, \tilde{H}_{e + 1}\big]\Big), & l + k - 1 \le e < T \\ -\infty, & \text{otherwise} \end{cases} \qquad (\text{输入} \in \mathbb{R}^{1280},\ z^{(3)} \in \mathbb{R}^{32})$$

$$e = 32 m_r + \arg\max_{0 \le t < 32} z^{(3)}_t \quad \Longrightarrow \quad [l,\, r) = [l,\, e + 1)$$

---

## 3. Training Loss（软标签起点 + 越界非对称重罚）

- **负样本**：$\mathcal{L}_{\text{start}} = -\ln p^{(1)}_0 = \mathrm{CE}\!\big(z^{(1)},\, 0\big)$
- **正样本起点损失**（真值 $[l, r)$，$L^* = r - l \ge k$）：记候选 $l_j$ 在可见边界 $T$ 内与目标序列的最长连续匹配长度为 $L_j \in [1,\, \min(L^*,\, T - l_j)]$，$\mathcal{V} = \{j \mid L_j \ge k\}$。若 $\mathcal{V} = \varnothing$ 置 `ignore`；否则令 $y_j = L_j^2 / \sum_{m \in \mathcal{V}} L_m^2$：

$$\mathcal{L}_{\text{start}} = -\sum_{j \in \mathcal{V}} y_j \ln p^{(1)}_j$$

- **正样本终点损失**（Teacher Forcing 代入真值 $l$、$e^* = r - 1$、$m_r^* = \lfloor e^* / 32 \rfloor$、$t^* = e^* \bmod 32$，记 $p^{(2)} = \mathrm{softmax}(z^{(2)})$、$p^{(3)} = \mathrm{softmax}(z^{(3)})$）：

$$\mathcal{L}_{r,\text{block}} = -\ln p^{(2)}_{m_r^*} - \lambda_{\text{over}}\ln\!\left(1 - \sum_{m > m_r^*} p^{(2)}_m\right), \qquad \mathcal{L}_{r,\text{tok}} = -\ln p^{(3)}_{t^*} - \lambda_{\text{over}}\ln\!\left(1 - \sum_{t > t^*} p^{(3)}_t\right)$$

- **Batch 聚合**（有效起点锚点数 $N_{\text{start}}$，正样本数 $N_{\text{pos}}$）：

$$\mathcal{L}_{\text{total}} = \frac{1}{N_{\text{start}}}\sum_{i \in \text{valid\_start}} \mathcal{L}_{\text{start}}^{(i)} + \frac{1}{N_{\text{pos}}}\sum_{i \in \text{pos}} \Big(\mathcal{L}_{r,\text{block}}^{(i)} + \mathcal{L}_{r,\text{tok}}^{(i)}\Big)$$
