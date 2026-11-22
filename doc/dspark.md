# DSpark 复现与项目整合方案

本文档规划如何在当前 `evolution-infer` 项目中复现 **DSpark**（*DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation*, arXiv:2607.05147），并与 `CopyGen / Copy Locator` 以及极简二分类 `Router` 整合到同一套推理验证引擎中。

---

## 1. 总体架构与职责分工

将 **Router** 简化为**只做路径决策（`COPY` vs `DSPARK`）**，把“本轮草稿验证几个 token（动态截断长度 $k$）”完全交还给 DSpark 原生的 **Confidence Head + 累积存活率调度**，结构最清晰、训练最解耦：

```text
当前步主模型输出 (cur_token, h_t)
         │
         ▼
   ┌───────────┐
   │  Router   │  (极简二分类: 看 h_t 与 Copy 信号)
   └─────┬─────┘
         │
    ┌────┴──────────────────────┐
    ▼ (COPY)                    ▼ (DSPARK)
CopyGen / Copy Locator     DSparkDraft (并行主干 + 低秩顺序头)
输出长片段草稿 [l, r]        一次输出 γ 个候选 + 条件置信度 c_1..c_γ
    │                           │
    │                      Confidence Scheduler:
    │                      计算累积存活率 a_j = ∏ c_i，动态截断长度 k
    │                           │
    └──────────────┬────────────┘
                   ▼
   共享主模型并行验证 (Qwen3.8-27B, 1 次 forward)
                   │
                   ▼
   接受前 a 个 token + 修正 token；若 a < k，48 层 GDN 状态回滚重放
```

---

## 2. 目录与文件存放规划

与 [copygen.md](copygen.md) 保持一致，新增 `src/dspark/` 与 `src/router.py`，复用 `src/qwen.py` 和 `src/cache.py` 中已扩展的多 token 验证与回滚接口：

```text
repo/
├── src/
│   ├── constant.py                  # [不变] Qwen3.8 超参常量
│   ├── load.py                      # [不变] 主模型权重加载
│   ├── cache.py                     # [复用] snapshot() / restore() 状态回滚
│   ├── qwen.py                      # [复用] return_hidden=True 与多 token 验证支持
│   ├── router.py                    # [新增] 极简二分类 Router (COPY vs DSPARK)
│   ├── copygen/                     # [已有规划] Copy 分支
│   └── dspark/                      # [新增] DSpark 核心模块
│       ├── __init__.py
│       ├── model.py                 # DSparkDraft (并行主干 + 低秩顺序头 + Confidence Head)
│       ├── scheduler.py             # 累积存活率计算、STS 温度校准与动态截断
│       └── decoder.py               # 统一解码循环 (Router -> Copy/DSpark -> Verify -> Rollback)
│
├── train/
│   ├── extract_dspark_feats.py      # 离线抽取主模型 hidden_states 与 target logits/top-k
│   ├── train_dspark.py              # Stage 1: 训并行主干与顺序头；Stage 2: 训 Confidence Head
│   ├── calibrate_sts.py             # 留出集拟合 Sequential Temperature Scaling (STS) 温度
│   └── train_router.py              # 基于 Copy / DSpark 实测收益训二分类 Router
│
├── test/
│   ├── smoke_dspark.py              # 随机权重/单样本验证 DSpark 前向与无损验证闭环
│   └── eval_dspark.py               # 在代码/Spec-Bench 上测接受长度、截断效率与 tok/s
│
└── checkpoints/                     # [不入 Git]
    ├── dspark_qwen38.pt             # DSpark 草稿头与置信度头权重
    ├── dspark_sts.json              # STS 每步校准温度参数
    └── router.pt                    # 二分类 Router 权重
```

---

## 3. 现有代码接口改动

**完全复用 [copygen.md](copygen.md) 第 3 节的改动，无需额外修改 `qwen.py` 或 `cache.py`：**
1. `LLM.forward(positions, input_ids, cache, return_hidden=True)` → 返回 `(logits, hidden_states)`；
2. `Attention.forward` 支持 `not cache.is_prefill and seq_len > 1` 的带偏移因果 `attn_mask`；
3. `LinearAttention.forward` 统一用 `F.conv1d(window, ...)` 支持任意验证长度 $k \ge 1$；
4. `Cache.snapshot()` / `Cache.restore(snap)` 支持 16 层 KV 截断与 48 层 GDN 状态回滚。

---

## 4. DSpark 核心模块设计（`src/dspark/`）

### 4.1 `src/dspark/model.py`：`DSparkDraft`
- **共享冻结参数**：直接复用主模型的 `embed_tokens` 与 `lm_head`（不单独存词表权重，显存极省）。
- **块大小 $\gamma$**：默认设 `block_size = 8`（或 `16`）。

核心子模块构成：
1. **输入拼装与降维投影**：
   - 将主模型上一步隐藏状态 $h_t \in \mathbb{R}^{5120}$、当前已确认 token 的 embedding $e_t \in \mathbb{R}^{5120}$ 融合成锚点向量；
   - 后面拼接 $\gamma$ 个可学习的块内位置占位符 `mask_embed`（shape `[gamma, d_draft]`），组成长度为 $\gamma + 1$ 的草稿块输入。
2. **并行草稿主干（Parallel Backbone）**：
   - 2～4 层轻量双向 Transformer Block（如 `d_draft = 1024` 或 `2048`，或者直接用 1 层 `5120` 维复用 `mtp.layers.0` 权重做冷启动），一次前向输出 $\gamma$ 个位置的隐状态 $u_1, \dots, u_\gamma$。
3. **轻量顺序头（Low-Rank Markov Sequential Head）**：
   - 解决纯并行草稿的 **Suffix Decay（尾部准确率衰减）** 问题；
   - 对 $k = 1 \dots \gamma$ 依次执行极轻量的低秩修正（如 `rank = 128`）：
     - 先用 $u_k$（加前一步修正量）过共享的 `lm_head`（或投影回 `5120` 再过 `lm_head`）采样/贪心选出草稿 token $\hat{y}_k$；
     - 把 $\hat{y}_k$ 的 embedding 经低秩投影融入 $u_{k+1}$，让第 $k+1$ 步感知到第 $k$ 步刚选出的词，大幅提升块内一致性。
4. **Confidence Head**：
   - 极小的线性/两层 MLP 头（`d_draft -> 1`），在每个位置 $k$ 输出条件存活率 logit $z_k$。

```python
class DSparkDraft(nn.Module):
  # 返回:
  #   draft_tokens: [gamma]       候选草稿 token ID
  #   draft_probs : [gamma, V]    (训练/随机采样时返回; 贪心推理可只留 top1)
  #   conf_logits : [gamma]       各位置未校准的置信度 logit
  def forward(self, anchor_hidden, cur_token_emb, embed_tokens, lm_head):
    ...
```

### 4.2 `src/dspark/scheduler.py`：置信度校准与动态截断
完全遵循 DSpark 原版机制：
1. **STS 温度校准**：对每个位置 $k \in [1, \gamma]$ 应用校准温度 $T_k$（由留出集拟合得到）：
   $$c_k = \sigma(z_k / T_k)$$
2. **前缀累积存活率**：
   $$a_j = \prod_{i=1}^{j} c_i, \quad j = 1, \dots, \gamma$$
3. **硬件负载自适应截断**：
   - 设定当前负载对应的边际收益阈值 $\tau$（单并发 Memory-Bound 场景如 `batch=1` 时设较低，如 $\tau = 0.25$；高并发 Compute-Bound 时调高如 $\tau = 0.65$）；
   - 找到满足 `a_j >= tau` 的最大下标 $k$（至少保留 $k = 1$），截断输出 **`draft_tokens[:k]`** 送主模型验证。

---

## 5. 极简二分类 Router 设计（`src/router.py`）

Router 不再负责预测长度 $k$，只做一个极轻量的二分类判断（`< 1 MB`）：
- **输入**：`[h_t, copy_signal]`（主模型隐状态 $h_t$ + Copy 分支置信度/匹配特征）；
- **结构**：`LayerNorm -> Linear(5121, 256) -> SiLU -> Linear(256, 2)`；
- **输出**：`0 = DSPARK` / `1 = COPY`；
- **执行**：
  - 选 `COPY` → 调 `CopyGen`（或后续 `Copy Locator`）吐出长片段草稿；
  - 选 `DSPARK` → 调 `DSparkDraft` + `scheduler` 吐出截断后的 $k$ 个草稿；
  - 两者都统一走 **`model(..., cache)` 并行验证 + `cache.restore(snap)` 回滚**。

---

## 6. 数据准备与两阶段训练流程（`train/`）

### 6.1 数据来源（自蒸馏对齐目标模型分布）
- **通用 + 代码对话**：从 `ShareGPT` / `UltraChat` 以及你的代码数据（`data/code/train.jsonl`、`CommitPackFT`）取 prompt；
- **用 `vllm` 跑一遍 `Qwen3.8-27B` 生成回答**：训练无损草稿模型必须对齐目标模型自身分布，因此以 `Qwen3.8-27B` 生成的 `target_ids` 作为监督目标；
- **离线抽特征（`extract_dspark_feats.py`）**：用冻结的 `LLM(..., return_hidden=True)` 跑一次前向，存下 `(input_ids, hidden_states, target_top_k_logits)`（只存每个位置的 top-16 logits 值与下标，体积缩小上万倍！）。

### 6.2 两阶段训练（`train_dspark.py` + `calibrate_sts.py`）
1. **Stage 1：训练并行主干 + 低秩顺序头**
   - 冻结主模型 `embed_tokens` 和 `lm_head`，只更新 `DSparkDraft` 参数；
   - 损失函数：对块内 $k = 1 \dots \gamma$ 的位置，计算草稿输出分布与主模型目标分布的 **KL 散度 / Cross-Entropy Loss** 加权和（可给靠前位置稍高权重）。
2. **Stage 2：训练 Confidence Head + STS 校准**
   - 冻结 Stage 1 训好的草稿主干；
   - 用草稿分布 $p_k^d$ 与主模型分布 $p_k^t$ 计算理论接受率标签：
     $$c_k^* = 1 - \frac{1}{2}\|p_k^d - p_k^t\|_1$$
     （贪心解码模式下也可直接用 `draft_token == target_top1` 的 `0/1` 标签）；
   - 用 **BCE Loss** 训练 `Confidence Head`；
   - 最后在留出验证集上运行 `calibrate_sts.py`，逐位置优化温度 $T_1 \dots T_\gamma$（最小化 NLL / ECE），存成 `checkpoints/dspark_sts.json`。

### 6.3 训练二分类 Router（`train_router.py`）
- 等 `Copy` 分支和 `DSpark` 分支都训好后，在验证/训练集上离线对比每个位置两者各自的真实接受长度：
  - 若 `copy_accept_len > dspark_accept_len` 且 `copy_accept_len >= 4` → 标为 `COPY (1)`；
  - 否则 → 标为 `DSPARK (0)`；
- 用 Cross-Entropy 训练 `src/router.py`（几分钟即可收敛）。

---

## 7. 推荐落地顺序

1. **先完成 [copygen.md](copygen.md) 的 Step 1**：把 `qwen.py` 多 token 验证和 `cache.py` 48 层 GDN 回滚写通并用 `smoke_verify.py` 测过。
2. **单独复现 DSpark 闭环**：写 `src/dspark/model.py` + `scheduler.py`，先在 1000～2000 条代码/对话数据上跑通 Stage 1 + Stage 2 训练，用 `eval_dspark.py` 验证 **单跑 DSpark** 相对纯 `PyTorch + Cache` 和 `vLLM + MTP` 的平均接受长度与加速比。
3. **接入二分类 Router 合体**：把 `CopyGen`（或升级后的 `Copy Locator`）与 `DSpark` 通过 `src/router.py` 拼在一起，在代码改写 + `CopyGen/data/code` + `Spec-Bench` 上做完整消融对比（`Base` vs `Only Copy` vs `Only DSpark` vs `Router + Copy + DSpark`）。
4. **最后做 TileLang 算子加速**：把瓶颈从纯 PyTorch 换成 TileLang 融合 Decode 核与长块轻量回滚核，拉满端到端真实 `tok/s`。
