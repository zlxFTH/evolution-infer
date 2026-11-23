# CopyGen 复现与项目整合方案

本文档规划如何在当前 `evolution-infer`（PyTorch + `Qwen3.8-27B` 单请求 Cache 基座）上复现 **CopyGen**（*When Copying Is Hard: Copy-Constrained Decoding for Exact Span Reproduction*, arXiv:2610.28393），并确保**多 token 并行验证与 48 层 GDN 状态回滚**这套底层机制能够 100% 复用到后续的 `Copy Locator + DSpark + Router` 架构中。

---

## 1. 复现目标与范围

- **基座模型**：冻结的 `Qwen3.8-27B`（`src/qwen.py` + `src/cache.py` + `src/load.py`），不微调主模型任何权重。
- **数据集**：使用本地已下载的 `data/copygen/code/train.jsonl`（821 条）与 `data/copygen/code/test.jsonl`（229 条）。
- **双模式验证开关**：
  - **`calibrate=False`（无损投机加速模式）**：仅使用 `Trigger Head` + `Length Head` 提出草稿，主模型原样并行验证，**不修改主模型 logits**，严格保证输出与原模型一致。
  - **`calibrate=True`（论文原版模式）**：额外启用 `Verifier Calibration Head` 微调验证期 logits，用于复现论文的 Exact Match（EM）准确率提升并与无损模式做消融对比。

---

## 2. 目录与文件存放规划

保持当前扁平简洁的目录风格，按“模型与推理核心 / 训练脚本 / 评测脚本 / 数据与产物”分层存放：

```text
repo/
├── src/
│   ├── constant.py                  # [不变] Qwen3.8 超参常量
│   ├── load.py                      # [不变] safetensors 权重加载
│   ├── cache.py                     # [改动] 新增 snapshot / rollback 支持投机验证回滚
│   ├── qwen.py                      # [改动] 暴露 hidden_states，支持多 token 续接验证
│   └── copygen/                     # [新增] CopyGen 核心模块
│       ├── __init__.py
│       ├── heads.py                 # TriggerHead / LengthHead / CalibrationHead
│       ├── matcher.py               # 上下文 token 后缀匹配（计算 m_t 与起点 pos）
│       └── decoder.py               # CopyGen 推理主循环（GEN -> COPY -> Verify -> Rollback）
│
├── train/                           # [新增] 特征提取与轻量头训练脚本
│   ├── extract_copygen_feats.py     # 冻结 LLM 跑 Teacher Forcing 离线抽取 hidden_states 与标签
│   └── train_copygen.py             # 训练 CopyGen 小头并保存权重
│
├── test/
│   ├── smoke_random.py              # [不变] 单步前向对拍
│   ├── smoke_cache.py               # [不变] Prefill + Decode 缓存对拍与测速
│   ├── smoke_verify.py              # [新增] 验证“多 token 并行验证 + 48 层 GDN 回滚”数值一致性
│   └── eval_copygen.py              # [新增] 在 data/copygen/code/test.jsonl 上评测 EM、接受长度与 tok/s
│
├── data/
│   └── copygen/
│       └── code/
│           ├── train.jsonl              # [已有] 821 条训练数据
│           ├── test.jsonl               # [已有] 229 条测试数据
│           ├── stats.json               # [已有] 数据统计
│           └── copygen_feats.pt         # [运行生成，不入 Git] 离线特征与标签
│
└── checkpoints/                     # [运行生成，建议加入 .gitignore]
    └── copygen_code.pt              # 训练好的 CopyGen 轻量头权重（约几 MB）
```

> **Git 忽略建议**：在 `.gitignore` 中补充 `checkpoints/` 与 `data/copygen/code/*.pt`，只提交源码和脚本。

---

## 3. 现有代码接口改动清单

为了支持“一次喂入 $K$ 个草稿 token 并行验证 + 验证拒绝后回滚缓存”，只需要对 `src/qwen.py` 和 `src/cache.py` 做小范围、向后兼容的接口扩展（不破坏现有 `smoke_random.py` 和 `smoke_cache.py`）。

### 3.1 `src/qwen.py` 改动点

#### ① `LLM.forward`：增加 `return_hidden` 参数
CopyGen 的 3 个小头都需要主模型最后一层 `Model.norm` 输出的 `hidden_states`（`[seq_len, 5120]`）：

```python
class LLM(nn.Module):
  ...
  def forward(
    self,
    positions: torch.Tensor,
    input_ids: torch.Tensor,
    cache: Cache | None,
    return_hidden: bool = False,
  ):
    hidden = self.model(positions, input_ids, cache)
    logits = self.lm_head(hidden)
    if return_hidden:
      return logits, hidden
    return logits
```

#### ② `Attention.forward`：支持多 token 续接验证（`seq_len > 1` 且 `cache.seq_len > 0`）
当前 `Attention.forward` 里 `is_causal = True if cache is None else cache.is_prefill` 只覆盖了两种情况：
- Prefill（`cache.seq_len == 0, seq_len == L`）：`q` 长度 $L$，`k/v` 长度 $L$，`is_causal=True` 正确；
- 单步 Decode（`seq_len == 1`）：`q` 长度 $1$，`k/v` 长度 `end`，`is_causal=False` 正确。

**新增第 3 种情况（投机验证：`cache.is_prefill == False` 且 `seq_len == K > 1`）**：
此时 `q` 长度为 $K$（对应位置 `beg .. beg + K - 1`），`k/v` 长度为 `end = beg + K`。每个草稿位置 `i`（$0 \le i < K$）只能看到所有历史位置 `0 .. beg - 1` 以及草稿内 `0 .. i`：
- **不能用 `is_causal=True`**（PyTorch SDPA 会按左上角对齐，导致第 0 个草稿看不到 `1 .. beg - 1`）；
- **不能用 `is_causal=False`**（会导致前面的草稿偷看到后面的草稿）。
- **正确做法**：在 `cache is not None and not cache.is_prefill and seq_len > 1` 时构造一个极小的布尔 `attn_mask`（形状 `[seq_len, end]`）：
  ```python
  if cache is not None and not cache.is_prefill and seq_len > 1:
    # 前 beg 列全部可见，后 seq_len 列为下三角因果掩码
    beg = end - seq_len
    mask = torch.ones((seq_len, end), dtype=torch.bool, device=q.device)
    mask[:, beg:] = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=q.device))
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
  else:
    out = F.scaled_dot_product_attention(
      q, k, v,
      is_causal=(True if cache is None else cache.is_prefill),
      enable_gqa=True,
    )
  ```

#### ③ `LinearAttention.forward`：统一支持 `seq_len == 1` 与 `seq_len == K` 的续接卷积
当前第 159～166 行在 `not cache.is_prefill` 时假设了 `window` 只有 4 列（`seq_len == 1`）。
当投机验证一次传入 `seq_len == K` 个草稿 token 时，`cache.upd_conv_cache` 返回的 `window` 形状为 **`[10240, 3 + K]`**。直接调用 `F.conv1d`（`padding=0`）即可**同时统一 `K == 1` 和 `K > 1`**：

```python
if cache is not None:
  window = cache.upd_conv_cache(self.layer_id, qkv.transpose(0, 1))
  if cache.is_prefill:
    qkv = F.silu(self.conv1d(qkv.transpose(0, 1))[:, : seq_len]).transpose(0, 1)
  else:
    # window 形状为 [conv_dim, 3 + seq_len]，无 padding 卷积后正好为 [conv_dim, seq_len]
    qkv = F.silu(F.conv1d(window, self.conv1d.weight, groups=self.conv_dim)).transpose(0, 1)
else:
  qkv = F.silu(self.conv1d(qkv.transpose(0, 1))[:, : seq_len]).transpose(0, 1)
```
而后面的 `for t in range(seq_len):` 递推循环**天然支持任意 `seq_len == K`**，无需改动。

---

### 3.2 `src/cache.py` 改动点（支持状态快照与回滚）

投机验证一次送入 $K$ 个草稿后，假设主模型只接受了前 $a$ 个（$0 \le a \le K$），必须把 64 层缓存精确退回到“接受完第 $a$ 个 token 后的状态”：

- **16 层全注意力（KV Cache）**：极其简单，只需把 `self.seq_len` 改回 `snap_seq_len + a` 即可（后续写入会自动覆盖废弃槽位）。
- **48 层线性注意力（`conv_state` + `recur_state`）**：状态被覆盖写脏，第一版采用 **起点快照 + 轻量化重放**（显存仅多占 1 份 ~151 MB，逻辑 100% 精确）：

在 `Cache` 类中新增 2 个方法：

```python
  def snapshot(self) -> dict:
    # 验证前保存起点快照（仅 1 份，约 151 MB）
    return {
      "seq_len": self.seq_len,
      "conv_state": [None if s is None else s.clone() for s in self.conv_state],
      "recur_state": [None if s is None else s.clone() for s in self.recur_state],
    }

  def restore(self, snap: dict) -> None:
    # 回滚到验证起点
    self.seq_len = snap["seq_len"]
    for i in range(num_hidden_layers):
      if self.conv_state[i] is not None:
        self.conv_state[i].copy_(snap["conv_state"][i])
      if self.recur_state[i] is not None:
        self.recur_state[i].copy_(snap["recur_state"][i])
```

**回滚流程（在 `decoder.py` 中调用）**：
1. 验证前：`snap = cache.snapshot()`；
2. 一次性把 `[cur_token, draft_0, ..., draft_{K-1}]`（共 $K+1$ 个 token，或 `draft_0 .. draft_{K-1}`）送进 `model` 并行前向；
3. 若全部接受（$a == K$）：**零额外开销**，直接继续；
4. 若部分接受（$a < K$）：调用 `cache.restore(snap)`，再把**确定被接受的前 $a$ 个 token** 过一遍前向（或等后续写 TileLang 轻量投影重放核直接重放递推），把 64 层状态推进到精确的 `snap_seq_len + a`。

---

## 4. CopyGen 新增模块设计（`src/copygen/`）

### 4.1 `src/copygen/matcher.py`：后缀匹配器
负责在当前已生成的 token 序列末尾与上下文 `ctx_ids` 之间查找最长匹配后缀：

- **输入**：`ctx_ids: list[int]`（prompt token 序列）、`gen_ids: list[int]`（已生成 token 序列）、`min_match: int = 2`（最小锚定前缀长度）。
- **输出**：`(m_t: int, copy_start: int | None)`
  - `m_t`：当前生成末尾与 `ctx_ids` 连续重合的最长后缀长度（未命中时为 `0`）；
  - `copy_start`：命中位置紧接着的下一个下标（即候选拷贝起点 `l`，直接切片 `ctx_ids[copy_start : copy_start + copy_len]`）。
- **歧义处理**：若 `ctx_ids` 中存在多处相同长度的后缀匹配，默认选最后一次出现的位置（或离当前最近的锚点）。

### 4.2 `src/copygen/heads.py`：3 个轻量 MLP 头
输入特征统一为：
$$u_t = \text{LayerNorm}\big([h_t;\ \log(1 + m_t)]\big) \in \mathbb{R}^{5121}$$

```python
class CopyGenHeads(nn.Module):
  def __init__(self, hidden_size: int = 5120, mid_dim: int = 512):
    super().__init__()
    in_dim = hidden_size + 1
    self.ln = nn.LayerNorm(in_dim)

    # 1. Trigger Head: 二分类 [GEN, COPY]
    self.trigger_head = nn.Linear(in_dim, 2)

    # 2. Length Head: 预测对数长度，推理时 floor(exp(softplus(x)))
    self.length_head = nn.Sequential(
      nn.Linear(in_dim, mid_dim),
      nn.SiLU(),
      nn.Linear(mid_dim, 1),
    )

    # 3. Verifier Calibration Head (可选，用于论文原版模式): 输出对目标拷贝 token 的 logit 偏置
    self.calib_head = nn.Sequential(
      nn.Linear(in_dim, mid_dim),
      nn.SiLU(),
      nn.Linear(mid_dim, 1),
    )
```

> 参数量估算：总计约 **5.2M 参数**（BF16 仅占 **~10 MB**），推理耗时在微秒级。

### 4.3 `src/copygen/decoder.py`：推理主循环
封装统一入口 `generate_copygen(model, cache, heads, prompt_ids, max_new_tokens, calibrate=False)`：

1. **Prefill**：`cache.reset(); cache.is_prefill = True`，跑完 `prompt_ids`，拿到首个 `next_id` 和 `h_t`；随后设 `cache.is_prefill = False`。
2. **主循环（直到 `eos` 或达到 `max_new_tokens`）**：
   - 调 `matcher.match(prompt_ids, gen_ids)` 得到 `(m_t, copy_start)`；
   - 若 `copy_start is not None`：拼接 `u_t = [h_t, log(1 + m_t)]`，跑 `heads.trigger_head(u_t)`；
   - **分支 A（保持 `GEN`）**：走普通单步 Decode，产 1 个 token，更新 `h_t`。
   - **分支 B（触发 `COPY`）**：
     1. 用 `heads.length_head(u_t)` 预测候选长度 $\hat{\ell}_t$（截断到 `max_copy_len` 与上下文剩余长度以内）；
     2. 切出草稿 `draft = prompt_ids[copy_start : copy_start + L]`；
     3. `snap = cache.snapshot()`，把 `[cur_id] + draft` 一次性送进 `model(..., return_hidden=True)`；
     4. 若 `calibrate=True`，用 `heads.calib_head` 对草稿对应 token 的 logit 施加偏置；
     5. 逐位比对主模型预测 `logits[k].argmax()` 与 `draft[k]`，找到最长连续接受长度 $a$ 以及第 $a$ 位的修正 token；
     6. 若 $a < L$，调用 `cache.restore(snap)` 并对已确认前缀重放推进状态；将所有被接受的 token + 修正 token 追加到 `gen_ids`。

---

## 5. 数据准备与训练流程（`train/`）

### 5.1 离线特征与标签提取（`train/extract_copygen_feats.py`）
1. 读取 `data/copygen/code/train.jsonl`（821 条），按 `f"{row['context']}\n\n{row['instruction']}"` 拼装并用 `Qwen3.8-27B` 的 `apply_chat_template(..., enable_thinking=False)` 编码为 `prompt_ids`。
2. **目标输出序列 `target_ids` 选择**：
   - 做**无损投机加速版（`calibrate=False`）**：建议先用 `vllm`（greedy `temperature=0`）对 821 条跑出 `Qwen3.8-27B` 自己的真实回答作为 `target_ids`；
   - 做**论文复现版（`calibrate=True`）**：可以直接使用 `gold_answer` 编码为 `target_ids`。
3. 在 `prompt_ids` 和 `target_ids` 上运行 `matcher` 自动打标：
   - 对 `target_ids` 中每一个位置 $t$：计算当前后缀匹配 `(m_t, copy_start)`，并向后统计从 `copy_start` 开始实际连续匹配的剩余长度 `true_len`；
   - 若 `m_t >= 2` 且 `true_len >= 4`：`trigger_label = 1 (COPY)`，`len_label = log(true_len)`；否则 `trigger_label = 0 (GEN)`。
4. 用冻结的 `LLM(..., return_hidden=True)` 对 `[prompt_ids, target_ids]` 跑 1 次无 Cache 前向，只保存 `target_ids` 区间对应的 `hidden_states`（BF16）与上述标签到 `data/copygen/code/copygen_feats.pt`。

### 5.2 离线训练脚本（`train/train_copygen.py`）
- **无需加载 27B 主模型**，直接读取 `data/copygen/code/copygen_feats.pt` 训练 `CopyGenHeads`：
  - `TriggerHead`：加权 Cross-Entropy Loss（平衡 `GEN` / `COPY` 正负样本比例）；
  - `LengthHead`：仅在 `trigger_label == 1` 的位置上对 `log(true_len)` 计算 Smooth L1 Loss；
  - `CalibrationHead`（如需）：在拷贝边界内外计算二元/margin 校准损失。
- 优化器：`AdamW(lr=1e-3, weight_decay=1e-2)`，余弦退火，训练 10～20 epoch（单卡 6000D 仅需几分钟），保存权重至 `checkpoints/copygen_code.pt`。

---

## 6. 测试与评测方案（`test/`）

1. **`test/smoke_verify.py`（必做单元测试）**：
   - 不依赖训练好的头，随机切一段草稿送进 `model` 做多 token 验证 + 故意在第 $a$ 步触发 `cache.restore(snap)` 回滚重放；
   - 对比回滚后的最终 `logits` 与无 Cache 整段前向的 `softmax` 最大误差，确保 **16 层 KV 掩码 + 48 层 GDN 回滚 100% 无 Bug**。
2. **`test/eval_copygen.py`（端到端评测）**：
   - 在 `data/copygen/code/test.jsonl`（229 条）上对比以下 5 组：
     1. 自研 `PyTorch + Cache`（无投机基线）
     2. 自研 `PyTorch + Cache + CopyGen (calibrate=False)`（无损拷贝加速）
     3. 自研 `PyTorch + Cache + CopyGen (calibrate=True)`（论文校准复现）
     4. `vLLM` 默认版（无投机）
     5. `vLLM + N-gram` / `vLLM + MTP`
   - 统一记录指标：**Exact Match（Quoted block 精确匹配率）**、**Copy 触发率与平均接受长度**、**总耗时与 `tok/s`**。

---

## 7. 推荐实施顺序

1. **Step 1**：改 `src/qwen.py`（`return_hidden`、`Attention` 验证掩码、`LinearAttention` `F.conv1d`）和 `src/cache.py`（`snapshot` / `restore`），写 `test/smoke_verify.py` 跑通数值对齐。
2. **Step 2**：写 `src/copygen/matcher.py` 并在 `data/copygen/code/test.jsonl` 上跑一遍 Oracle 统计（先确认这套数据上有多少比例的 token 能被后缀匹配覆盖）。
3. **Step 3**：写 `src/copygen/heads.py`、`train/extract_copygen_feats.py`、`train/train_copygen.py`，在 AutoDL 上训出 `checkpoints/copygen_code.pt`。
4. **Step 4**：写 `src/copygen/decoder.py` 和 `test/eval_copygen.py`，跑通端到端 EM 与加速比评测。
5. **Step 5（平滑衔接下一阶段）**：保留已验证通过的多 token 验证 + GDN 回滚引擎，把 `matcher.py` 升级为指针式 **`Copy Locator [l, r]`**，把 `GEN` 分支升级为 **`DSpark`**，把 `TriggerHead` 升级为 **`Router`**。
