# DSpark 与 Qwen 2B/4B Draft Baseline 接入、源码改造与文件管理方案

本文档在保留后续 **Router + Copy Locator + DSpark** 完整演进路径的基础上，重点规划当前阶段的两个核心任务：
1. **不加 Router 的单 DSpark 轻量模型（Standalone DSpark）**：独立验证 DSpark 在无路由、无 Copy 辅助下的纯草稿生成能力与推理加速比。
2. **普通 Qwen 2B / 4B 小模型 Draft Baseline**：引入同词表小模型作为标准自回归 Speculative Decoding 对照组，用来标定 DSpark 的训练质量是否达标（MAT、接受率 $\alpha$、Draft 开销、端到端吞吐）。

---

## 1. 总体架构：共享 Verification Loop，可插拔 Draft 后端

无论是 **Standalone DSpark**、**Qwen 2B/4B 小模型 Draft**，还是后续的 **Router + Copy + DSpark**，Target 端的验证与状态回滚逻辑完全一致。因此架构上应当做到 **「Verify 层统一，Draft 层可插拔」**：

```
                   ┌─────────────────────────────┐
                   │   统一 Speculative Loop      │
                   │  (src/spec_verify.py)        │
                   └──────────────┬──────────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        ▼                         ▼                         ▼
┌───────────────┐       ┌───────────────────┐       ┌───────────────────┐
│ Qwen 2B / 4B  │       │ Standalone DSpark │       │ Phase 2:          │
│ Draft Baseline│       │ (无 Router)        │       │ Router + Copy     │
│ (自回归 Draft) │       │ (单步并行预测 γ 个)│       │ + DSpark          │
└───────────────┘       └───────────────────┘       └───────────────────┘
        │                         │                         │
        └─────────────────────────┼─────────────────────────┘
                                  ▼
                   ┌─────────────────────────────┐
                   │   Target: Qwen3.8-27B       │
                   │   src/qwen.py + src/cache.py│
                   │   (一次验证 1+γ 个 token     │
                   │    + 48 层 GDN 状态快照回滚)  │
                   └─────────────────────────────┘
```

### 1.1 为什么选同系列 Qwen 2B / 4B 作为 Baseline
- **词表必须严格对齐**：当前 Target `Qwen3.8-27B`（`src/constant.py`）的 `vocab_size = 248320`。作为 lossless speculative decoding 的 Draft 模型，必须与 Target 共享完全一致的 Tokenizer 与 Token ID 映射（即同系列的 `Qwen3.5-2B` 或 `Qwen3.5-4B`）。
- **单卡 84 GB 显存可行性（RTX 6000D）**：
  - `Qwen3.8-27B`（BF16）权重约 **54 GB**，剩余约 **30 GB**。
  - `Qwen 2B`（BF16）约 **4 GB**，双模型总权重约 **58 GB**。
  - `Qwen 4B`（BF16）约 **8 GB**，双模型总权重约 **62 GB**。
  - 两者都留有 **20+ GB** 空间给 Target/Draft 的 Cache 与前向激活，可在单卡上同时驻留对比。

---

## 2. 目录与文件存放管理规范

为了区分**外部下载的预训练大模型权重**、**自己训练产出的轻量权重**、**推理代码**、**训练脚本**与**Benchmark 结果**，建议统一按如下结构管理（大权重目录和评测产出不提交 Git）：

```
repo/
├── model/                        # [Git 忽略] 外部下载的预训练模型权重
│   ├── qwen/                     # 现有 Qwen3.8-27B Target 权重与 tokenizer
│   ├── qwen-2b/                  # 对照组：Qwen 2B 小模型权重与 tokenizer
│   └── qwen-4b/                  # 对照组：Qwen 4B 小模型权重与 tokenizer
│
├── checkpoints/                  # [Git 忽略] 自己训练的模型权重与实验记录
│   ├── dspark_v1/
│   │   ├── model.safetensors     # 或 .pt，DSparkDraft 权重
│   │   ├── config.json           # 结构超参（d_draft, n_layers, gamma 等）
│   │   └── train_meta.json       # 训练数据来源、loss、step 等元信息
│   └── router_v1/                # Phase 2 预留：Router 权重
│
├── data/                         # [Git 忽略大文件] 训练与评测数据集
│   ├── code/                     # 现有代码数据集（train.jsonl / test.jsonl）
│   └── dspark_distill/           # Target 蒸馏生成的 DSpark 训练数据（含 hidden states 或 token 序列）
│
├── results/                      # [Git 忽略] Benchmark 评测输出的 JSON/CSV 报告
│   ├── baseline_27b.json
│   ├── qwen_2b_draft.json
│   ├── qwen_4b_draft.json
│   └── dspark_standalone.json
│
├── src/                          # 核心推理代码
│   ├── constant.py               # 27B 常量（或补充 Config dataclass）
│   ├── load.py                   # safetensors 权重加载
│   ├── cache.py                  # [需改] 增加 snapshot() / restore()
│   ├── qwen.py                   # [需改] return_hidden、Verify attn_mask、GDN F.conv1d
│   ├── spec_verify.py            # [新增] 统一的 Target 验证 + 回滚主循环
│   ├── draft/                    # [新增] 标准小模型 Draft Baseline 封装
│   │   ├── __init__.py
│   │   └── qwen_draft.py         # Qwen 2B / 4B 自回归 Draft 封装（含自身 Cache 回滚）
│   ├── dspark/                   # [新增] DSpark 轻量并行草稿模块
│   │   ├── __init__.py
│   │   ├── model.py              # DSparkDraft（双向 Attention + SwiGLU + 共享 Target lm_head）
│   │   ├── scheduler.py          # 置信度动态裁剪 k（Phase 1 先支持固定 gamma 与动态 k 切换）
│   │   └── draft.py              # 适配 spec_verify.py 的 Standalone DSpark 包装器
│   ├── copygen/                  # 已有：Copy Locator 匹配与区间打分头
│   └── router.py                 # Phase 2 预留：极简二分类 Router
│
├── train/                        # [新增] 训练与数据构建脚本
│   ├── collect_dspark_data.py    # 用 27B Target（或 vLLM）批量生成/提取训练样本
│   ├── train_dspark.py           # 训练 DSparkDraft（mask 交叉熵 + KL 蒸馏）
│   └── train_router.py           # Phase 2 预留
│
└── test/                         # 单测与 Benchmark 脚本
    ├── test_qwen_single.py       # 已有单测
    ├── bench_target_only.py      # [新增] 纯 27B Baseline 评测
    ├── bench_qwen_draft.py       # [新增] Qwen 2B / 4B Draft + 27B Target 评测
    ├── bench_dspark.py           # [新增] Standalone DSpark + 27B Target 评测
    └── bench_vllm_spec.py        # [新增] vLLM draft_model / mtp 工业对照组（可选）
```

> **管理原则**：
> - `model/` 只放只读的原始大模型；`checkpoints/` 只放自己训出来的产物；两者严格分离，避免误删或混淆。
> - `results/` 下所有 benchmark 脚本输出**同一套 JSON Schema**（包含 `mat`, `accept_rate`, `draft_ms_per_step`, `verify_ms_per_step`, `tokens_per_sec`, `peak_vram_gb`），便于直接出对比表。

---

## 3. 当前源码需要修改的地方（精确到文件与接口）

### 3.1 `src/cache.py`：增加状态快照与回滚（必须）
当前 `Cache`（`max_len=4096, dtype=torch.bfloat16, device="cuda"`）已有 `seq_len`, `is_prefill`, `k_cache`, `v_cache`, `conv_state`, `recur_state` 及 `upd_kv_cache`、`upd_conv_cache`、`upd_recur_cache`、`reset`。

因为 `Qwen3.8-27B` 包含 **48 层 Gated DeltaNet (`LinearAttention`)**，其状态是原地更新的（in-place），不像标准 KV Cache 只需要移动 `seq_len` 指针，所以必须显式保存和恢复每层的 `conv_state` 和 `recur_state`：

- **新增 `Cache.snapshot(self) -> dict`**：
  - 记录当前 `seq_len`（int）。
  - 克隆当前所有线性层的 `conv_state` 与 `recur_state`（例如 `[t.clone() for t in self.conv_state]` 和 `[t.clone() for t in self.recur_state]`；注意 **KV Cache (`k_cache`, `v_cache`) 不需要克隆**，回滚时只需要把 `seq_len` 恢复回去，后续写入会直接覆盖）。
- **新增 `Cache.restore(self, snap: dict)`**：
  - 恢复 `self.seq_len = snap["seq_len"]`。
  - 使用 `copy_()` 将 `snap["conv_state"]` 与 `snap["recur_state"]` 写回原地 tensor。

> **显存估算**：`conv_state` + `recur_state` 共 48 层，单份快照仅约 **~150 MB**，远小于 84 GB 显存上限。

---

### 3.2 `src/qwen.py`：支持多 Token 验证与 Hidden State 导出（必须）
当前 `src/qwen.py` 的实现仅面向单 token decode 和全序列 prefill，需要做以下 3 处核心改动：

1. **`LLM.forward` 增加 `return_hidden: bool = False`**：
   - 当前签名：`forward(self, positions: torch.Tensor, input_ids: torch.Tensor, cache: Cache | None)`。
   - 改动后：当 `return_hidden=True` 时，在 `lm_head` 之前保留 `Model` 输出的 `hidden_states`（shape `[B, L, 5120]`），返回 `(logits, hidden_states)`；默认 `return_hidden=False` 时行为不变，保证现有单测兼容。
2. **`Attention.forward` 支持多 Token 验证掩码（Verification Mask）**：
   - 在 Speculative Decoding 验证阶段，一次向 Target 输入 $1 + \gamma$ 个 token（即 `last_accepted_token` + $\gamma$ 个 draft token）。
   - 当 `not cache.is_prefill and seq_len > 1` 时，`Attention` 不能只算单个 query，而需要传入或构造长度为 `seq_len` 的因果掩码（覆盖 `[0, ..., old_seq_len + seq_len - 1]` 区间），确保第 $i$ 个验证 token 只能看到历史 KV 和它之前的验证 token。
3. **`LinearAttention.forward` 支持 `K > 1` 验证阶段的滑窗卷积与递推**：
   - 当前 `LinearAttention.forward` 在 decode 阶段假设单步输入，需要兼容 `K = 1 + \gamma` 个验证 token 一次性进入的情况：
     - 将历史 `conv_state`（最后 `kernel - 1 = 3` 列）与当前 `K` 个 token 拼接成长度为 `3 + K` 的窗口，使用 `F.conv1d(window, ...)` 一次性算出 `K` 步卷积结果，并用最终末 3 列更新 `conv_state`。
     - 随后按时间步循环递推更新 `recur_state`（$K$ 通常在 $4 \sim 8$，循环开销极小）。
   - 验证完成后，若只接受了 $m < K$ 个 token，如何让 `cache` 精确停在「第 $m$ 个 token 之后的状态」？
     - **推荐做法（最简单零 bug）**：验证时先 `snap = cache.snapshot()`，用 $K = 1 + \gamma$ 跑完一整次 Target 前向；确定接受长度为 $a \in [0, \gamma]$ 后，若全部接受则直接保留当前 cache；若有拒绝（$a < \gamma$），先 `cache.restore(snap)`，再用刚刚已经拿到的被接受 prefix（$1 + a$ 个 token）**仅对最后接受的几步做一次轻量的 state catch-up**（或对 `LinearAttention` 保存每一步的中间状态直接回切，避免再跑一遍大矩阵 MLP）。
     - 注：更精细的做法是在 `LinearAttention` 验证时把这 $K$ 步每一步的 `(conv_state_i, recur_state_i)` 暂存到一个临时 list，确定接受 $a$ 个后直接把第 $a$ 步的状态写回 `cache`，这样可以做到**零重算**！

---

### 3.3 Qwen 2B / 4B 小模型 Draft 的源码适配方式（两种方案二选一）
当前 `src/qwen.py` 顶部写死了 `from constant import *`（硬编码了 27B 的 `hidden_size=5120`, `num_hidden_layers=64` 等）。要在同一个 PyTorch 测试框架里跑 **Qwen 2B / 4B Draft + 27B Target**，有两个选择：

- **方案 A（快速上线对比，推荐当前先用）**：
  - **Target (`27B`)**：保持用我们自己的 `src/qwen.py` + `src/cache.py`。
  - **Draft (`Qwen 2B / 4B`)**：新建 `src/draft/qwen_draft.py`，直接用 `transformers.AutoModelForCausalLM.from_pretrained("model/qwen-2b", torch_dtype=torch.bfloat16, device_map="cuda")` 加载。
  - **状态管理**：HF 自带的 `past_key_values`（或 `DynamicCache`）支持按长度裁剪（`crop(target_len)`）或直接回滚；在每个验证步里，如果 Target 拒绝了部分 token，就将 `qwen_draft` 的 cache 裁剪到实际接受长度即可。
  - **优点**：不用立刻把 `src/qwen.py` 改成完全参数化配置，1 天内就能把 `2B/4B Draft vs 27B Target` 的基准线跑出来。
  - **注意**：对比延迟时，HF 默认 forward 会有少量 Python/framework overhead，记录数据时需把 `draft_ms` 和 `verify_ms` 分开统计，以便客观对比。

- **方案 B（完全自研同构，适合后续深度优化）**：
  - 将 `src/constant.py` 中的超参封装成 `@dataclass class QwenConfig`（从 `model/*/config.json` 读取 `hidden_size`, `num_hidden_layers`, `head_dim` 等），并让 `src/qwen.py` 的 `LLM(cfg)`、`src/cache.py` 的 `Cache(cfg)` 接收 `cfg` 参数（默认仍为 27B 配置）。
  - 这样 `27B Target` 和 `2B/4B Draft` 都跑在完全同一套手写 PyTorch 内核下，消除了框架差异，未来也方便一起做 `torch.compile` 或 CUDA Graph。

---

### 3.4 新增 `src/spec_verify.py`：统一验证循环
所有 Draft 后端统一实现一个简单的 Protocol / 基类接口：

```python
class DraftBackend(Protocol):
  def on_prefill(self, input_ids: torch.Tensor, target_hidden: torch.Tensor | None = None) -> None: ...
  def propose(self, last_token: torch.Tensor, target_hidden: torch.Tensor | None) -> torch.Tensor:
    """返回 shape [1, gamma] 的候选 token_ids"""
    ...
  def on_verify_done(self, accepted_tokens: torch.Tensor, target_hidden: torch.Tensor | None) -> None:
    """根据 Target 最终接受的 prefix（含 bonus token）更新 Draft 侧自身状态"""
    ...
```

`spec_verify.py` 的主循环职责非常单一：
1. **Prefill 阶段**：Target 跑完 prompt，得到首 token `next_token` 和 `last_hidden`；调用 `draft.on_prefill(prompt_ids, last_hidden)`。
2. **Decode 循环**（直到 EOS 或 `max_new_tokens`）：
   - `cand_ids = draft.propose(last_token, last_hidden)`（产出 $\gamma$ 个候选 token）。
   - `snap = target_cache.snapshot()`。
   - 拼接 `verify_input = [last_token, cand_ids]`（长度 $1 + \gamma$），Target 跑一次 `forward(..., return_hidden=True)`，拿到 `logits` 和 `hidden_states`。
   - **Greedy / Sampling 验证**：逐位比对 Target 预测与 `cand_ids`，算出连续接受长度 $a \in [0, \gamma]$，同时从第 $a$ 个位置的 Target logits 采样出 1 个 **bonus token**（即本轮总共推进 $a + 1$ 个 token）。
   - **状态同步**：Target 侧回退/对齐到已接受长度 $a + 1$ 的精确状态；调用 `draft.on_verify_done(accepted_prefix, accepted_hidden)` 同步 Draft 侧状态。
   - 更新 `last_token = bonus_token`、`last_hidden = accepted_hidden[:, -1:]`。

---

## 4. Standalone DSpark 模型架构与代码设计（`src/dspark/model.py`）

在没有 Router 的阶段，DSpark 独立承担所有位置的草稿生成。核心思路是用 **Target 末层 Hidden State** 与 **Token Embedding** 作为条件，配合 $\gamma$ 个可学习的 `[MASK]` query，经过一个极轻量的 **双向注意力 Decoder**，**一步并行**输出未来 $\gamma$ 个位置的 logits。

### 4.1 推荐超参配置（`DSparkConfig`）
- `target_hidden_size = 5120`（对齐 `Qwen3.8-27B`）
- `vocab_size = 248320`（**直接共享冻结 Target 的 `embed_tokens` 与 `lm_head`**，节省 ~2.5B 参数量）
- `draft_hidden_size = 1024`（或 `1536`）
- `num_layers = 2`（先用 2 层验证延迟与 MAT，后续可消融 3~4 层）
- `num_heads = 8`, `num_kv_heads = 4`, `head_dim = 128`（GQA，与 Qwen 一致）
- `intermediate_size = 2816`（SwiGLU）
- `gamma = 4`（单步并行草稿长度，可设 4 或 5）
- 自身可训练参数量：2 层约 **~85M**（含投影层），BF16 仅占 **~0.17 GB** 显存。

### 4.2 数据流与 Attention Mask 设计（核心）
DSpark 有两种序列位置：
1. **Confirmed Tokens（历史已确认前缀 $x_{\le t}$）**：每个 confirmed token $x_j$ 的输入融合了它的 embedding 和 Target hidden state：`proj_in(cat([embed(x_j), target_hidden_j]))`。它们之间是**因果自回归注意力（Causal）**，并在推理时缓存在 DSpark 自己的轻量 KV Cache 中。
2. **Draft Mask Queries（当前步的 $\gamma$ 个 `[MASK]` 槽位 $m_1, \dots, m_\gamma$）**：
   - 每个 `[MASK]` 都能 attend 到**全部历史 Confirmed Tokens**；
   - $\gamma$ 个 `[MASK]` 之间是**全双向注意力（Bidirectional）**，使第 1~$\gamma$ 个预测位置能在单次前向中互相协调语法与词组搭配；
   - **推理时 `[MASK]` 的 KV 不写入永久 Cache**，只有 Target 验证通过后，才把真正 accepted 的 token 写入 DSpark KV Cache。

```
注意力可见性矩阵示例（Confirmed 长度 = 3, gamma = 3）：
          c0  c1  c2  m1  m2  m3
c0       [ 1   0   0   0   0   0 ]   <- Confirmed 之间因果掩码
c1       [ 1   1   0   0   0   0 ]
c2       [ 1   1   1   0   0   0 ]
m1       [ 1   1   1   1   1   1 ]   <- Mask 能看所有 Confirmed，且 Mask 之间双向可见
m2       [ 1   1   1   1   1   1 ]
m3       [ 1   1   1   1   1   1 ]
```

### 4.3 `src/dspark/model.py` 核心代码骨架
```python
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class DSparkConfig:
  target_hidden_size: int = 5120
  draft_hidden_size: int = 1024
  intermediate_size: int = 2816
  num_layers: int = 2
  num_heads: int = 8
  num_kv_heads: int = 4
  head_dim: int = 128
  gamma: int = 4
  rms_norm_eps: float = 1e-6


class RMSNorm(nn.Module):
  def __init__(self, dim: int, eps: float = 1e-6):
    super().__init__()
    self.weight = nn.Parameter(torch.ones(dim))
    self.eps = eps

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    x_f = x.float()
    out = x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + self.eps)
    return (out * self.weight.float()).to(dtype)


class DSparkMLP(nn.Module):
  def __init__(self, cfg: DSparkConfig):
    super().__init__()
    self.gate_proj = nn.Linear(cfg.draft_hidden_size, cfg.intermediate_size, bias=False)
    self.up_proj = nn.Linear(cfg.draft_hidden_size, cfg.intermediate_size, bias=False)
    self.down_proj = nn.Linear(cfg.intermediate_size, cfg.draft_hidden_size, bias=False)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DSparkAttention(nn.Module):
  def __init__(self, cfg: DSparkConfig):
    super().__init__()
    self.num_heads = cfg.num_heads
    self.num_kv_heads = cfg.num_kv_heads
    self.head_dim = cfg.head_dim
    self.q_proj = nn.Linear(cfg.draft_hidden_size, cfg.num_heads * cfg.head_dim, bias=False)
    self.k_proj = nn.Linear(cfg.draft_hidden_size, cfg.num_kv_heads * cfg.head_dim, bias=False)
    self.v_proj = nn.Linear(cfg.draft_hidden_size, cfg.num_kv_heads * cfg.head_dim, bias=False)
    self.o_proj = nn.Linear(cfg.num_heads * cfg.head_dim, cfg.draft_hidden_size, bias=False)

  def forward(
    self,
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    commit_len: int = 0,
  ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
    bsz, seq_len, _ = x.shape
    q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    k = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
    v = self.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

    # 施加 RoPE（复用/对齐 src/qwen.py 的 rotate_half 逻辑）
    q = apply_rope(q, cos, sin)
    k = apply_rope(k, cos, sin)

    if past_kv is not None:
      k_all = torch.cat([past_kv[0], k], dim=2)
      v_all = torch.cat([past_kv[1], v], dim=2)
    else:
      k_all, v_all = k, v

    # 仅将 confirmed token（前 commit_len 个）写入持久 KV；[MASK] 不落盘
    new_kv = (
      (k_all[:, :, : (past_kv[0].size(2) if past_kv else 0) + commit_len],
       v_all[:, :, : (past_kv[1].size(2) if past_kv else 0) + commit_len])
      if commit_len > 0 else past_kv
    )

    k_rep = k_all.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
    v_rep = v_all.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
    out = F.scaled_dot_product_attention(q, k_rep, v_rep, attn_mask=attn_mask)
    out = out.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
    return self.o_proj(out), new_kv


class DSparkBlock(nn.Module):
  def __init__(self, cfg: DSparkConfig):
    super().__init__()
    self.norm1 = RMSNorm(cfg.draft_hidden_size, cfg.rms_norm_eps)
    self.attn = DSparkAttention(cfg)
    self.norm2 = RMSNorm(cfg.draft_hidden_size, cfg.rms_norm_eps)
    self.mlp = DSparkMLP(cfg)

  def forward(self, x, cos, sin, attn_mask=None, past_kv=None, commit_len=0):
    h, new_kv = self.attn(self.norm1(x), cos, sin, attn_mask, past_kv, commit_len)
    x = x + h
    x = x + self.mlp(self.norm2(x))
    return x, new_kv


class DSparkDraft(nn.Module):
  def __init__(self, cfg: DSparkConfig):
    super().__init__()
    self.cfg = cfg
    # 将 [token_emb, target_hidden] 融合映射到 draft 隐空间
    self.in_proj = nn.Linear(cfg.target_hidden_size * 2, cfg.draft_hidden_size, bias=False)
    self.mask_queries = nn.Parameter(torch.randn(1, cfg.gamma, cfg.draft_hidden_size) * 0.02)
    self.layers = nn.ModuleList([DSparkBlock(cfg) for _ in range(cfg.num_layers)])
    self.norm_out = RMSNorm(cfg.draft_hidden_size, cfg.rms_norm_eps)
    self.out_proj = nn.Linear(cfg.draft_hidden_size, cfg.target_hidden_size, bias=False)

  def encode_confirmed(self, token_emb: torch.Tensor, target_hidden: torch.Tensor) -> torch.Tensor:
    return self.in_proj(torch.cat([token_emb, target_hidden], dim=-1))

  def forward_draft(
    self,
    confirmed_feats: torch.Tensor | None,   # 推理增量阶段可只传新确认的 token 特征
    positions_offset: int,
    past_kvs: list | None = None,
    commit_len: int = 0,
    with_masks: bool = True,
    attn_mask: torch.Tensor | None = None,
  ):
    bsz = confirmed_feats.size(0) if confirmed_feats is not None else 1
    parts = []
    if confirmed_feats is not None:
      parts.append(confirmed_feats)
    if with_masks:
      parts.append(self.mask_queries.expand(bsz, -1, -1))
    x = torch.cat(parts, dim=1)

    cos, sin = compute_rope(positions_offset, x.size(1), self.cfg.head_dim, x.device, x.dtype)
    new_kvs = []
    for i, layer in enumerate(self.layers):
      pkv = past_kvs[i] if past_kvs is not None else None
      x, nkv = layer(x, cos, sin, attn_mask=attn_mask, past_kv=pkv, commit_len=commit_len)
      new_kvs.append(nkv)

    if not with_masks:
      return None, new_kvs

    mask_hidden = self.norm_out(x[:, -self.cfg.gamma :])
    # 投影回 5120 维后，外部直接乘冻结的 target_lm_head
    return self.out_proj(mask_hidden), new_kvs
```

### 4.4 `src/dspark/draft.py`（推理期 DraftBackend）与 `scheduler.py`
- **`on_prefill(prompt_ids, target_hidden)`**：用冻结的 `target.embed_tokens(prompt_ids)` 与 `target_hidden` 算出 `confirmed_feats`，调用 `forward_draft(..., commit_len=L, with_masks=False, attn_mask=causal_mask)` 填满 DSpark 的 `past_kvs`。
- **`propose(last_token, last_hidden)`**：只把 $\gamma$ 个 `[MASK]`（如果 `last_token` 还没 commit，也可顺带把 `last_token` 作为 `commit_len=1` 一起送入）拼在 `past_kvs` 后做一次 `forward_draft(..., with_masks=True)`，经过 `target.lm_head` 拿到 `[1, gamma, vocab_size]` logits，送入 `scheduler.py` 输出候选 token。
- **`on_verify_done(accepted_tokens, accepted_hidden)`**：把 Target 本轮最终接受的 $a+1$ 个 token（含 bonus token）通过 `with_masks=False, commit_len=a+1` 写入 `past_kvs`。
- **`src/dspark/scheduler.py`**：
  - Phase 1 先支持固定 `gamma`（如 4 或 5）；
  - 可选开启置信度早停：对每个草稿位取 $p_i = \max(\text{softmax}(z_i))$，若累积概率 $\prod_{j=1}^k p_j < \tau$（如 $\tau=0.25$），截断并只提交前 $k$ 个 token。

---

## 5. DSpark 训练数据、蒸馏方案与训练代码（`train/`）

### 5.1 两种训练模式对比（推荐先跑「单卡在线蒸馏」）
| 维度 | 模式 A：单卡在线蒸馏（推荐首选） | 模式 B：离线 Top-K 蒸馏 |
| :--- | :--- | :--- |
| **显存占用** | 27B Target（54 GB, `@torch.no_grad`）+ DSpark（~1 GB 含优化器）≈ **56 GB / 84 GB** | 仅 DSpark（< 5 GB），可开大 Batch |
| **磁盘占用** | **极低**（只需存几十 MB 的纯文本/token 序列 `train_seqs.jsonl`） | **较大**（需落盘海量 `hidden_states` + `topK_logits`） |
| **数据质量** | Target 实时产出精确 hidden & logits，无量化损失 | 受限于采样存盘的 Top-K 截断 |
| **适用阶段** | 快速验证架构、跑通第一个可用 checkpoint | 大规模多 epoch 调参 |

### 5.2 数据准备（`train/collect_dspark_data.py`）
1. 从 `data/code/train.jsonl`（821 条）及补充代码/通用文本语料出发，用 `Qwen3.8-27B`（可直接用 vLLM 起服务批量生成，吞吐极高）对每个 prompt 生成完整续写（`temperature=0` 或 `0.6`），保存为 `data/dspark_distill/sequences.jsonl`（每行一条完整 `input_ids` 序列，长度控制在 `512 ~ 2048`）。
2. 这样 DSpark 学到的是 **Target 模型自己最倾向生成的分布（on-policy 蒸馏）**，接受率远高于直接拿开源代码文本硬训。

### 5.3 训练逻辑与锚点采样策略（解决「如何高效并行训练 `[MASK]`」）
训练时输入一条长度为 $L$ 的序列 `input_ids`（`[1, L]`）：
1. **Teacher 前向（冻结，不反传）**：Target 跑一次 `forward(..., return_hidden=True)`，得到整条序列的 `target_hidden`（`[1, L, 5120]`）和 `target_logits`（`[1, L, V]`）。
2. **锚点采样（Anchor Sampling）**：
   - 在每条序列中随机抽取 $M$ 个锚点位置 $t \in [16, L - \gamma - 1]$（例如每条序列抽 $M = 8$ 个锚点）。
   - 对每个锚点 $t$：
     - 将 $x_{\le t}$ 的特征 `[embed(x_{\le t}), target_hidden_{\le t}]` 视为 Confirmed 前缀；
     - 后面拼接 $\gamma$ 个 `[MASK]`，用第 4.2 节的注意力掩码前向一次，预测 $t+1, \dots, t+\gamma$ 的 logits $\hat{z}_1, \dots, \hat{z}_\gamma$。
   - **提速技巧（Prefix Cache 共享）**：对同一条序列的多个锚点，只需要先用 `with_masks=False` 因果跑一遍整条前缀 `x_{0:L}` 拿到每层全序列的 `(K, V)`，然后对每个锚点 $t$ 直接切片 `K[:, :, :t+1], V[:, :, :t+1]` 喂给 $\gamma$ 个 `[MASK]`（仅 $\gamma$ 个 token 做前向！），**整条序列 $M$ 个锚点的训练开销接近 0**！
3. **损失函数（位置加权 CE + KL 蒸馏）**：
   $$\mathcal{L} = \sum_{i=1}^{\gamma} w_i \left( (1 - \lambda)\,\text{CE}(\hat{z}_i, x_{t+i}) + \lambda\,T^2\,\text{KL}\!\left(\text{softmax}(z^{\text{target}}_{t+i}/T) \;\middle\|\; \text{log\_softmax}(\hat{z}_i/T)\right) \right)$$
   - 推荐默认超参：位置衰减权重 $w_i = 0.8^{i-1}$（因为第 1 位错则后续全废，第一位权重最高），$\lambda = 0.7$，蒸馏温度 $T = 1.0$。

### 5.4 `train/train_dspark.py` 核心训练代码骨架
```python
import json, random
import torch
import torch.nn.functional as F
from src.qwen import LLM
from src.dspark.model import DSparkConfig, DSparkDraft


def compute_distill_loss(
  student_logits: torch.Tensor,  # [M, gamma, V]
  teacher_logits: torch.Tensor,  # [M, gamma, V]
  target_ids: torch.Tensor,      # [M, gamma]
  lam: float = 0.7,
  decay: float = 0.8,
) -> torch.Tensor:
  gamma = student_logits.size(1)
  total_loss = 0.0
  weight_sum = 0.0
  for i in range(gamma):
    w = decay ** i
    s_log_p = F.log_softmax(student_logits[:, i].float(), dim=-1)
    t_prob = F.softmax(teacher_logits[:, i].float(), dim=-1)
    ce = F.nll_loss(s_log_p, target_ids[:, i])
    kl = F.kl_div(s_log_p, t_prob, reduction="batchmean")
    total_loss = total_loss + w * ((1.0 - lam) * ce + lam * kl)
    weight_sum += w
  return total_loss / weight_sum


def train_step(
  target_model: LLM,
  draft_model: DSparkDraft,
  input_ids: torch.Tensor,       # [1, L]
  optimizer: torch.optim.Optimizer,
  num_anchors: int = 8,
):
  seq_len = input_ids.size(1)
  gamma = draft_model.cfg.gamma

  # 1. Target 整序列前向（冻结，不保留梯度）
  with torch.no_grad():
    positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
    t_logits, t_hidden = target_model(positions, input_ids, cache=None, return_hidden=True)
    tok_emb = target_model.model.embed_tokens(input_ids)
    lm_head_weight = target_model.lm_head.weight  # [V, 5120]

  # 2. 对整条前缀提取 DSpark 的全序列 KV（仅因果前向一次）
  confirmed_feats = draft_model.encode_confirmed(tok_emb, t_hidden)
  # 注意：此处调用因果 forward 获取每层全序列 K, V（供各锚点切片复用）
  _, full_kvs = draft_model.forward_draft(
    confirmed_feats, positions_offset=0, past_kvs=None,
    commit_len=seq_len, with_masks=False, attn_mask="causal",
  )

  # 3. 随机采样 M 个锚点位置 t，切片前缀 KV 并对 [MASK] 做双向预测
  anchors = random.sample(range(16, seq_len - gamma - 1), k=min(num_anchors, seq_len - gamma - 17))
  s_logits_list, t_logits_list, y_list = [], [], []

  for t in anchors:
    slice_kvs = [(k[:, :, : t + 1], v[:, :, : t + 1]) for (k, v) in full_kvs]
    draft_hidden_5120, _ = draft_model.forward_draft(
      confirmed_feats=None,
      positions_offset=t + 1,
      past_kvs=slice_kvs,
      commit_len=0,
      with_masks=True,
      attn_mask=None,  # 仅 gamma 个 [MASK] 查询全量 slice_kvs + 互相双向可见，无需额外 mask
    )
    # 共享冻结的 Target lm_head（注意：此处不要用 torch.no_grad，让梯度通过 lm_head 回传给 DSpark）
    s_logits = F.linear(draft_hidden_5120, lm_head_weight)  # [1, gamma, V]
    s_logits_list.append(s_logits)
    t_logits_list.append(t_logits[:, t : t + gamma])        # Target 在 t..t+gamma-1 预测 t+1..t+gamma
    y_list.append(input_ids[:, t + 1 : t + 1 + gamma])

  loss = compute_distill_loss(
    torch.cat(s_logits_list, dim=0),
    torch.cat(t_logits_list, dim=0),
    torch.cat(y_list, dim=0),
  )

  optimizer.zero_grad()
  loss.backward()
  torch.nn.utils.clip_grad_norm_(draft_model.parameters(), 1.0)
  optimizer.step()
  return loss.item()
```

> **显存与工程细节提示**：
> - `F.linear(draft_hidden_5120, lm_head_weight)` 中 `lm_head_weight` 虽被冻结（`requires_grad=False`），但 PyTorch 仍会通过它反传梯度到 `draft_hidden_5120`；单次 `num_anchors=8, gamma=4` 时 logits 仅 `[8, 4, 248320]`（BF16 仅 ~15 MB），显存极其轻量。
> - 训练完成后，只保存 `draft_model.state_dict()` 到 `checkpoints/dspark_v1/model.safetensors`，并写入 `config.json` 与 `train_meta.json`。

---

## 6. Benchmark 对比矩阵与验收标准

在 `data/code/test.jsonl`（229 条测试样本，统一 `temperature=0`, `max_new_tokens` 一致）上，按以下 5 组横向评测并写入 `results/`：

| 组别 | Draft 模型 | Target 模型 | 核心观察指标 | 目的 |
| :--- | :--- | :--- | :--- | :--- |
| **G1: Pure Target** | 无 | `Qwen3.8-27B` (PyTorch) | 基准 tok/s, TPOT | 测算无加速下的真实单卡基准 |
| **G2: Qwen-2B Draft** | `Qwen 2B` (自回归) | `Qwen3.8-27B` (PyTorch) | MAT, $\alpha$, Draft ms, tok/s | 标定 2B 标准小模型的接受率与加速比 |
| **G3: Qwen-4B Draft** | `Qwen 4B` (自回归) | `Qwen3.8-27B` (PyTorch) | MAT, $\alpha$, Draft ms, tok/s | 标定更强小模型的接受率上限与自身延迟代价 |
| **G4: DSpark Standalone** | `DSparkDraft` (无 Router) | `Qwen3.8-27B` (PyTorch) | MAT, $\alpha$, Draft ms, tok/s | **核心验证**：对比 G2/G3，看 DSpark 能否以更低的 Draft ms 逼近甚至追平 2B/4B 的 MAT，从而实现更高的端到端 tok/s |
| **G5: vLLM Spec (可选)** | `Qwen 2B/4B` 或 `MTP` | `Qwen3.8-27B` (vLLM) | 工业界端到端 tok/s | 验证系统级调度开销与优化天花板 |

### 如何判断 DSpark 是否「训好了」
1. **质量线（Acceptance / MAT）**：在同 $\gamma$ 下，Standalone DSpark 的 **MAT** 应接近或达到 `Qwen 2B Draft` 的水平（例如代码任务上达到 `2.2+` ~ `3.0+`），且显著高于随机/未蒸馏版本。
2. **速度线（Draft Latency）**：`Qwen 2B/4B` 每步需要串行跑 $\gamma$ 次小模型前向；而 DSpark 只需 **1 次并行前向**，Draft 阶段耗时应做到 `Qwen 2B` 方案的 **1/3 ~ 1/5**。
3. **端到端收益**：只有当 `(MAT 提升带来的 Verify 次数减少) > (Draft 耗时 + 回滚开销)`，且最终 `tok/s` 明显高于 **G1（Pure Target）** 并优于/持平 **G2/G3（Qwen 2B/4B）** 时，才算 Standalone DSpark 验证过关；随后即可无缝接入 `Router + Copy Locator`（Phase 2）进一步拉高长代码段的 MAT。

---

## 7. 推荐实施顺序

1. **Step 1 — 打通 `Cache` 回滚与 `qwen.py` 多 token 验证**：完成 `Cache.snapshot()/restore()`、`LLM.forward(..., return_hidden)`、`Attention` 验证掩码与 `LinearAttention` `F.conv1d`，并用单测验证「单步 decode」与「批量 verify + 回滚」输出完全一致（bitwise / tol 内等价）。
2. **Step 2 — 跑通 Baseline（G1 + G2/G3）**：搭建 `src/spec_verify.py` 和 `src/draft/qwen_draft.py`，下载 `Qwen 2B`（或 `4B`）到 `model/qwen-2b/`，先在 `test/` 下测出 G1 和 G2 的 MAT 与 tok/s 基准线。
3. **Step 3 — 实现与训练 Standalone DSpark（G4）**：实现 `src/dspark/model.py` 与 `train/train_dspark.py`（利用 84 GB 单卡在线蒸馏训练），保存权重到 `checkpoints/dspark_v1/`，在完全相同的 `spec_verify.py` 下跑出 G4 对比结果。
4. **Step 4 — 接入 Router + Copy Locator（Phase 2）**：在 Standalone DSpark 达标后，接入 `src/copygen/` 与 `src/router.py`，形成完整混合加速方案。
