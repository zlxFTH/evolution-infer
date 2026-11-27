import json
import random
from collections import defaultdict, deque
from pathlib import Path
from urllib.request import urlretrieve

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
MODEL = str(ROOT / "model/qwen")
BASE_URL = "https://raw.githubusercontent.com/mbzuai-nlp/CopyGen/main/data"
COT_RATIO = {"code": 1 / 3, "contract": 1 / 3, "literature": 1 / 3, "random_sequence": 1 / 6}
MIN_COPY = 5
COPY_COST = 4
FIX_CONTRACT_ID = "legal_clause.train.0000723125-25-000028.ex1034-amendedandrestatedp.htm.article_17.bundle_5"

def fix_row(row):
  # copygen_train_contract.jsonl 第 86 条官方原数据末尾多了一个逗号，手动去掉使其成为合法 JSON。
  if row["task_id"] == FIX_CONTRACT_ID and row["gold_answer"].endswith(",\n}"):
    row["gold_answer"] = row["gold_answer"][:-3] + "\n}"
  return row

def copy_spans(tokens, offset, n, prompt_len, cot_len):
  if n < MIN_COPY:
    return []

  starts = defaultdict(list)
  for l in range(offset + n - MIN_COPY + 1):
    starts[tuple(tokens[l : l + MIN_COPY])].append(l)

  dp = [(0, 0)] * (n + 1)
  chosen_end = {}
  ends = deque()
  next_matches = {}
  for i in range(n - MIN_COPY, -1, -1):
    pos = offset + i
    matches = {}
    longest = 0
    for l in starts[tuple(tokens[pos : pos + MIN_COPY])]:
      if l + MIN_COPY > pos:
        break
      length = min(pos - l, n - i, 1 + next_matches.get(l + 1, MIN_COPY - 1))
      matches[l] = length
      if length > longest:
        longest = length
    next_matches = matches

    dp[i] = dp[i + 1]
    end = i + MIN_COPY
    score, neg_cnt = dp[end]
    key = end + score, neg_cnt
    while ends and ends[-1][1] < key:
      ends.pop()
    ends.append((end, key))
    right = i + max(longest, MIN_COPY - 1)
    while ends and ends[0][0] > right:
      ends.popleft()
    if ends:
      end, (score, neg_cnt) = ends[0]
      cand = score - i - COPY_COST, neg_cnt - 1
      if cand >= dp[i]:
        dp[i] = cand
        chosen_end[i] = end

  # 对 DP 敲定的每段 [i, end)，按 1.md 规则择优：(源头 prompt>cot>output, 左侧重合长度, 最靠后 l)
  spans = []
  i = 0
  while i < n:
    if i not in chosen_end:
      i += 1
      continue
    end = chosen_end[i]
    span_len = end - i
    pos = offset + i
    best_l, best_key = -1, None
    for l in starts[tuple(tokens[pos : pos + MIN_COPY])]:
      if l + span_len > pos:
        break
      if tokens[l : l + span_len] != tokens[pos : pos + span_len]:
        continue
      prio = 2 if l + span_len <= prompt_len else (1 if l + span_len <= prompt_len + cot_len else 0)
      ov = 0
      lim = min(l, pos)
      while ov < lim and tokens[l - 1 - ov] == tokens[pos - 1 - ov]:
        ov += 1
      rk = (prio, ov, l)
      if best_key is None or rk > best_key:
        best_l, best_key = l, rk
    spans.append([i, best_l, best_l + span_len])
    i = end
  return spans

def build_prompt(row, tokenizer, enable_thinking):
  user = f"{row['context']}\n\n{row['instruction']}".strip()
  return tokenizer.apply_chat_template(
    [{"role": "user", "content": user}],
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=enable_thinking,
  )

def convert(row, tokenizer, cot=None):
  prompt = build_prompt(row, tokenizer, enable_thinking=cot is not None)
  cot_text = cot or ""
  output = row["gold_answer"]
  prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
  cot_ids = tokenizer.encode(cot_text, add_special_tokens=False) if cot_text else []
  output_ids = tokenizer.encode(output, add_special_tokens=False)
  tokens = prompt_ids + cot_ids + output_ids
  p_len, c_len = len(prompt_ids), len(cot_ids)
  return {
    "id": row["task_id"],
    "prompt": prompt,
    "cot": cot_text,
    "output": output,
    "prompt_ids": prompt_ids,
    "cot_ids": cot_ids,
    "output_ids": output_ids + [tokenizer.eos_token_id],
    "copy_cot": copy_spans(tokens, p_len, c_len, p_len, c_len),
    "copy_output": copy_spans(tokens, p_len + c_len, len(output_ids), p_len, c_len),
  }


def generate_cots(llm, prompts):
  from vllm import SamplingParams

  sp = SamplingParams(temperature=0.3, top_p=0.95, max_tokens=16384)
  outs = llm.generate(prompts, sp)
  cots = []
  for out in outs:
    text = out.outputs[0].text
    cots.append(text.split("</think>", 1)[0] + "</think>\n\n" if "</think>" in text else None)
  return cots


def main():
  from vllm import LLM

  tokenizer = AutoTokenizer.from_pretrained(MODEL)
  rng = random.Random(0)
  tmp, data = ROOT / "tmp", ROOT / "data"
  tmp.mkdir(exist_ok=True)
  data.mkdir(exist_ok=True)

  train_rows = {}
  cot_jobs = []  # (domain, row_idx, prompt)
  for domain, ratio in COT_RATIO.items():
    for split in ("train", "test"):
      name = f"copygen_{split}_{domain}.jsonl"
      source, target = tmp / name, data / name
      source.unlink(missing_ok=True)
      target.unlink(missing_ok=True)
      urlretrieve(f"{BASE_URL}/{domain}/{split}.jsonl", source)
      rows = [fix_row(json.loads(line)) for line in source.open(encoding="utf-8")]
      if split == "test":
        with target.open("w", encoding="utf-8") as out:
          for row in rows:
            print(json.dumps(convert(row, tokenizer), ensure_ascii=False), file=out)
        print(target, flush=True)
      else:
        train_rows[domain] = rows
        for i in sorted(rng.sample(range(len(rows)), round(len(rows) * ratio))):
          cot_jobs.append((domain, i, build_prompt(rows[i], tokenizer, enable_thinking=True)))

  # 四个域要开 CoT 的样本合并为单个批次，交给 vLLM 满并发一次性跑完
  llm = LLM(
    model=MODEL, dtype="bfloat16", max_model_len=24576,
    max_num_seqs=64, max_num_batched_tokens=32768,
    gpu_memory_utilization=0.95, enable_prefix_caching=True, enable_chunked_prefill=True,
    limit_mm_per_prompt={"image": 0, "video": 0},
  )
  cots = generate_cots(llm, [prompt for _, _, prompt in cot_jobs])
  cot_map = {(domain, i): cot for (domain, i, _), cot in zip(cot_jobs, cots)}

  for domain, rows in train_rows.items():
    target = data / f"copygen_train_{domain}.jsonl"
    with target.open("w", encoding="utf-8") as out:
      for i, row in enumerate(rows):
        print(json.dumps(convert(row, tokenizer, cot_map.get((domain, i))), ensure_ascii=False), file=out)
    print(target, flush=True)

if __name__ == "__main__":
  main()
