import json
from collections import defaultdict, deque
from pathlib import Path
from urllib.request import urlretrieve

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
BASE_URL = "https://raw.githubusercontent.com/mbzuai-nlp/CopyGen/main/data"
DOMAINS = ("code", "contract", "literature", "random_sequence")
MIN_COPY = 5
COPY_COST = 5

def copy_spans(prompt_ids, output_ids):
  tokens = prompt_ids + output_ids
  starts = defaultdict(list)
  for l in range(len(tokens) - MIN_COPY + 1):
    starts[tuple(tokens[l : l + MIN_COPY])].append(l)

  n = len(output_ids)
  dp = [(0, 0)] * (n + 1)  # (净收益, -复制次数)
  chosen = {}
  ends = deque()
  next_matches = {}
  for i in range(n - MIN_COPY, -1, -1):
    visible = len(prompt_ids) + i
    matches = {}
    source, longest = 0, 0
    for l in starts[tuple(output_ids[i : i + MIN_COPY])]:
      if l + MIN_COPY > visible:
        break
      # 已匹配 MIN_COPY 个 token，复用下一位置的匹配长度。
      length = min(visible - l, 1 + next_matches.get(l + 1, MIN_COPY - 1))
      matches[l] = length
      if length > longest:
        source, longest = l, length
    next_matches = matches

    dp[i] = dp[i + 1]
    # 复制收益是 end - i - COPY_COST，加上 end 之后的收益。
    end = i + MIN_COPY
    score, neg_count = dp[end]
    key = end + score, neg_count
    while ends and ends[-1][1] < key:
      ends.pop()
    ends.append((end, key))
    right = i + max(longest, MIN_COPY - 1)
    while ends and ends[0][0] > right:
      ends.popleft()
    if ends:
      end, (score, neg_count) = ends[0]
      candidate = score - i - COPY_COST, neg_count - 1
      if candidate >= dp[i]:
        dp[i] = candidate
        chosen[i] = source, end

  i = 0
  while i < n:
    if i in chosen:
      l, end = chosen[i]
      yield [i, l, l + end - i]
      i = end
    else:
      i += 1


def convert(row, tokenizer):
  user = f"{row['context']}\n\n{row['instruction']}".strip()
  prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": user}],
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
  )
  output = row["gold_answer"]
  prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
  output_ids = tokenizer.encode(output, add_special_tokens=False)
  return {
    "id": row["task_id"],
    "prompt": prompt,
    "cot": "",
    "output": output,
    "prompt_ids": prompt_ids,
    "cot_ids": [],
    "output_ids": output_ids + [tokenizer.eos_token_id],
    "copy_cot": [],
    "copy_output": list(copy_spans(prompt_ids, output_ids)),
  }

def main():
  tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.8-27B")
  tmp, data = ROOT / "tmp", ROOT / "data"
  tmp.mkdir(exist_ok=True)
  data.mkdir(exist_ok=True)
  for domain in DOMAINS:
    for split in ("train", "test"):
      name = f"copygen_{split}_{domain}.jsonl"
      source, target = tmp / name, data / name
      source.unlink(missing_ok=True)
      target.unlink(missing_ok=True)
      urlretrieve(f"{BASE_URL}/{domain}/{split}.jsonl", source)
      with source.open(encoding="utf-8") as raw, target.open("w", encoding="utf-8") as out:
        for row in map(json.loads, raw):
          print(json.dumps(convert(row, tokenizer), ensure_ascii=False), file=out)
      print(target, flush=True)

if __name__ == "__main__":
  main()
