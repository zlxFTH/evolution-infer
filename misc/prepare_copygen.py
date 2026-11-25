import json
import urllib.request
from pathlib import Path
from transformers import AutoTokenizer

BASE_URL = "https://raw.githubusercontent.com/mbzuai-nlp/CopyGen/main/data"
DOMAINS = ["code", "literature", "random_sequence", "contract"]
MIN_LEN = 3  # doc/1.md: r - l > k (k = 2)

def find_sub(seq, sub):
  m = len(sub)
  if m == 0:
    return -1
  first = sub[0]
  for idx in range(len(seq) - m + 1):
    if seq[idx] == first and seq[idx : idx + m] == sub:
      return idx
  return -1

def locate_one(p_ids, o_ids, cur_i, text, tok):
  e_ids = tok.encode(text, add_special_tokens=False)
  core = e_ids[2:-2] if len(e_ids) > 6 else e_ids[1:-1]
  l = find_sub(p_ids, core)
  rel_i = find_sub(o_ids[cur_i:], core)
  if l == -1 or rel_i == -1:
    return None
  i = cur_i + rel_i
  n = len(core)
  while i > cur_i and l > 0 and o_ids[i - 1] == p_ids[l - 1]:
    i, l, n = i - 1, l - 1, n + 1
  while i + n < len(o_ids) and l + n < len(p_ids) and o_ids[i + n] == p_ids[l + n]:
    n += 1
  return [i, l, l + n] if n >= MIN_LEN else None

def extract_spans(p_ids, o_ids, evidence, tok):
  hit = locate_one(p_ids, o_ids, 0, evidence, tok)
  if hit is not None:
    return [hit]
  spans, cur_i = [], 0
  for line in evidence.splitlines():
    hit = locate_one(p_ids, o_ids, cur_i, line, tok)
    if hit is not None:
      spans.append(hit)
      cur_i = hit[0] + (hit[2] - hit[1])
  return spans

def main():
  root = Path(__file__).resolve().parents[1]
  data_dir = root / "data" / "copygen"
  tok = AutoTokenizer.from_pretrained(str(root / "model" / "qwen"))

  for dom in DOMAINS:
    dom_dir = data_dir / dom
    dom_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "test"]:
      src = dom_dir / f"{split}.jsonl"
      if not src.exists():
        url = f"{BASE_URL}/{dom}/{split}.jsonl"
        print(f"Downloading {url}")
        src.write_bytes(urllib.request.urlopen(url, timeout=60).read())

      dst = dom_dir / f"formatted_{split}.jsonl"
      with open(src, encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for idx, line in enumerate(fin):
          row = json.loads(line)
          user = f"{row['context']}\n\n{row['instruction']}".strip()
          prompt = tok.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
          )
          out = row["gold_answer"]
          p_ids = tok.encode(prompt, add_special_tokens=False)
          o_ids = tok.encode(out, add_special_tokens=False) + [tok.eos_token_id]
          spans = extract_spans(p_ids, o_ids, row["gold_evidence"], tok)

          for i, l, r in spans:
            assert r - l >= MIN_LEN and 0 <= l < r <= len(p_ids) + i
            assert o_ids[i : i + (r - l)] == p_ids[l : r]

          rec = {
            "id": row.get("task_id") or f"{dom}_{split}_{idx}",
            "prompt": prompt,
            "cot": "",
            "output": out,
            "prompt_ids": p_ids,
            "cot_ids": [],
            "output_ids": o_ids,
            "copy_cot": [],
            "copy_output": spans,
          }
          fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
      print(f"Saved {dst}")

if __name__ == "__main__":
  main()
