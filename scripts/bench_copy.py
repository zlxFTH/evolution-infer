import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import torch

ROOT = Path(__file__).resolve().parents[1]
DOMAINS = ("code", "contract", "literature", "random_sequence")
sys.path.insert(0, str(ROOT / "src"))

from cache import Cache
from copygen import CopyGenHeads, generate_copygen
from copy_locator import CopyLocator, generate_copy_locator
from load import load_model
from qwen import LLM
from spec_decode import DecodeStats


def think_prompt(ids, enabled, open_suffix, closed_suffix):
  for suffix in (closed_suffix, open_suffix):
    if ids[-len(suffix):] == suffix:
      return ids[:-len(suffix)] + (open_suffix if enabled else closed_suffix)
  raise ValueError("Expected a Qwen thinking suffix at the end of the prompt")


def split_thinking(tokens, enabled, delimiter):
  if not enabled:
    return 0, tokens, None, None
  if delimiter[0] not in tokens:
    return len(tokens), [], False, False
  marker = tokens.index(delimiter[0])
  formatted = tokens[marker : marker + len(delimiter)] == delimiter
  end = marker + (len(delimiter) if formatted else 1)
  return end, tokens[end:], True, formatted


def sample_rows(path, domains, limit, seed):
  paths = [(domain, path.parent / f"copygen_test_{domain}.jsonl") for domain in domains] if domains else [(None, path)]
  selected = []
  for domain, file in paths:
    with file.open(encoding="utf-8") as stream:
      rows = list(map(json.loads, stream))
    for row in random.Random(seed).sample(rows, min(limit, len(rows))):
      row["domain"] = domain
      selected.append(row)
  return selected


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", type=Path, default=ROOT / "model/qwen")
  parser.add_argument("--data", type=Path, default=ROOT / "data/copygen_test_code.jsonl")
  parser.add_argument("--domains", nargs="+", choices=DOMAINS, help="按域分别抽样，--limit 是每域样本数")
  parser.add_argument("--copygen", type=Path)
  parser.add_argument("--locator", type=Path, required=True)
  parser.add_argument("--limit", type=int, default=8)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--max-new-tokens", type=int, default=1024)
  parser.add_argument("--min-start-prob", type=float, default=0.6)
  parser.add_argument("--cot", action=argparse.BooleanOptionalAction, default=None)
  parser.add_argument("--skip-ar", action="store_true")
  parser.add_argument("--chunked-gdn", action="store_true")
  parser.add_argument("--out", type=Path, required=True)
  args = parser.parse_args()

  from transformers import AutoTokenizer

  rows = sample_rows(args.data, args.domains, args.limit, args.seed)
  tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
  open_suffix = tokenizer.encode("<think>\n", add_special_tokens=False)
  closed_suffix = tokenizer.encode("<think>\n\n</think>\n\n", add_special_tokens=False)
  cot_end_ids = tokenizer.encode("</think>\n\n", add_special_tokens=False)
  if args.cot is not None:
    for row in rows:
      row["prompt_ids"] = think_prompt(row["prompt_ids"], args.cot, open_suffix, closed_suffix)
  config = json.loads((args.model / "generation_config.json").read_text())
  eos = config["eos_token_id"]
  stop_ids = {eos} if isinstance(eos, int) else set(eos)

  print(f"loading Qwen for {len(rows)} rows", flush=True)
  torch.set_default_dtype(torch.bfloat16)
  with torch.device("cuda"):
    model = LLM(chunked_gdn=args.chunked_gdn).eval()
  torch.set_default_dtype(torch.float32)
  load_model(model, str(args.model))
  locator = CopyLocator().cuda().eval()
  locator.load_state_dict(torch.load(args.locator, map_location="cpu", weights_only=True))
  cache = Cache(max_len=max(len(row["prompt_ids"]) for row in rows) + args.max_new_tokens)
  generate_copygen(model, cache, None, [tokenizer.bos_token_id or tokenizer.eos_token_id], 2)
  torch.cuda.synchronize()

  modes = [("locator", generate_copy_locator, locator)]
  if args.copygen is not None:
    heads = CopyGenHeads().cuda().eval()
    heads.load_state_dict(torch.load(args.copygen, map_location="cpu", weights_only=True))
    modes.insert(0, ("copygen", generate_copygen, heads))
  if not args.skip_ar:
    modes.insert(0, ("ar", generate_copygen, None))
  args.out.parent.mkdir(parents=True, exist_ok=True)
  with args.out.open("w", encoding="utf-8") as out:
    for i, row in enumerate(rows, start=1):
      cot = bool(row["cot_ids"]) if args.cot is None else args.cot
      outputs = {}
      for mode, generate, plugin in modes:
        print(f"{i}/{len(rows)} {row['domain'] or 'data'} {mode} starting", flush=True)
        stats = DecodeStats(accepted_spans=[] if cot else None)
        kwargs = {}
        if mode == "locator":
          kwargs = {"cot_end_ids": cot_end_ids if cot else (), "min_start_prob": args.min_start_prob}
        t0 = perf_counter()
        tokens = generate(
          model, cache, plugin, row["prompt_ids"], args.max_new_tokens,
          eos_token_id=eos, stats=stats, **kwargs,
        )
        torch.cuda.synchronize()
        elapsed = perf_counter() - t0
        outputs[mode] = tokens
        cot_tokens, answer, closed, formatted = split_thinking(tokens, cot, cot_end_ids)
        metrics = asdict(stats)
        spans = metrics.pop("accepted_spans")
        cot_accepted = sum(max(0, min(cot_tokens - start, length)) for start, length in spans or ())
        row_result = {
          "task_id": row["id"], "domain": row["domain"], "mode": mode, "tokens": len(tokens), "seconds": elapsed,
          "chunked_gdn": args.chunked_gdn,
          "same_as_ar": tokens == outputs["ar"] if "ar" in outputs else None,
          "same_as_copygen": tokens == outputs["copygen"] if "copygen" in outputs else None,
          "stopped": bool(tokens) and tokens[-1] in stop_ids,
          "cot": cot, "cot_tokens": cot_tokens, "cot_closed": closed, "cot_delimiter": formatted,
          "cot_accepted_tokens": cot_accepted,
          "prediction": tokenizer.decode(answer, skip_special_tokens=True), **metrics,
        }
        print(json.dumps(row_result, ensure_ascii=False), file=out, flush=True)
        print(
          f"{i}/{len(rows)} {mode} tokens={len(tokens)} cot={cot_tokens} closed={closed} "
          f"seconds={elapsed:.2f} steps={stats.steps} accepted={stats.accepted_tokens} "
          f"same_as_{modes[0][0]}={tokens == outputs[modes[0][0]]}",
          flush=True,
        )


if __name__ == "__main__":
  main()
