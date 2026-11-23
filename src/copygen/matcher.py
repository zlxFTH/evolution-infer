from collections.abc import Sequence

def match(
  ctx_ids: Sequence[int],
  gen_ids: Sequence[int],
  min_match: int = 2,
) -> tuple[int, int | None]:
  # 返回后缀最长匹配的长度和拷贝起点，长度相等取最靠后的匹配。
  assert min_match >= 1

  max_match = min(len(ctx_ids), len(gen_ids))
  if max_match < min_match:
    return 0, None

  best_length = min_match - 1
  copy_start = None
  last_id = gen_ids[-1]
  for end in range(len(ctx_ids), min_match - 1, -1):
    if end <= best_length:
      break
    if ctx_ids[end - 1] != last_id:
      continue

    length = 1
    limit = min(end, max_match)
    while length < limit and ctx_ids[end - length - 1] == gen_ids[-length - 1]:
      length += 1

    if length > best_length:
      best_length = length
      copy_start = end
      if length == max_match:
        break

  if copy_start is None:
    return 0, None
  return best_length, copy_start
