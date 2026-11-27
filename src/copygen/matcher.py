from collections import defaultdict
from collections.abc import Sequence

MIN_MATCH = 5
MIN_COPY = 5


class SuffixMatcher:
  def __init__(self, ctx_ids: Sequence[int]):
    self.ends = defaultdict(list)
    for end, token in enumerate(ctx_ids, start=1):
      self.ends[token].append(end)
    self.lengths = {}

  def append(self, token: int):
    # 只有以新 token 结尾的匹配才可能延长上一步的后缀。
    self.lengths = {end: 1 + self.lengths.get(end - 1, 0) for end in self.ends.get(token, ())}

  def best(self, min_match: int = MIN_MATCH) -> tuple[int, int | None]:
    length, end = max(((length, end) for end, length in self.lengths.items()), default=(0, 0))
    return (length, end) if length >= min_match else (0, None)


def match(
  ctx_ids: Sequence[int],
  gen_ids: Sequence[int],
  min_match: int = 2,
) -> tuple[int, int | None]:
  # 返回后缀最长匹配的长度和拷贝起点，长度相等取最靠后的匹配。
  assert min_match >= 1
  matcher = SuffixMatcher(ctx_ids)
  for token in gen_ids:
    matcher.append(token)
  return matcher.best(min_match)
