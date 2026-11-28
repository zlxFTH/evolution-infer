from collections import defaultdict

MIN_MATCH = 5
MIN_COPY = 5

class SuffixMatcher:
  def __init__(self, ctx_ids):
    self.ends = defaultdict(list)
    for end, tok in enumerate(ctx_ids, 1):
      self.ends[tok].append(end)
    self.lengths = {}

  def append(self, tok):
    self.lengths = {end: 1 + self.lengths.get(end - 1, 0) for end in self.ends.get(tok, ())}

  def best(self, min_match=MIN_MATCH):
    length, end = max(((l, e) for e, l in self.lengths.items()), default=(0, 0))
    return (length, end) if length >= min_match else (0, None)
