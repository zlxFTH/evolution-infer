from collections import defaultdict

MIN_MATCH = 5
MIN_COPY = 1
MAX_CANDIDATES = 8

class SuffixMatcher:
  def __init__(self, ctx_ids):
    self.n = len(ctx_ids)
    self.ends = defaultdict(list)
    for end, tok in enumerate(ctx_ids, 1):
      self.ends[tok].append(end)
    self.lengths = {}
    self.longest = 0

  def append(self, tok):
    self.lengths = {end: 1 + self.lengths.get(end - 1, 0) for end in self.ends.get(tok, ())}
    self.longest = max(self.lengths.values(), default=0)

  def candidates(self, max_cands=MAX_CANDIDATES):
    if not self.longest:
      return []
    return sorted(end for end, length in self.lengths.items() if end < self.n and length == self.longest)[:max_cands]

  def fallback(self, tok, max_cands=MAX_CANDIDATES):
    return [end - 1 for end in self.ends.get(tok, ())][:max_cands]
