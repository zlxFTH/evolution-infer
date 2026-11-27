from bisect import bisect_right
from collections import defaultdict
from heapq import nlargest

MAX_CANDIDATES = 32


class StartIndex:
  def __init__(self, prompt_ids, min_copy):
    self.min_copy = min_copy
    self.positions = defaultdict(list)
    for pos, token in enumerate(prompt_ids):
      self.positions[token].append(pos)
    self.prompt_len = self.visible = len(prompt_ids)
    self.suffixes = {}

  def extend(self, tokens):
    for token in tokens:
      positions = self.positions[token]
      previous = self.suffixes
      self.suffixes = {pos + 1: 1 + previous.get(pos, 0) for pos in positions}
      positions.append(self.visible)
      self.visible += 1
      self.suffixes[self.visible] = self.visible - self.prompt_len

  def candidates(self, next_id):
    positions = self.positions.get(next_id, ())
    count = bisect_right(positions, self.visible - self.min_copy)
    ranked = nlargest(MAX_CANDIDATES, ((self.suffixes.get(pos, 0), pos) for pos in positions[:count]))
    return [(pos, overlap) for overlap, pos in ranked]
