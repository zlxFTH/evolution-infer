import random
import torch
from pathlib import Path

from constant import *

case = 10000
max_vocab = 248000
path = Path(__file__).resolve().parents[1]

def main():
    random.seed(0)
    data = []
    for i in range(case):
        len = random.randint(0, max_positions)
        token = torch.Tensor([random.randint(0, max_vocab) for _ in range(len)])
        data.append({
            "id": i,
            "token": token
        })

if __name__ == "__main__":
    main()