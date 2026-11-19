import random
import torch
from pathlib import Path

case = 10000
max_vocab = 248000
max_len = 128

def main():
    random.seed(0)
    torch.manual_seed(0)
    data = []
    for i in range(case):
        seq_len = random.randint(1, max_len)
        token = torch.randint(0, max_vocab, (seq_len,), dtype=torch.long)
        data.append({
            "id": i,
            "token": token
        })
    dir = Path(__file__).resolve().parents[1] / "data" / "random"
    dir.mkdir(parents=True, exist_ok=True)
    torch.save(data, dir / "inputs.pt")

if __name__ == "__main__":
    main()