import sys
from pathlib import Path
import torch

src = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(src)) # 0 的意思是插入在最前面避免重名

from qwen import LLM
from load import load_model
from constant_random import *

def main():
    inputs = torch.load(dir / "inputs.pt", map_location="cpu", weights_only=True)
    outputs = []

    for i in range(case):
        seq_len = inputs[i]["token"].shape[0]
        positions = torch.arange(0, seq_len, dtype=torch.long)
        outputs.append({
            "id": i,
            "logits": LLM()
        })

if __name__ == "__main__":
    main()
