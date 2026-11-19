from pathlib import Path
import torch

from constant_random import *

def main():
    dir = Path(__file__).resolve().parents[1] / "data" / "random"
    inputs = torch.load(dir / "inputs.pt", map_location="cpu", weights_only=True)
    outputs = torch.load(dir / "outputs.pt", map_location="cpu", weights_only=True)
    answers = torch.load(dir / "answers.pt", map_location="cpu", weights_only=True)

    for 


if __name__ == "__main__":
    main()