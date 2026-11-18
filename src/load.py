import os
from glob import glob
from torch import nn
from safetensors import safe_open

def load_model(model: nn.Module, path: str):
    files = glob(os.path.join(path, "*.safetensors"))   
    assert files

    loaded: set[str] = set()
    for file in files:
        with safe_open(file, framework="pt", device="cpu") as f:
            for name in f.keys():
                if name.startswith(("model.visual.", "mtp.")):
                    continue

                para_name = name.replace("model.language_model.", "model.", 1)
                param = model.get_parameter(para_name)
                tensor = f.get_tensor(name)

                assert param.shape == tensor.shape

                param.data.copy_(tensor)
                loaded.add(para_name)

    all_para = {k for k, _ in model.named_parameters()}
    assert loaded == all_para
