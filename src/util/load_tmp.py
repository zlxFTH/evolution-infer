import os
from glob import glob

import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(
    param: nn.Parameter,
    loaded_weight: torch.Tensor,
) -> None:
    if param.shape != loaded_weight.shape:
        raise ValueError(
            f"Shape mismatch: parameter {tuple(param.shape)}, "
            f"checkpoint {tuple(loaded_weight.shape)}"
        )
    param.data.copy_(loaded_weight)


def load_model(
    model: nn.Module,
    path: str,
    *,
    ignore_prefixes: tuple[str, ...] = (),
) -> None:
    files = sorted(glob(os.path.join(path, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"No .safetensors files found in {path}")

    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    loaded_names: set[str] = set()

    for file in files:
        with safe_open(file, framework="pt", device="cpu") as f:
            for weight_name in f.keys():
                if weight_name.startswith(ignore_prefixes):
                    continue

                # 可选：由模型提供 checkpoint 名称 -> 模块名称的映射。
                # 例如 model.language_model.layers -> model.layers
                param_name = weight_name
                for old, new in getattr(model, "weight_name_mapping", {}).items():
                    param_name = param_name.replace(old, new)

                for key, (target_name, shard_id) in packed_modules_mapping.items():
                    if key in param_name:
                        param_name = param_name.replace(key, target_name)
                        param = model.get_parameter(param_name)

                        # 避免把同一张量重复计入统计。
                        if weight_name in loaded_names:
                            raise RuntimeError(f"Duplicate weight: {weight_name}")

                        weight_loader = getattr(param, "weight_loader", None)
                        if weight_loader is None:
                            raise AttributeError(
                                f"{param_name} is a packed parameter but has "
                                f"no weight_loader"
                            )
                        weight_loader(
                            param,
                            f.get_tensor(weight_name),
                            shard_id,
                        )
                        loaded_names.add(weight_name)
                        break
                else:
                    param = model.get_parameter(param_name)
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, f.get_tensor(weight_name))
                    loaded_names.add(weight_name)

    print(f"Loaded {len(loaded_names)} tensors from {len(files)} shards")