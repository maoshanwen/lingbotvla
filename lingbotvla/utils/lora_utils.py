# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re

import torch
import torch.nn as nn
from peft import LoraConfig, inject_adapter_in_model
from safetensors import safe_open


def freeze_parameters(model: nn.Module):
    # Freeze parameters
    model.requires_grad_(False)
    model.eval()
    model.train()


def _matches_scope(name: str, scope: str) -> bool:
    if scope == "action_expert":
        return ".qwen_expert." in f".{name}."
    if scope == "vlm":
        return ".qwenvl." in f".{name}." and ".visual." not in f".{name}."
    if scope == "vision":
        return ".qwenvl.visual." in f".{name}."
    if scope == "action_expert+vlm":
        return _matches_scope(name, "action_expert") or _matches_scope(name, "vlm")
    raise ValueError(f"Unsupported LoRA scope: {scope}")


def resolve_lora_targets(
    model: nn.Module,
    scope: str,
    module_names: str,
) -> list[str]:
    """Return exact Linear module paths selected for LoRA injection."""
    suffixes = {item.strip() for item in module_names.split(",") if item.strip()}
    targets = [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
        and name.rsplit(".", 1)[-1] in suffixes
        and _matches_scope(name, scope)
    ]
    if not targets:
        raise ValueError(
            f"No Linear modules matched scope={scope!r}, module_names={sorted(suffixes)!r}"
        )
    return targets


def add_lora_to_model(
    model: nn.Module,
    lora_rank=16,
    lora_alpha=32,
    lora_dropout=0.05,
    lora_target_scope="action_expert",
    lora_target_modules="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    lora_modules_to_save="state_proj,action_in_proj,action_out_proj",
    use_rslora=True,
    init_lora_weights="kaiming",
    pretrained_lora_path=None,
    state_dict_converter=None,
):
    freeze_parameters(model)
    model.lora_alpha = lora_alpha
    if init_lora_weights == "kaiming":
        init_lora_weights = True

    target_modules = resolve_lora_targets(
        model,
        scope=lora_target_scope,
        module_names=lora_target_modules,
    )

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_rslora=use_rslora,
        bias="none",
        init_lora_weights=init_lora_weights,
        target_modules=target_modules,
    )
    model = inject_adapter_in_model(lora_config, model)

    save_patterns = [
        re.compile(pattern.strip())
        for pattern in lora_modules_to_save.split(",")
        if pattern.strip()
    ]
    for name, param in model.named_parameters():
        if any(pattern.search(name) for pattern in save_patterns):
            param.requires_grad_(True)

    for param in model.parameters():
        if param.requires_grad:
            param.data = param.to(torch.float32)

    for name, param in model.named_parameters():
        if "lora" in name:
            param.data = param.data.to(dtype=torch.float32)

    # Lora pretrained lora weights
    if pretrained_lora_path is not None:
        state_dict = load_state_dict(pretrained_lora_path)
        if state_dict_converter is not None:
            state_dict = state_dict_converter(state_dict)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        all_keys = [i for i, _ in model.named_parameters()]
        num_updated_keys = len(all_keys) - len(missing_keys)
        num_unexpected_keys = len(unexpected_keys)
        print(
            f"{num_updated_keys} parameters are loaded from {pretrained_lora_path}. {num_unexpected_keys} parameters are unexpected."
        )

    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total = sum(param.numel() for param in model.parameters())
    print(
        f"LoRA targets={len(target_modules)}, trainable={trainable:,}/{total:,} "
        f"({100 * trainable / total:.4f}%)"
    )
    return model


def load_state_dict(file_path, torch_dtype=None):
    if file_path.endswith(".safetensors"):
        return load_state_dict_from_safetensors(file_path, torch_dtype=torch_dtype)
    else:
        return load_state_dict_from_bin(file_path, torch_dtype=torch_dtype)


def load_state_dict_from_safetensors(file_path, torch_dtype=None):
    state_dict = {}
    with safe_open(file_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            state_dict[k] = f.get_tensor(k)
            if torch_dtype is not None:
                state_dict[k] = state_dict[k].to(torch_dtype)
    return state_dict


def load_state_dict_from_bin(file_path, torch_dtype=None):
    state_dict = torch.load(file_path, map_location="cpu", weights_only=True)
    if torch_dtype is not None:
        for i in state_dict:
            if isinstance(state_dict[i], torch.Tensor):
                state_dict[i] = state_dict[i].to(torch_dtype)
    return state_dict
