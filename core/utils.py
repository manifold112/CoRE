import math
import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import yaml


EPS = 1e-12


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def entropy_base2(probs: torch.Tensor, dim: int = -1, eps: float = EPS) -> torch.Tensor:
    probs = probs.clamp_min(eps)
    return -(probs * torch.log2(probs)).sum(dim=dim)


def js_divergence_base2(
    p: torch.Tensor,
    q: torch.Tensor,
    dim: int = -1,
    eps: float = EPS,
) -> torch.Tensor:
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    p = p / p.sum(dim=dim, keepdim=True)
    q = q / q.sum(dim=dim, keepdim=True)
    m = 0.5 * (p + q)

    kl_pm = (p * (torch.log2(p) - torch.log2(m))).sum(dim=dim)
    kl_qm = (q * (torch.log2(q) - torch.log2(m))).sum(dim=dim)
    return 0.5 * (kl_pm + kl_qm)


def softmax_logits(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.softmax(logits, dim=dim)


def tensor_to_list(x: torch.Tensor | np.ndarray | list[float]) -> list[float]:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return list(x)


def save_temp_wav(
    audio: np.ndarray,
    sr: int,
    suffix: str = ".wav",
    prefix: str = "core_",
) -> str:
    audio = np.asarray(audio, dtype=np.float32)
    with tempfile.NamedTemporaryFile(prefix=prefix, suffix=suffix, delete=False) as f:
        sf.write(f.name, audio, sr)
        return f.name


def remove_file_silent(path: str | None) -> None:
    if path is None:
        return
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def option_letters(num_options: int) -> list[str]:
    return [chr(65 + i) for i in range(num_options)]


def format_options(options: list[str]) -> str:
    lines = [f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)]
    return "\n".join(lines)


def stable_mean(values: list[float]) -> float:
    if len(values) == 0:
        return 0.0
    return float(sum(values) / len(values))


def stable_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = stable_mean(values)
    var = sum((x - mean) ** 2 for x in values) / len(values)
    return float(math.sqrt(var))
