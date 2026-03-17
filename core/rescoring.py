from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from core.scoring import ScoreResult, score_options_with_counterfactual
from core.utils import entropy_base2, js_divergence_base2


@dataclass
class CoREResult:
    z_pos: torch.Tensor
    z_neg: torch.Tensor
    p_pos: torch.Tensor
    p_neg: torch.Tensor
    delta: torch.Tensor
    beta: float
    u_j: float
    u_h: float
    final_logits: torch.Tensor
    pred_index: int
    pred_option: str


def compute_core_gate(
    p_pos: torch.Tensor,
    p_neg: torch.Tensor,
    eps: float = 1e-12,
) -> tuple[float, float, float]:
    """
    CoRE gate:
        J = JSD_2(p_pos || p_neg)
        u_J = 2^J - 1
        ΔH+ = max(0, H(p_neg) - H(p_pos))
        u_H = ΔH+ / log2(K)
        beta = sqrt(u_J * u_H)
    """
    if p_pos.ndim != 1 or p_neg.ndim != 1:
        raise ValueError("p_pos and p_neg must be 1D probability tensors.")

    k = p_pos.numel()
    if k <= 1:
        return 0.0, 0.0, 0.0

    j = js_divergence_base2(p_pos, p_neg, dim=-1).item()
    u_j = float((2.0 ** j) - 1.0)

    h_pos = entropy_base2(p_pos, dim=-1).item()
    h_neg = entropy_base2(p_neg, dim=-1).item()
    delta_h_pos = max(0.0, h_neg - h_pos)

    denom = max(float(np.log2(k)), eps)
    u_h = float(delta_h_pos / denom)

    beta = float(np.sqrt(max(0.0, u_j * u_h)))
    beta = min(max(beta, 0.0), 1.0)
    return beta, u_j, u_h


def fuse_logits(
    z_pos: torch.Tensor,
    z_neg: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    return (1.0 - beta) * z_neg + beta * z_pos


def core_rescore(
    model_adapter,
    audio: str | np.ndarray,
    question: str,
    options: Sequence[str],
    sample_rate: int,
    normalize: bool = True,
    block_ms: float = 40.0,
    reverse_prob: float = 0.5,
    crossfade_ms: float = 3.0,
    seed: int = 42,
    counterfactual_mode: str = "core",
) -> CoREResult:
    pos_result, neg_result = score_options_with_counterfactual(
        model_adapter=model_adapter,
        audio=audio,
        question=question,
        options=options,
        sample_rate=sample_rate,
        normalize=normalize,
        block_ms=block_ms,
        reverse_prob=reverse_prob,
        crossfade_ms=crossfade_ms,
        seed=seed,
        counterfactual_mode=counterfactual_mode,
    )

    z_pos = pos_result.logits
    z_neg = neg_result.logits
    p_pos = pos_result.probs
    p_neg = neg_result.probs

    delta = z_pos - z_neg
    beta, u_j, u_h = compute_core_gate(p_pos, p_neg)
    final_logits = fuse_logits(z_pos=z_pos, z_neg=z_neg, beta=beta)

    pred_index = int(torch.argmax(final_logits).item())
    pred_option = list(options)[pred_index]

    return CoREResult(
        z_pos=z_pos,
        z_neg=z_neg,
        p_pos=p_pos,
        p_neg=p_neg,
        delta=delta,
        beta=beta,
        u_j=u_j,
        u_h=u_h,
        final_logits=final_logits,
        pred_index=pred_index,
        pred_option=pred_option,
    )


def default_predict(
    model_adapter,
    audio: str | np.ndarray,
    question: str,
    options: Sequence[str],
    normalize: bool = True,
) -> dict:
    logits = model_adapter.score_options(
        audio,
        question,
        list(options),
        normalize=normalize,
    )
    if not isinstance(logits, torch.Tensor):
        logits = torch.tensor(logits, dtype=torch.float32)
    probs = torch.softmax(logits.float(), dim=-1)
    pred_index = int(torch.argmax(logits).item())
    return {
        "logits": logits.float(),
        "probs": probs,
        "pred_index": pred_index,
        "pred_option": list(options)[pred_index],
    }
