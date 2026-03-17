from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from core.counterfactual_audio import (
    CounterfactualConfig,
    make_counterfactual_audio,
    make_silence_counterfactual,
)
from core.utils import remove_file_silent, save_temp_wav


@dataclass
class ScoreResult:
    logits: torch.Tensor
    probs: torch.Tensor


def _call_model_score_options(
    model_adapter,
    audio_input,
    question: str,
    options: Sequence[str],
    normalize: bool = True,
) -> torch.Tensor:
    logits = model_adapter.score_options(
        audio_input,
        question,
        list(options),
        normalize=normalize,
    )
    if not isinstance(logits, torch.Tensor):
        logits = torch.tensor(logits, dtype=torch.float32)
    return logits.float()


def score_options(
    model_adapter,
    audio: str | np.ndarray,
    question: str,
    options: Sequence[str],
    normalize: bool = True,
) -> ScoreResult:
    logits = _call_model_score_options(
        model_adapter=model_adapter,
        audio_input=audio,
        question=question,
        options=options,
        normalize=normalize,
    )
    probs = torch.softmax(logits, dim=-1)
    return ScoreResult(logits=logits, probs=probs)


def score_options_with_counterfactual(
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
) -> tuple[ScoreResult, ScoreResult]:
    """
    Returns:
        positive_result, negative_result
    """
    pos_result = score_options(
        model_adapter=model_adapter,
        audio=audio,
        question=question,
        options=options,
        normalize=normalize,
    )

    if counterfactual_mode == "core":
        cfg = CounterfactualConfig(
            sample_rate=sample_rate,
            block_ms=block_ms,
            reverse_prob=reverse_prob,
            crossfade_ms=crossfade_ms,
            seed=seed,
        )
        neg_audio = make_counterfactual_audio(audio, cfg)
    elif counterfactual_mode == "silence":
        neg_audio = make_silence_counterfactual(audio, sample_rate=sample_rate)
    else:
        raise ValueError(
            f"Unsupported counterfactual_mode: {counterfactual_mode}. "
            "Expected one of {'core', 'silence'}."
        )

    neg_audio_path = None
    try:
        # Qwen adapter can consume ndarray directly.
        # Kimi adapter expects an audio path.
        if hasattr(model_adapter, "__class__") and "Kimi" in model_adapter.__class__.__name__:
            neg_audio_path = save_temp_wav(neg_audio, sr=sample_rate, prefix="core_cf_")
            neg_input = neg_audio_path
        else:
            neg_input = neg_audio

        neg_result = score_options(
            model_adapter=model_adapter,
            audio=neg_input,
            question=question,
            options=options,
            normalize=normalize,
        )
    finally:
        remove_file_silent(neg_audio_path)

    return pos_result, neg_result
