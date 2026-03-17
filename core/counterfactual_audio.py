from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import librosa
import numpy as np
import soundfile as sf


@dataclass
class CounterfactualConfig:
    sample_rate: int
    block_ms: float = 40.0
    reverse_prob: float = 0.5
    crossfade_ms: float = 3.0
    seed: int = 42


def load_audio_mono(audio: str | np.ndarray, sr: int) -> np.ndarray:
    if isinstance(audio, str):
        wav, _ = librosa.load(audio, sr=sr, mono=True)
        return wav.astype(np.float32)
    if isinstance(audio, np.ndarray):
        audio = audio.astype(np.float32)
        if audio.ndim != 1:
            raise ValueError(f"Expected 1D mono waveform, got shape {audio.shape}")
        return audio
    raise TypeError(f"Unsupported audio type: {type(audio)}")


def save_audio(path: str, audio: np.ndarray, sr: int) -> None:
    sf.write(path, audio.astype(np.float32), sr)


def split_into_blocks(audio: np.ndarray, block_len: int) -> list[np.ndarray]:
    if block_len <= 0:
        raise ValueError(f"block_len must be > 0, got {block_len}")
    if len(audio) == 0:
        return [audio.astype(np.float32)]

    num_blocks = int(np.ceil(len(audio) / block_len))
    blocks = []
    for i in range(num_blocks):
        start = i * block_len
        end = min((i + 1) * block_len, len(audio))
        block = audio[start:end].copy()
        blocks.append(block.astype(np.float32))
    return blocks


def linear_crossfade_concat(
    blocks: Sequence[np.ndarray],
    crossfade_len: int,
) -> np.ndarray:
    if len(blocks) == 0:
        return np.zeros(0, dtype=np.float32)
    if len(blocks) == 1:
        return blocks[0].astype(np.float32)

    output = blocks[0].astype(np.float32).copy()

    for block in blocks[1:]:
        block = block.astype(np.float32)

        if crossfade_len <= 0:
            output = np.concatenate([output, block], axis=0)
            continue

        cf = min(crossfade_len, len(output), len(block))
        if cf == 0:
            output = np.concatenate([output, block], axis=0)
            continue

        fade_out = np.linspace(1.0, 0.0, cf, dtype=np.float32)
        fade_in = np.linspace(0.0, 1.0, cf, dtype=np.float32)

        mixed = output[-cf:] * fade_out + block[:cf] * fade_in
        output = np.concatenate([output[:-cf], mixed, block[cf:]], axis=0)

    return output.astype(np.float32)


def make_counterfactual_audio(
    audio: str | np.ndarray,
    cfg: CounterfactualConfig,
) -> np.ndarray:
    wav = load_audio_mono(audio, sr=cfg.sample_rate)
    if len(wav) == 0:
        return wav

    rng = np.random.default_rng(cfg.seed)

    block_len = max(1, round(cfg.sample_rate * cfg.block_ms / 1000.0))
    crossfade_len = max(0, round(cfg.sample_rate * cfg.crossfade_ms / 1000.0))

    blocks = split_into_blocks(wav, block_len=block_len)
    perm = rng.permutation(len(blocks))

    reordered_blocks: list[np.ndarray] = []
    for idx in perm:
        block = blocks[int(idx)].copy()
        if rng.random() < cfg.reverse_prob:
            block = block[::-1].copy()
        reordered_blocks.append(block.astype(np.float32))

    wav_neg = linear_crossfade_concat(reordered_blocks, crossfade_len=crossfade_len)
    return wav_neg.astype(np.float32)


def make_silence_counterfactual(
    audio: str | np.ndarray,
    sample_rate: int,
) -> np.ndarray:
    wav = load_audio_mono(audio, sr=sample_rate)
    return np.zeros_like(wav, dtype=np.float32)
