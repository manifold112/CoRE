# CoRE: Contrastive Rescoring for Evidence-Aware Multiple-Choice Audio Question Answering

> Anonymous submission to **Interspeech 2026**

This repository contains the anonymous implementation of **CoRE**, a **training-free** and **plug-and-play** test-time option re-scoring method for **multiple-choice Audio Question Answering (AQA)**.

CoRE addresses **modality bias** in Large Audio-Language Models (LALMs): in multiple-choice AQA, models may over-rely on textual priors from the question and answer options instead of grounded acoustic evidence. To mitigate this issue, CoRE constructs a **counterfactual audio** condition, estimates **option-level evidence gain** by contrasting original and counterfactual scores, and applies an **adaptive evidence-aware gate** for final prediction.

Under a unified option-scoring protocol, CoRE consistently improves strong LALM backbones such as **Qwen2-Audio** and **Kimi-Audio** on **DCASE 2025 Task 5** and **AIR-Bench SoundQA**.

---

## Overview

### Motivation

Large Audio-Language Models have shown promising performance on AQA, but they can still answer multiple-choice questions using **linguistic plausibility** rather than **audio-grounded evidence**.

CoRE is designed to improve **evidence-aware prediction** at **test time**, without retraining the backbone model.

### Key idea

Given an audio clip, a question, and a set of candidate options, CoRE:

1. computes option scores under the **original audio**,
2. constructs a **counterfactual audio** by:
   - chunk permutation,
   - random within-chunk time reversal,
3. computes option scores again under the counterfactual audio,
4. estimates **evidence gain** from the score difference,
5. uses an **adaptive gate** based on distributional conflict and confidence gain,
6. produces the final re-scored prediction.

This design preserves short-time acoustic statistics while disrupting long-range temporal structure, providing a more informative reference than naive silence-based contrast.

---

## Method

Let the model output option-level logits for the original audio:

\[
z^{+} = z(a, q, X), \qquad p^{+} = \text{softmax}(z^{+})
\]

We construct a counterfactual audio \(a_{\text{neg}}\) via **block-wise temporal scrambling** and obtain:

\[
z^{-} = z(a_{\text{neg}}, q, X), \qquad p^{-} = \text{softmax}(z^{-})
\]

The option-level **evidence gain** is defined as:

\[
\Delta = z^{+} - z^{-}
\]

To avoid over-correction, CoRE introduces an adaptive gate \(\beta \in [0,1]\), computed from:

- **Jensen–Shannon divergence** between \(p^{+}\) and \(p^{-}\),
- **one-sided entropy reduction**, which measures whether the original audio leads to a more confident decision.

The final re-scored logits are:

\[
Z = (1-\beta)z^{-} + \beta z^{+}
\]

The final prediction is the option with the highest value in \(Z\).

---

## Main characteristics

- **Training-free**: no parameter updates or retraining
- **Plug-and-play**: can be added on top of existing LALMs
- **Option-level**: designed for multiple-choice AQA under a unified scoring protocol
- **Evidence-aware**: explicitly estimates how much each option gains from real audio evidence
- **Stable test-time correction**: adaptive gating reduces noisy or uninformative corrections

---

## Experimental benchmarks

We evaluate CoRE on:

- **DCASE 2025 Challenge Task 5**
  - **BQA**: Bioacoustics QA
  - **TSQA**: Temporal Soundscapes QA
  - **CQA**: Complex QA
- **AIR-Bench SoundQA**

### Backbone models

- **Qwen2-Audio-7B-Instruct**
- **Kimi-Audio-7B-Instruct**

### Evaluation protocol

All methods are evaluated under a **unified option-level scoring protocol** based on **teacher-forced conditional log-likelihood** under a fixed answer template, which avoids generation-time decoding artifacts.

To assess robustness to answer-choice ordering, results are reported as **mean ± std** over **8 random answer-order permutations** per example.

---

## Results

### Top-1 accuracy (%) on DCASE 2025 Task 5 and AIR-Bench SoundQA

| Model | Method | BQA | TSQA | CQA | SoundQA |
|---|---|---:|---:|---:|---:|
| Qwen2-Audio-7B-Instruct | Default | 30.0±2.6 | 39.2±0.9 | 49.6±1.1 | 67.2±1.2 |
| Qwen2-Audio-7B-Instruct | + Prompt Engineering | 31.6±2.3 | 42.5±0.8 | 51.0±1.0 | 68.5±1.1 |
| Qwen2-Audio-7B-Instruct | + AAD | 33.1±2.1 | 45.7±0.8 | 52.6±0.9 | 71.5±1.0 |
| Qwen2-Audio-7B-Instruct | + CoRE-Silence | 33.8±2.2 | 45.6±0.8 | 52.9±0.9 | 72.0±1.0 |
| **Qwen2-Audio-7B-Instruct** | **+ CoRE** | **40.7±2.2** | **49.6±0.8** | **53.2±0.9** | **73.8±0.9** |
| Kimi-Audio-7B-Instruct | Default | 43.3±3.2 | 42.5±1.1 | 60.3±1.1 | 71.5±1.4 |
| Kimi-Audio-7B-Instruct | + Prompt Engineering | 44.2±2.9 | 44.4±1.0 | 61.2±1.0 | 72.7±1.3 |
| Kimi-Audio-7B-Instruct | + AAD | 45.2±2.7 | 46.6±1.0 | 62.2±0.9 | 75.0±1.2 |
| Kimi-Audio-7B-Instruct | + CoRE-Silence | 46.0±2.8 | 47.3±1.0 | 62.8±0.9 | 75.3±1.2 |
| **Kimi-Audio-7B-Instruct** | **+ CoRE** | **51.9±2.8** | **51.3±1.0** | **63.5±0.9** | **77.0±1.1** |

CoRE consistently improves over default inference, prompt engineering, AAD, and the silence-based matched control, indicating that the gains come from a more informative counterfactual reference and adaptive option-level re-scoring.

---

## Counterfactual construction

In our implementation, counterfactual audio is constructed in the **waveform domain** after resampling to the backbone input rate.

Default settings:

- block duration: **40 ms**
- random within-block reversal probability: **0.5**
- number of counterfactual waveforms per example: **1**
- boundary smoothing: **3 ms linear cross-fade**

This design aims to:

- preserve **short-time acoustic statistics**,
- disrupt **long-range temporal continuity**,
- provide a stronger contrastive reference than simple audio removal.

---

## Repository contents

The repository includes code for:

- option-level scoring
- counterfactual audio construction
- adaptive evidence-aware gating
- evaluation on DCASE 2025 Task 5
- evaluation on AIR-Bench SoundQA
- experiments with Qwen2-Audio and Kimi-Audio

> Please refer to the corresponding scripts/configuration files in this repository for the exact implementation details.

---

## Setup

### 1. Create environment

```bash
conda create -n core python=3.10
conda activate core
