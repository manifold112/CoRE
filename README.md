# CoRE: Contrastive Rescoring for Evidence-Aware Multiple-Choice Audio Question Answering

> Anonymous submission to **Interspeech 2026**

This repository contains the anonymous implementation of **CoRE**, a **training-free** and **plug-and-play** test-time option re-scoring method for **multiple-choice Audio Question Answering (AQA)**.

Large Audio-Language Models (LALMs) often exhibit **modality bias** in multiple-choice AQA: instead of grounding predictions in the input audio, they may over-rely on **textual priors** from the question and answer choices. CoRE addresses this issue by introducing a **counterfactual audio** condition at test time, estimating **option-level evidence gain** from the contrast between original and counterfactual inputs, and applying an **adaptive evidence-aware gate** for final prediction.

Under a unified option-level scoring protocol, CoRE consistently improves strong LALM backbones such as **Qwen2-Audio** and **Kimi-Audio** on **DCASE 2025 Task 5** and **AIR-Bench SoundQA**.

**Anonymous project page:** https://anonymous.4open.science/r/CoRE-000E

---

## Abstract

Large Audio-Language Models (LALMs) achieve strong performance on multiple-choice Audio Question Answering (AQA) but often exhibit modality bias, over-relying on textual priors in questions and candidate options rather than grounded acoustic evidence. We present **CoRE**, a training-free, plug-and-play test-time option re-scoring method. CoRE constructs **counterfactual audio** via chunk permutation and random segment reversal to disrupt long-range temporal structure while largely preserving short-time acoustics. It estimates **option-level evidence gain** by contrasting scores from original and counterfactual audio, and applies an **adaptive evidence-aware gate** for final prediction. Under a unified option-scoring protocol, experiments on **DCASE 2025 Task 5** and **AIR-Bench SoundQA** show consistent gains with **Qwen2-Audio** and **Kimi-Audio**.

---

## Highlights

- **Training-free**: no retraining, fine-tuning, or parameter updates
- **Plug-and-play**: can be added on top of existing LALMs at test time
- **Option-level correction**: tailored for multiple-choice AQA
- **Evidence-aware**: estimates how much each option benefits from real acoustic evidence
- **Stable re-scoring**: adaptive gating suppresses noisy or low-conflict corrections

---

## Motivation

Although recent LALMs perform well on AQA, they can still answer multiple-choice questions using **linguistic plausibility** rather than **audio-grounded evidence**. In practice, the model may prefer an option that is semantically likely given the question, even when the actual recording supports a different answer.

CoRE is designed to improve **evidence-aware prediction** at **test time**, without changing the backbone model.

---

## Method Overview

Given an audio clip, a question, and a set of candidate options, CoRE:

1. computes option scores under the **original audio**;
2. constructs a **counterfactual audio** by:
   - **block permutation**,
   - **random within-block time reversal**;
3. computes option scores again under the counterfactual audio;
4. estimates **evidence gain** from the score difference;
5. computes an **adaptive gate** from:
   - **Jensen-Shannon divergence** between the two option distributions,
   - **one-sided entropy reduction** under real audio;
6. fuses the two score vectors for final prediction.

This design disrupts **long-range temporal semantics** while preserving **short-time acoustic statistics** to a practical extent, making the counterfactual condition more informative than naive silence-based contrast.

---

## Formulation

Let the model output option-level logits for the original audio:

\[
z^{+} = z(a, q, X), \qquad p^{+} = \text{softmax}(z^{+})
\]

where \(a\) is the input audio, \(q\) is the question, and \(X\) is the set of candidate options.

### Counterfactual audio

We construct a counterfactual audio \(a_{\text{neg}}\) by:

- splitting the waveform into fixed-length blocks,
- randomly permuting the block order,
- applying random within-block time reversal.

The model then produces:

\[
z^{-} = z(a_{\text{neg}}, q, X), \qquad p^{-} = \text{softmax}(z^{-})
\]

### Evidence gain

The option-level evidence gain is defined as:

\[
\Delta = z^{+} - z^{-}
\]

Intuitively, if an option is truly supported by the original audio, it should receive a stronger score under the real audio than under the counterfactual condition.

### Adaptive evidence-aware gate

To avoid over-correction, CoRE introduces a gate coefficient \(\beta \in [0,1]\), computed from two signals:

- **distributional conflict** between \(p^{+}\) and \(p^{-}\),
- **confidence gain** under real audio.

Specifically,

\[
J = \mathrm{JSD}_2(p^{+} \parallel p^{-}), \qquad u_J = 2^J - 1
\]

\[
\Delta H^{+} = \max(0, H(p^{-}) - H(p^{+}))
\]

\[
u_H = \frac{\Delta H^{+}}{\log_2 K + \epsilon}
\]

\[
\beta = \sqrt{u_J u_H}
\]

### Final re-scoring

The final fused logits are:

\[
Z = (1-\beta) z^{-} + \beta z^{+}
\]

The predicted answer is the option with the highest value in \(Z\).

---

## Counterfactual Audio Construction

In the paper, counterfactual audio is constructed in the **waveform domain** after resampling to the backbone input rate.

Default settings:

- **block duration**: 40 ms
- **within-block reversal probability**: 0.5
- **number of counterfactual waveforms per example**: 1
- **boundary smoothing**: 3 ms linear cross-fade

This choice is intended to:

- preserve **short-time acoustic statistics**,
- disrupt **long-range semantic continuity**,
- provide a more informative contrastive reference than silence or full audio removal.

---

## Experimental Setup

### Benchmarks

We evaluate CoRE on:

- **DCASE 2025 Challenge Task 5**
  - **BQA**: Bioacoustics QA
  - **TSQA**: Temporal Soundscapes QA
  - **CQA**: Complex QA
- **AIR-Bench SoundQA**

### Backbone models

- **Qwen2-Audio-7B-Instruct**
- **Kimi-Audio-7B-Instruct**

### Unified option-level scoring

All methods are evaluated under a **unified option-level scoring protocol**. For each candidate option, the score is computed using **teacher-forced conditional log-likelihood** under a fixed answer template. This avoids generation-time decoding artifacts and makes comparison across methods more stable.

### Ordering robustness

To assess robustness to answer-choice ordering, results are reported as **mean ± std** over **8 random answer-order permutations** per example.

---

## Main Results

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

CoRE consistently improves over:

- **default inference**
- **prompt engineering**
- **Audio-Aware Decoding (AAD)**
- **CoRE-Silence**

The gains are especially pronounced on **BQA** and **TSQA**, where grounded acoustic evidence is critical.

---

## Why CoRE Works

Compared with silence-based or token-level contrastive approaches, CoRE is designed specifically for **multiple-choice AQA**.

Its main advantages are:

1. **More informative negative condition**  
   Instead of removing audio entirely, CoRE constructs a counterfactual waveform that still resembles real audio locally while disrupting the temporal structure needed for reliable semantic understanding.

2. **Option-level correction**  
   CoRE directly contrasts candidate-option scores, which matches the structure of multiple-choice AQA more naturally than token-level contrastive decoding.

3. **Adaptive gating**  
   CoRE only applies strong correction when:
   - the original and counterfactual conditions disagree meaningfully, and
   - the original audio makes the decision more confident.

This avoids unstable or noisy updates in low-conflict settings.

---

## Installation

### Clone the repository

```bash
git clone https://anonymous.4open.science/r/CoRE-000E
cd CoRE-000E
