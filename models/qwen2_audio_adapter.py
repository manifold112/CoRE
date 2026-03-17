import os
from typing import List, Sequence, Union

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration


class Qwen2AudioAdapter:
    """
    Qwen2-Audio adapter for multiple-choice AQA option scoring.

    This adapter exposes a unified API:
        score_options(audio_path, question, options) -> torch.Tensor[K]

    Scoring protocol:
    - build a chat-style prompt with one audio placeholder + question/options text
    - append each candidate answer as the assistant continuation
    - compute teacher-forced conditional log-likelihood of the candidate tokens
    - optionally length-normalize by number of candidate tokens
    """

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen2-Audio-7B-Instruct",
        device: str | None = None,
        torch_dtype: torch.dtype | None = None,
        system_prompt: str = "You are a helpful assistant.",
        answer_prefix: str = "Answer: ",
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype or (
            torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        )
        self.system_prompt = system_prompt
        self.answer_prefix = answer_prefix

        self.processor = AutoProcessor.from_pretrained(model_name_or_path)
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_name_or_path,
            torch_dtype=self.torch_dtype,
        ).to(self.device)
        self.model.eval()

        self.sampling_rate = self.processor.feature_extractor.sampling_rate

    def _load_audio(self, audio: Union[str, np.ndarray]) -> np.ndarray:
        if isinstance(audio, str):
            wav, _ = librosa.load(audio, sr=self.sampling_rate, mono=True)
            return wav.astype(np.float32)
        if isinstance(audio, np.ndarray):
            return audio.astype(np.float32)
        raise TypeError(f"Unsupported audio type: {type(audio)}")

    def _format_question_with_options(
        self,
        question: str,
        options: Sequence[str],
    ) -> str:
        option_lines = [f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)]
        return (
            f"{question.strip()}\n\n"
            f"Options:\n" + "\n".join(option_lines) + "\n\n"
            "Please choose the single best answer."
        )

    def _build_history_conversation(self, user_text: str) -> list[dict]:
        return [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    # The actual waveform is passed separately to `processor(..., audios=...)`.
                    # This placeholder string is only used by the chat template.
                    {"type": "audio", "audio_url": "file://placeholder.wav"},
                    {"type": "text", "text": user_text},
                ],
            },
        ]

    def _prepare_inputs(self, prompt_text: str, audio: np.ndarray) -> dict:
        inputs = self.processor(
            text=prompt_text,
            audios=[audio],
            return_tensors="pt",
            padding=True,
        )
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.device)
        return inputs

    @torch.inference_mode()
    def score_candidate(
        self,
        audio: Union[str, np.ndarray],
        question: str,
        options: Sequence[str],
        candidate: str,
        normalize: bool = True,
    ) -> float:
        """
        Score one candidate answer with teacher-forced conditional log-likelihood.
        """
        wav = self._load_audio(audio)
        user_text = self._format_question_with_options(question, options)
        history = self._build_history_conversation(user_text)

        prompt_text = self.processor.apply_chat_template(
            history,
            add_generation_prompt=True,
            tokenize=False,
        )
        full_text = prompt_text + f"{self.answer_prefix}{candidate}"

        prompt_inputs = self._prepare_inputs(prompt_text, wav)
        full_inputs = self._prepare_inputs(full_text, wav)

        prompt_len = prompt_inputs["input_ids"].shape[1]
        full_ids = full_inputs["input_ids"][0]

        if full_ids.shape[0] <= prompt_len:
            return 0.0

        outputs = self.model(**full_inputs)
        logits = outputs.logits[0]  # [seq_len, vocab]

        target_ids = full_ids[prompt_len:]  # candidate continuation tokens
        # token t is predicted from logits at t-1
        pred_logits = logits[prompt_len - 1 : full_ids.shape[0] - 1]

        log_probs = F.log_softmax(pred_logits, dim=-1)
        token_log_probs = log_probs.gather(
            dim=-1,
            index=target_ids.unsqueeze(-1),
        ).squeeze(-1)

        score = token_log_probs.mean() if normalize else token_log_probs.sum()
        return float(score.item())

    @torch.inference_mode()
    def score_options(
        self,
        audio: Union[str, np.ndarray],
        question: str,
        options: Sequence[str],
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Return a tensor of shape [num_options].
        """
        scores = [
            self.score_candidate(
                audio=audio,
                question=question,
                options=options,
                candidate=opt,
                normalize=normalize,
            )
            for opt in options
        ]
        return torch.tensor(scores, dtype=torch.float32)

    @torch.inference_mode()
    def generate_text(
        self,
        audio: Union[str, np.ndarray],
        question: str,
        options: Sequence[str] | None = None,
        max_new_tokens: int = 64,
    ) -> str:
        """
        Optional helper for debugging.
        """
        wav = self._load_audio(audio)
        user_text = question.strip()
        if options is not None:
            user_text = self._format_question_with_options(question, options)

        history = self._build_history_conversation(user_text)
        prompt_text = self.processor.apply_chat_template(
            history,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = self._prepare_inputs(prompt_text, wav)
        generate_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
        text = self.processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return text.strip()
