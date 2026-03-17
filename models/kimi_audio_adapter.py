from typing import Sequence

import torch
import torch.nn.functional as F
from kimia_infer.api.kimia import KimiAudio


class KimiAudioAdapter:
    """
    Kimi-Audio adapter for multiple-choice AQA option scoring.

    This adapter uses the official KimiAudio wrapper for loading resources,
    but performs teacher-forced text scoring directly through:
        - prompt_manager.get_prompt(...).to_tensor()
        - self.model.alm.forward(...)

    Public quick-start examples only show `generate(...)`, but the open-source
    implementation exposes the underlying causal LM + prompt manager, which makes
    teacher-forced candidate scoring possible.
    """

    def __init__(
        self,
        model_name_or_path: str = "moonshotai/Kimi-Audio-7B-Instruct",
        answer_prefix: str = "Answer: ",
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("KimiAudioAdapter currently requires CUDA.")

        self.device = torch.device("cuda")
        self.answer_prefix = answer_prefix

        # We only need text scoring, not waveform detokenization.
        self.model = KimiAudio(
            model_path=model_name_or_path,
            load_detokenizer=False,
        )

        self.text_blank = self.model.extra_tokens.kimia_text_blank
        self.text_eos = self.model.extra_tokens.kimia_text_eos

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

    def _build_messages(
        self,
        audio_path: str,
        question: str,
        options: Sequence[str],
    ) -> list[dict]:
        """
        Kimi-Audio official examples place text instruction before audio content.
        """
        prompt_text = self._format_question_with_options(question, options)
        return [
            {"role": "user", "message_type": "text", "content": prompt_text},
            {"role": "user", "message_type": "audio", "content": audio_path},
        ]

    def _encode_candidate(self, candidate: str) -> list[int]:
        tokenizer = self.model.prompt_manager.text_tokenizer
        token_ids = tokenizer.encode(
            f"{self.answer_prefix}{candidate}",
            bos=False,
            eos=False,
        )
        if len(token_ids) == 0:
            raise ValueError("Candidate tokenization produced an empty sequence.")
        return token_ids

    @staticmethod
    def _last_step_logits(x: torch.Tensor) -> torch.Tensor:
        """
        Normalize possible output shapes to [1, vocab].
        """
        if x.ndim == 3:
            return x[:, -1, :]
        if x.ndim == 2:
            return x
        raise ValueError(f"Unexpected logits shape: {tuple(x.shape)}")

    @torch.inference_mode()
    def _prime_history(self, messages: list[dict]):
        """
        Build the multimodal prefix and run one forward pass to obtain the
        next-token text logits + KV cache.
        """
        history = self.model.prompt_manager.get_prompt(
            messages,
            output_type="text",
            add_assistant_start_msg=True,
        )

        audio_input_ids, text_input_ids, is_continuous_mask, _, _ = history.to_tensor()
        audio_features = [f.to(self.device) for f in history.continuous_feature]

        audio_input_ids = audio_input_ids.to(self.device)
        text_input_ids = text_input_ids.to(self.device)
        is_continuous_mask = is_continuous_mask.to(self.device)

        position_ids = (
            torch.arange(0, audio_input_ids.shape[1], device=self.device)
            .unsqueeze(0)
            .long()
        )

        audio_logits, text_logits, past_key_values = self.model.alm.forward(
            input_ids=audio_input_ids,
            text_input_ids=text_input_ids,
            whisper_input_feature=audio_features,
            is_continuous_mask=is_continuous_mask,
            position_ids=position_ids,
            past_key_values=None,
            return_dict=False,
        )

        last_position_id = audio_input_ids.shape[1] - 1
        return self._last_step_logits(text_logits), past_key_values, last_position_id

    @torch.inference_mode()
    def score_candidate(
        self,
        audio_path: str,
        question: str,
        options: Sequence[str],
        candidate: str,
        normalize: bool = True,
        include_eos: bool = False,
    ) -> float:
        """
        Score one candidate answer with step-by-step teacher forcing.

        At each text step, the audio stream is forced to a blank token, which matches
        the official generation path for `output_type="text"`.
        """
        messages = self._build_messages(audio_path, question, options)
        next_text_logits, past_key_values, last_position_id = self._prime_history(messages)

        candidate_token_ids = self._encode_candidate(candidate)
        if include_eos:
            candidate_token_ids = candidate_token_ids + [self.text_eos]

        total_logprob = 0.0

        for token_id in candidate_token_ids:
            log_probs = F.log_softmax(next_text_logits, dim=-1)
            token_logprob = log_probs[0, token_id]
            total_logprob += float(token_logprob.item())

            decoder_input_audio_ids = torch.tensor(
                [[self.text_blank]],
                device=self.device,
                dtype=torch.long,
            )
            decoder_input_text_ids = torch.tensor(
                [[token_id]],
                device=self.device,
                dtype=torch.long,
            )
            decoder_position_ids = torch.tensor(
                [[last_position_id + 1]],
                device=self.device,
                dtype=torch.long,
            )

            audio_logits, text_logits, past_key_values = self.model.alm.forward(
                input_ids=decoder_input_audio_ids,
                text_input_ids=decoder_input_text_ids,
                whisper_input_feature=None,
                is_continuous_mask=None,
                position_ids=decoder_position_ids,
                past_key_values=past_key_values,
                return_dict=False,
            )

            next_text_logits = self._last_step_logits(text_logits)
            last_position_id += 1

        if normalize:
            return total_logprob / len(candidate_token_ids)
        return total_logprob

    @torch.inference_mode()
    def score_options(
        self,
        audio_path: str,
        question: str,
        options: Sequence[str],
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Return a tensor of shape [num_options].
        """
        scores = [
            self.score_candidate(
                audio_path=audio_path,
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
        audio_path: str,
        question: str,
        options: Sequence[str] | None = None,
        max_new_tokens: int = 128,
    ) -> str:
        """
        Optional helper for debugging.
        """
        if options is None:
            prompt_text = question.strip()
        else:
            prompt_text = self._format_question_with_options(question, options)

        messages = [
            {"role": "user", "message_type": "text", "content": prompt_text},
            {"role": "user", "message_type": "audio", "content": audio_path},
        ]

        _, text_output = self.model.generate(
            messages,
            output_type="text",
            audio_temperature=0.0,
            audio_top_k=5,
            text_temperature=0.0,
            text_top_k=1,
            audio_repetition_penalty=1.0,
            audio_repetition_window_size=64,
            text_repetition_penalty=1.0,
            text_repetition_window_size=16,
            max_new_tokens=max_new_tokens,
        )
        return text_output.strip()
