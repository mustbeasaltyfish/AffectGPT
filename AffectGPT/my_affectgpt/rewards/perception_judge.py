import re

import torch


class GemmaPerceptionJudge:
    def __init__(
        self,
        model_id="google/gemma-4-E4B-it",
        dtype="bfloat16",
        device_map="auto",
        max_new_tokens=4,
    ):
        self.model_id = model_id
        self.dtype = dtype
        self.device_map = device_map
        self.max_new_tokens = int(max_new_tokens)

        self._processor = None
        self._model = None

    def _resolve_dtype(self):
        dtype_name = str(self.dtype).lower()
        if dtype_name in {"auto", ""}:
            return "auto"
        if dtype_name == "bfloat16":
            return torch.bfloat16
        if dtype_name == "float16":
            return torch.float16
        if dtype_name == "float32":
            return torch.float32
        raise ValueError(f"Unsupported judge dtype: {self.dtype}")

    def _load_transformers(self):
        try:
            import transformers
            from transformers import AutoProcessor
        except ImportError as error:
            raise ImportError(
                "PerceptionReward requires transformers to load the local Gemma judge."
            ) from error

        model_cls = getattr(transformers, "AutoModelForMultimodalLM", None)
        if model_cls is None:
            model_cls = getattr(transformers, "Gemma3nForConditionalGeneration", None)
        if model_cls is None:
            model_cls = getattr(transformers, "AutoModelForImageTextToText", None)
        if model_cls is None:
            raise ImportError(
                "No multimodal model loader is available in transformers for the Gemma judge."
            )
        return AutoProcessor, model_cls

    def _ensure_loaded(self):
        if self._processor is not None and self._model is not None:
            return

        AutoProcessor, model_cls = self._load_transformers()
        torch_dtype = self._resolve_dtype()
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        model_kwargs = {"device_map": self.device_map}
        if torch_dtype != "auto":
            model_kwargs["torch_dtype"] = torch_dtype
        self._model = model_cls.from_pretrained(self.model_id, **model_kwargs)
        self._model.eval()

    def _move_inputs(self, model_inputs):
        model_device = getattr(self._model, "device", None)
        if model_device is None or str(model_device) == "meta":
            return model_inputs
        moved_inputs = {}
        for key, value in model_inputs.items():
            if hasattr(value, "to"):
                moved_inputs[key] = value.to(model_device)
            else:
                moved_inputs[key] = value
        return moved_inputs

    def _build_prompt_text(self, subtitle, answer_a, answer_b):
        subtitle = subtitle or ""
        return (
            "You are judging which candidate answer better matches the emotion conveyed by the provided "
            "video, audio, and subtitle context.\n"
            "Compare only the two candidate answers below and decide which one is better.\n"
            "Candidate A:\n"
            f"{answer_a}\n\n"
            "Candidate B:\n"
            f"{answer_b}\n\n"
            f"Subtitle:\n{subtitle}\n\n"
            "Output exactly one character: A or B."
        )

    def _build_messages(self, video_array, audio_array, subtitle, answer_a, answer_b, audio_sampling_rate):
        prompt_text = self._build_prompt_text(subtitle, answer_a, answer_b)
        return [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_array},
                    {"type": "audio", "audio": audio_array},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

    def _parse_verdict(self, generated_text):
        text = str(generated_text).strip().upper()
        if text in {"A", "B"}:
            return text

        match = re.search(r"\b([AB])\b", text)
        if match is not None:
            return match.group(1)

        raise ValueError(f"Invalid judge verdict: {generated_text!r}")

    def compare_pair(self, video_array, audio_array, subtitle, answer_a, answer_b, audio_sampling_rate=16000):
        self._ensure_loaded()
        messages = self._build_messages(
            video_array=video_array,
            audio_array=audio_array,
            subtitle=subtitle,
            answer_a=answer_a,
            answer_b=answer_b,
            audio_sampling_rate=audio_sampling_rate,
        )
        model_inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            sampling_rate=audio_sampling_rate,
        )
        model_inputs = self._move_inputs(model_inputs)

        with torch.no_grad():
            generated_ids = self._model.generate(
                **model_inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        input_len = model_inputs["input_ids"].shape[1]
        new_tokens = generated_ids[:, input_len:]
        generated_text = self._processor.batch_decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return self._parse_verdict(generated_text)

    def compare_pairs(self, pair_inputs, audio_sampling_rate=16000):
        verdicts = []
        for pair_input in pair_inputs:
            try:
                verdicts.append(
                    self.compare_pair(
                        video_array=pair_input["video_array"],
                        audio_array=pair_input["audio_array"],
                        subtitle=pair_input["subtitle"],
                        answer_a=pair_input["answer_a"],
                        answer_b=pair_input["answer_b"],
                        audio_sampling_rate=audio_sampling_rate,
                    )
                )
            except ValueError:
                verdicts.append(None)
        return verdicts
