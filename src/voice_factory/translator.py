from __future__ import annotations

import os
from typing import Any


class PromptTranslator:
    def __init__(self) -> None:
        self._cache: dict[str, tuple[Any, Any]] = {}

    def translate_ja_to_zh(self, text: str, *, model_id: str) -> str:
        cleaned = text.strip()
        if not cleaned:
            return ""

        device_preference = os.environ.get("VOICE_FACTORY_TRANSLATOR_DEVICE", "auto").strip().lower() or "auto"
        try:
            tokenizer, model = self._load_model(
                model_id,
                device_preference=device_preference,
                load_in_4bit=self._env_flag("VOICE_FACTORY_TRANSLATOR_LOAD_IN_4BIT", default=True),
            )
            return self._translate_with_model(tokenizer, model, cleaned)
        except Exception as exc:
            if device_preference.startswith("cuda") or device_preference in {"gpu", "auto"}:
                if exc.__class__.__name__ in {"OutOfMemoryError"} or "out of memory" in str(exc).lower():
                    self._cache.pop(
                        self._cache_key(
                            model_id,
                            device_preference,
                            load_in_4bit=self._env_flag(
                                "VOICE_FACTORY_TRANSLATOR_LOAD_IN_4BIT",
                                default=True,
                            ),
                        ),
                        None,
                    )
                    tokenizer, model = self._load_model(
                        model_id,
                        device_preference="cpu",
                        load_in_4bit=False,
                    )
                    return self._translate_with_model(tokenizer, model, cleaned)
            raise

    def _env_flag(self, name: str, *, default: bool) -> bool:
        value = os.environ.get(name)
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "on", "y"}

    def _cache_key(self, model_id: str, device_preference: str, *, load_in_4bit: bool) -> str:
        return f"{model_id}::{device_preference}::4bit={int(load_in_4bit)}"

    def _translate_with_model(self, tokenizer: Any, model: Any, text: str) -> str:
        prompt = (
            "Translate the following voice-design instruction from Japanese into natural Chinese. "
            "Keep speaker attributes, speaking style, pacing, and tone intact.\n"
            f"Japanese: {text}\n"
            "Chinese:"
        )
        messages = [{"role": "user", "content": prompt}]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([rendered], return_tensors="pt")
        model_device = next(model.parameters()).device
        model_inputs = {key: value.to(model_device) for key, value in model_inputs.items()}
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=192,
            num_beams=1,
            do_sample=False,
        )
        output_ids = generated_ids[0][len(model_inputs["input_ids"][0]) :].tolist()
        return tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    def _load_model(
        self,
        model_id: str,
        *,
        device_preference: str,
        load_in_4bit: bool,
    ) -> tuple[Any, Any]:
        cache_key = self._cache_key(model_id, device_preference, load_in_4bit=load_in_4bit)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError("Prompt translation requires transformers and torch") from exc

        normalized_device = device_preference
        if normalized_device in {"gpu", "cuda"}:
            normalized_device = "cuda"
        elif normalized_device == "auto":
            normalized_device = "cuda" if torch.cuda.is_available() else "cpu"
        elif normalized_device not in {"cpu", "cuda"}:
            normalized_device = "cpu"

        model_kwargs: dict[str, Any] = {"device_map": normalized_device, "low_cpu_mem_usage": True}
        if normalized_device == "cuda":
            if load_in_4bit:
                model_kwargs["device_map"] = "auto"
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                model_kwargs["dtype"] = torch.float16
            else:
                model_kwargs["dtype"] = torch.float16
        else:
            model_kwargs["dtype"] = torch.float32

        tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        self._cache[cache_key] = (tokenizer, model)
        return tokenizer, model
