from __future__ import annotations

import base64
import json
from pathlib import Path


def export_standalone_piper_module(
    *,
    onnx_path: Path,
    config_path: Path,
    output_path: Path,
    module_name: str = "portable_voice",
    class_name: str = "PortablePiperVoice",
) -> Path:
    """Emit a single-file Python module with embedded ONNX + config assets."""
    onnx_bytes = onnx_path.read_bytes()
    config_json = json.loads(config_path.read_text(encoding="utf-8"))

    onnx_b64 = base64.b64encode(onnx_bytes).decode("ascii")
    config_b64 = base64.b64encode(
        json.dumps(config_json, ensure_ascii=False).encode("utf-8")
    ).decode("ascii")

    module_source = f'''"""Standalone generated voice module: {module_name}."""

from __future__ import annotations

import base64
import json
import tempfile
import wave
from pathlib import Path

import numpy as np
import onnxruntime
from piper_train.phonemize.japanese import phonemize_japanese_with_prosody
from piper_train.vits.utils import audio_float_to_int16

_EMBEDDED_ONNX = """{onnx_b64}"""
_EMBEDDED_CONFIG = """{config_b64}"""


class {class_name}:
    def __init__(self, providers=None):
        self._workdir = Path(tempfile.mkdtemp(prefix="{module_name}_"))
        self._model_path = self._workdir / "{module_name}.onnx"
        self._config_path = self._workdir / "{module_name}.json"
        self._model_path.write_bytes(base64.b64decode(_EMBEDDED_ONNX))
        self._config_path.write_bytes(base64.b64decode(_EMBEDDED_CONFIG))
        self.config = json.loads(self._config_path.read_text(encoding="utf-8"))
        self.sample_rate = int(self.config["audio"]["sample_rate"])
        self._input_names = set()
        self._session = onnxruntime.InferenceSession(
            str(self._model_path),
            providers=providers or self._default_providers(),
        )
        self._input_names = {{inp.name for inp in self._session.get_inputs()}}

    def _default_providers(self):
        available = onnxruntime.get_available_providers()
        preferred = [
            provider
            for provider in ("CUDAExecutionProvider", "CPUExecutionProvider")
            if provider in available
        ]
        return preferred or available

    def _phonemize(self, text: str):
        phonemes, prosody_info_list = phonemize_japanese_with_prosody(text)
        phoneme_id_map = self.config["phoneme_id_map"]
        phoneme_ids = []
        prosody_features = []
        missing = []
        for phoneme, prosody_info in zip(phonemes, prosody_info_list):
            ids = phoneme_id_map.get(phoneme)
            if ids is None:
                missing.append(phoneme)
                continue
            phoneme_ids.extend(ids)
            for _ in ids:
                if prosody_info is None:
                    prosody_features.append([0, 0, 0])
                else:
                    prosody_features.append([prosody_info.a1, prosody_info.a2, prosody_info.a3])
        if missing:
            missing_repr = ", ".join(sorted(repr(token) for token in set(missing)))
            raise ValueError(f"Missing tokens in phoneme_id_map: {{missing_repr}}")
        if not phoneme_ids:
            raise ValueError("No phoneme IDs generated from input text.")
        return phoneme_ids, prosody_features

    def synthesize(
        self,
        text: str,
        *,
        noise_scale: float | None = None,
        length_scale: float | None = None,
        noise_w: float | None = None,
    ) -> np.ndarray:
        phoneme_ids, prosody_features = self._phonemize(text)
        phoneme_ids_array = np.expand_dims(np.array(phoneme_ids, dtype=np.int64), 0)
        phoneme_ids_lengths = np.array([phoneme_ids_array.shape[1]], dtype=np.int64)
        scales = np.array(
            [
                noise_scale if noise_scale is not None else self.config["inference"]["noise_scale"],
                length_scale if length_scale is not None else self.config["inference"]["length_scale"],
                noise_w if noise_w is not None else self.config["inference"]["noise_w"],
            ],
            dtype=np.float32,
        )
        input_feed = {{
            "input": phoneme_ids_array,
            "input_lengths": phoneme_ids_lengths,
            "scales": scales,
        }}
        if "prosody_features" in self._input_names:
            input_feed["prosody_features"] = np.expand_dims(
                np.array(prosody_features, dtype=np.int64), 0
            )
        unsupported = self._input_names - set(input_feed.keys())
        if unsupported:
            unsupported_names = ", ".join(sorted(unsupported))
            raise RuntimeError(f"Model requires unsupported inputs: {{unsupported_names}}")
        audio = self._session.run(
            None,
            input_feed,
        )[0].squeeze()
        if audio.ndim != 1:
            raise RuntimeError(f"Unexpected audio tensor shape after squeeze: {{audio.shape}}")
        return np.asarray(audio, dtype=np.float32)

    def save_wav(self, text: str, output_path: str | Path, **kwargs) -> Path:
        output_path = Path(output_path)
        audio = self.synthesize(text, **kwargs)
        audio_int16 = audio_float_to_int16(audio)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(output_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_int16.tobytes())
        return output_path


def load_voice(*, providers=None) -> {class_name}:
    return {class_name}(providers=providers)
'''

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(module_source, encoding="utf-8")
    return output_path
