from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class VoiceProjectSpec:
    project_name: str
    project_id: str
    speaker_name: str
    style_instruction: str
    seed_text: str
    gpu_memory_gb: int = 16
    language: str = "Japanese"
    seed_voice_backend: str = "kizuna"
    seed_voice_model: str = "kizuna-intelligence/kizuna-voice-designer"
    seed_voice_embedding_mode: str = "local_lightweight"
    seed_voice_gguf_model: str = "Qwen/Qwen3-Embedding-4B-GGUF"
    seed_voice_gguf_file: str = "Qwen3-Embedding-4B-Q8_0.gguf"
    style_instruction_zh: str | None = None
    qwen_model_id: str = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    translation_model_id: str = "NiuTrans/LMT-60-4B"
    mio_model_label: str = "Aratako/MioTTS-1.7B"
    target_model_family: str = "piper"
    piper_base_checkpoint_repo: str = "ayousanz/piper-plus-base"
    piper_base_checkpoint_filename: str = "model.ckpt"
    sbv2_pretrained_variant: str = "jp_extra"
    mio_reference_preset_id: str | None = None
    prompt_categories: list[str] = field(
        default_factory=lambda: ["emotional", "neutral", "news", "academic"]
    )
    items_per_category: int = 50

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VoiceProjectSpec":
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProjectPaths:
    root: Path
    project_dir: Path
    artifacts_dir: Path
    scripts_dir: Path
    training_output_dir: Path
    output_dir: Path
    distribution_dir: Path
    preview_dir: Path
    raw_dataset_dir: Path
    piper_dir: Path
    piper_ljspeech_dir: Path
    piper_preprocessed_dir: Path
    sbv2_dir: Path
    sbv2_data_dir: Path

    def to_dict(self) -> dict[str, str]:
        return {key: str(value) for key, value in asdict(self).items()}
