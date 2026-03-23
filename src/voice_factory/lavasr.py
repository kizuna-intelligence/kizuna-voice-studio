from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf
import torch


class LavaSRUnavailableError(RuntimeError):
    pass


@dataclass
class LavaSRConfig:
    model_id: str = "YatharthS/LavaSR"
    device: str = "auto"
    output_sample_rate: int = 48000
    denoise: bool = False
    batch: bool = False
    cache_dir: str | None = None


class LavaSREnhancer:
    def __init__(self, config: LavaSRConfig | None = None) -> None:
        self.config = config or LavaSRConfig()
        self.device = self._resolve_device(self.config.device)
        self._model = self._load_model()

    def enhance_file(self, input_path: Path, output_path: Path) -> Path:
        input_path = input_path.expanduser().resolve()
        output_path = output_path.expanduser().resolve()
        input_sr = self._detect_input_sample_rate(input_path)
        wav, _ = self._model.load_audio(str(input_path), input_sr=input_sr)
        enhanced = self._model.enhance(
            wav,
            denoise=self.config.denoise,
            batch=self.config.batch,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(
            str(output_path),
            enhanced.detach().cpu().numpy().squeeze(),
            self.config.output_sample_rate,
        )
        return output_path

    def _load_model(self):
        try:
            from LavaSR.model import LavaEnhance2
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise LavaSRUnavailableError(
                "LavaSR is not installed. Install training extras with "
                "`pip install -e .[train]` or install "
                "`git+https://github.com/ysharma3501/LavaSR.git`."
            ) from exc
        model_path = self._resolve_model_path(snapshot_download)
        return LavaEnhance2(model_path, self.device)

    def _resolve_model_path(self, snapshot_download) -> str:
        configured = Path(self.config.model_id).expanduser()
        if configured.exists():
            return str(configured.resolve())

        cache_root = self._cache_root()
        cache_root.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        try:
            return snapshot_download(
                repo_id=self.config.model_id,
                local_dir=str(cache_root),
                local_dir_use_symlinks=False,
            )
        except TypeError:
            # Older huggingface_hub releases may not expose local_dir_use_symlinks.
            return snapshot_download(
                repo_id=self.config.model_id,
                local_dir=str(cache_root),
            )

    def _cache_root(self) -> Path:
        if self.config.cache_dir:
            return Path(self.config.cache_dir).expanduser().resolve()
        return Path.home() / ".cache" / "voice-factory" / "lavasr" / self._safe_model_cache_name()

    def _safe_model_cache_name(self) -> str:
        return self.config.model_id.replace("\\", "_").replace("/", "__").replace(":", "_")

    def _resolve_device(self, requested: str) -> str:
        normalized = (requested or "auto").strip().lower()
        if normalized == "cpu":
            return "cpu"
        if normalized == "cuda":
            if torch.cuda.is_available():
                return "cuda"
            return "cpu"
        if normalized not in {"", "auto"}:
            return normalized
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _detect_input_sample_rate(self, input_path: Path) -> int:
        info = sf.info(str(input_path))
        sample_rate = int(info.samplerate)
        return min(48000, max(8000, sample_rate))
