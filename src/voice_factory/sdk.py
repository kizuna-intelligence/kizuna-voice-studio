from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import VoiceFactoryConfig
from .models import VoiceProjectSpec
from .service import VoiceFactoryService


class VoiceFactory:
    def __init__(self, config: VoiceFactoryConfig | None = None) -> None:
        self.config = config or VoiceFactoryConfig()
        self._service = VoiceFactoryService(
            workspace_root=self.config.resolved_workspace_root(),
        )

    @property
    def workspace_root(self) -> Path:
        return self._service.workspace_root

    @property
    def service(self) -> VoiceFactoryService:
        return self._service

    def list_projects(self) -> list[str]:
        return self._service.list_projects()

    def describe_project(self, project_id: str) -> dict[str, Any]:
        return self._service.describe_project(project_id)

    def plan_project(self, spec: VoiceProjectSpec | dict[str, Any]) -> dict[str, Any]:
        return self._service.plan_project(self._coerce_spec(spec))

    def create_simple_project(
        self,
        *,
        style_instruction: str,
        gpu_memory_gb: int,
        model_family: str = "piper",
        seed_voice_backend: str = "kizuna",
    ) -> dict[str, Any]:
        return self._service.create_simple_project(
            style_instruction=style_instruction,
            gpu_memory_gb=gpu_memory_gb,
            model_family=model_family,
            seed_voice_backend=seed_voice_backend,
        )

    def generate_preview(self, project_id: str) -> dict[str, Any]:
        return self._service.generate_preview(project_id)

    def build_dataset(
        self,
        project_id: str,
        *,
        mio_base_url: str,
        use_preview_reference: bool = True,
    ) -> dict[str, Any]:
        return self._service.build_dataset(
            project_id,
            mio_base_url=mio_base_url,
            use_preview_reference=use_preview_reference,
        )

    def train_piper(self, project_id: str, *, execute: bool = False) -> dict[str, Any]:
        return self._service.train_piper(project_id, execute=execute)

    def train_sbv2(
        self,
        project_id: str,
        *,
        execute: bool = False,
        style_bert_vits2_root: str | None = None,
    ) -> dict[str, Any]:
        return self._service.train_sbv2(
            project_id,
            execute=execute,
            style_bert_vits2_root=style_bert_vits2_root,
        )

    def build_piper_package(self, project_id: str) -> dict[str, Any]:
        return self._service.build_installable_package(project_id)

    def build_sbv2_package(
        self,
        project_id: str,
        *,
        style_bert_vits2_root: str | None = None,
    ) -> dict[str, Any]:
        return self._service.build_installable_sbv2_package(
            project_id,
            style_bert_vits2_root=style_bert_vits2_root,
        )

    def build_miotts_package(
        self,
        project_id: str,
        *,
        mio_base_url: str | None = None,
        model_id: str | None = None,
        reference_audio_paths: list[str | Path] | None = None,
    ) -> dict[str, Any]:
        return self._service.build_installable_miotts_package(
            project_id,
            mio_base_url=mio_base_url,
            model_id=model_id,
            reference_audio_paths=reference_audio_paths,
        )

    def build_irodori_package(
        self,
        project_id: str,
        *,
        model_id: str | None = None,
        reference_audio_paths: list[str | Path] | None = None,
    ) -> dict[str, Any]:
        return self._service.build_installable_irodori_package(
            project_id,
            model_id=model_id,
            reference_audio_paths=reference_audio_paths,
        )

    def build_miotts_previews(
        self,
        project_id: str,
        *,
        texts: list[str] | None = None,
    ) -> dict[str, Any]:
        return self._service.build_miotts_package_previews(project_id, texts=texts)

    def build_package_previews(
        self,
        project_id: str,
        *,
        family: str,
        texts: list[str] | None = None,
        compute_target: str = "auto",
    ) -> dict[str, Any]:
        return self._service.build_generated_package_previews(
            project_id,
            family=family,
            texts=texts,
            compute_target=compute_target,
        )

    @staticmethod
    def _coerce_spec(spec: VoiceProjectSpec | dict[str, Any]) -> VoiceProjectSpec:
        if isinstance(spec, VoiceProjectSpec):
            return spec
        return VoiceProjectSpec.from_dict(spec)
