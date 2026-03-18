from __future__ import annotations

import audioop
import importlib.util
import json
import os
import pathlib
import shutil
import subprocess
import sys
import traceback
import wave
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import httpx
import torch
from huggingface_hub import hf_hub_download, snapshot_download

from .exporter import export_standalone_piper_module
from .models import ProjectPaths, VoiceProjectSpec
from .translator import PromptTranslator

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[1]
WORKSPACE_ROOT = Path(os.environ.get("VOICE_FACTORY_WORKSPACE_ROOT", str(REPO_ROOT / "workspace"))).expanduser().resolve()
PROJECTS_ROOT = WORKSPACE_ROOT / "projects"
VOICE_DESIGNER_CACHE_DIR = Path.home() / ".cache" / "kizuna-voice-designer"


class VoiceFactoryService:
    _DEFAULT_MIO_BASE_URL = os.environ.get(
        "VOICE_FACTORY_MIO_BASE_URL",
        "https://miotts-hybrid-gngpt3r4wq-as.a.run.app",
    )
    _PIPER_PACKAGE_NAME = "piper-voice"
    _PIPER_MODULE_NAME = "piper_voice"
    _MIOTTS_PACKAGE_NAME = "miotts-reference-voice"
    _MIOTTS_MODULE_NAME = "miotts_reference_voice"
    _SBV2_PACKAGE_NAME = "style-bert-vits2-voice"
    _SBV2_MODULE_NAME = "style_bert_vits2_voice"

    _PROMPT_VARIANT_SUFFIXES: dict[str, list[str]] = {
        "emotional": [
            "",
            " それでも前を向いて、次の一歩を踏み出したいと思います。",
            " その瞬間の空気まで鮮やかによみがえってきます。",
            " だからこそ、今ここで気持ちを込めて伝えたいです。",
            " 小さな変化でも、心の動きははっきりと感じられます。",
        ],
        "neutral": [
            "",
            " 基本的な仕組みを知っておくと、理解がより深まります。",
            " 日常の中でも、同じ考え方が役立つ場面は少なくありません。",
            " こうした特徴を押さえることで、判断しやすくなります。",
            " まずは身近な例から考えてみると分かりやすいです。",
        ],
        "news": [
            "",
            " 現地では引き続き、慎重な対応が求められています。",
            " 関係機関は詳しい状況の確認を進めています。",
            " 今後の発表に注意しながら、最新情報を確認してください。",
            " 住民生活への影響についても、幅広く調査が進められています。",
        ],
        "academic": [
            "",
            " この観点は、関連する研究分野にも広く応用されています。",
            " 背景にある前提を整理すると、議論の見通しが良くなります。",
            " 実証結果を丁寧に読むことが、妥当な解釈につながります。",
            " 理論と実践の両面から検討することが重要です。",
        ],
    }
    _PROMPT_VARIANT_ROUND_TAILS: list[str] = [
        "",
        " 具体例を一つ加えながら、落ち着いて説明します。",
        " 重要な点を一つずつ区切りながら、自然に読み上げます。",
        " 少し言い回しを変えつつ、同じ内容を分かりやすく伝えます。",
        " 聞き手に伝わりやすいように、抑揚を保ちながら丁寧に話します。",
        " 全体の流れが見えやすいように、要点を整理して伝えます。",
    ]
    _MIOTTS_PREVIEW_TEXTS: list[tuple[str, str]] = [
        ("sample_01", "こんにちは。今日は種音声をそのまま使う、MioTTS パッケージの試聴です。"),
        ("sample_02", "この音声は、学習をせずに参照音声だけを同封したパッケージから生成しています。"),
        ("sample_03", "落ち着いた読み上げや、アプリへの組み込み前の確認に使えるサンプルです。"),
    ]
    _PACKAGE_PREVIEW_TEXTS: list[tuple[str, str]] = [
        ("sample_01", "こんにちは。これはパッケージ化した音声の動作確認です。"),
        ("sample_02", "自由に入力した文章を、そのまま読み上げ品質の確認に使えます。"),
        ("sample_03", "アプリへ組み込む前に、話し方や聞き取りやすさをここで試せます。"),
    ]

    def __init__(self, *, workspace_root: Path | None = None) -> None:
        self.workspace_root = workspace_root or WORKSPACE_ROOT
        self.projects_root = self.workspace_root / "projects"
        self.jobs_root = self.workspace_root / "jobs"
        self.projects_root.mkdir(parents=True, exist_ok=True)
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        self.prompt_bank = json.loads((PACKAGE_DIR / "prompt_bank.json").read_text(encoding="utf-8"))
        self.translator = PromptTranslator()

    def _timestamp_token(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")

    def _short_label(self, text: str, default: str = "custom-voice", max_len: int = 32) -> str:
        keep = []
        prev_dash = False
        for ch in text.lower():
            if ("a" <= ch <= "z") or ("0" <= ch <= "9"):
                keep.append(ch)
                prev_dash = False
            elif not prev_dash:
                keep.append("-")
                prev_dash = True
        label = "".join(keep).strip("-")
        return (label[:max_len].rstrip("-") or default)

    def _normalize_model_family(self, model_family: str | None) -> str:
        normalized = (model_family or "piper").strip().lower()
        if normalized not in {"piper", "sbv2"}:
            raise ValueError(f"Unsupported model family: {model_family}")
        return normalized

    def _normalize_seed_voice_backend(self, seed_voice_backend: str | None) -> str:
        normalized = (seed_voice_backend or "kizuna").strip().lower()
        if normalized not in {"kizuna", "qwen"}:
            raise ValueError(f"Unsupported seed voice backend: {seed_voice_backend}")
        return normalized

    def default_mio_base_url(self) -> str:
        return self._DEFAULT_MIO_BASE_URL

    def _normalize_compute_target(self, compute_target: str | None) -> str:
        normalized = (compute_target or "auto").strip().lower()
        if normalized in {"", "auto"}:
            return "auto"
        if normalized == "cpu":
            return "cpu"
        if normalized.startswith("gpu:"):
            gpu_id = normalized.split(":", 1)[1]
        elif normalized.startswith("gpu"):
            gpu_id = normalized[3:]
        else:
            raise ValueError(f"Unsupported compute target: {compute_target}")
        if not gpu_id.isdigit():
            raise ValueError(f"Unsupported compute target: {compute_target}")
        return f"gpu:{gpu_id}"

    def _preview_device_for_compute_target(self, compute_target: str | None) -> str:
        normalized = self._normalize_compute_target(compute_target)
        if normalized == "cpu":
            return "cpu"
        if normalized.startswith("gpu:"):
            return "cuda"
        return "auto"

    def _apply_compute_target_to_env(self, env: dict[str, str], compute_target: str | None) -> None:
        normalized = self._normalize_compute_target(compute_target)
        env["VOICE_FACTORY_COMPUTE_TARGET"] = normalized
        env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        if normalized == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""
        elif normalized.startswith("gpu:"):
            env["CUDA_VISIBLE_DEVICES"] = normalized.split(":", 1)[1]

    def list_compute_targets(self) -> list[dict[str, str]]:
        targets = [
            {
                "value": "auto",
                "label": "自動で選ぶ",
                "description": "利用可能な GPU を自動で使います。",
            },
            {
                "value": "cpu",
                "label": "CPU",
                "description": "GPU を使わず CPU で実行します。",
            },
        ]
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            return targets

        for raw_line in result.stdout.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            parts = [part.strip() for part in line.split(",", maxsplit=4)]
            if len(parts) != 5:
                continue
            gpu_id, name, memory_total, memory_used, utilization = parts
            targets.append(
                {
                    "value": f"gpu:{gpu_id}",
                    "label": f"GPU {gpu_id}",
                    "description": f"{name} / {memory_used} MiB 使用中 / {memory_total} MiB / 利用率 {utilization}%",
                }
            )
        return targets

    def _env_int(self, name: str, default: int) -> int:
        value = os.environ.get(name)
        if value is None:
            return default
        try:
            return int(value)
        except ValueError:
            return default

    def _env_flag(self, name: str, default: bool = False) -> bool:
        value = os.environ.get(name)
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "on", "y"}

    def create_simple_project(
        self,
        *,
        style_instruction: str,
        gpu_memory_gb: int,
        model_family: str = "piper",
        seed_voice_backend: str = "kizuna",
    ) -> dict[str, Any]:
        token = self._timestamp_token()
        label = self._short_label(style_instruction)
        project_id = f"voice-{token}-{label}"
        speaker_name = f"voice_{token}"
        normalized_model_family = self._normalize_model_family(model_family)
        spec = VoiceProjectSpec(
            project_name=f"Voice {token}",
            project_id=project_id,
            speaker_name=speaker_name,
            style_instruction=style_instruction.strip(),
            seed_text="こんにちは。こちらは新しい声の確認用サンプルです。自然で聞き取りやすく読み上げます。",
            gpu_memory_gb=max(4, int(gpu_memory_gb)),
            seed_voice_backend=self._normalize_seed_voice_backend(seed_voice_backend),
            target_model_family=normalized_model_family,
            items_per_category=max(1, self._env_int("VOICE_FACTORY_ITEMS_PER_CATEGORY", 50)),
        )
        planned = self.plan_project(spec)
        planned["recommended_training_profile"] = self.recommend_training_profile(spec)
        return planned

    def start_simple_preview_job(
        self,
        *,
        style_instruction: str,
        gpu_memory_gb: int,
        model_family: str = "piper",
        seed_voice_backend: str = "kizuna",
        compute_target: str = "auto",
    ) -> dict[str, Any]:
        planned = self.create_simple_project(
            style_instruction=style_instruction,
            gpu_memory_gb=gpu_memory_gb,
            model_family=model_family,
            seed_voice_backend=seed_voice_backend,
        )
        project_id = planned["project"]["project_id"]
        job = self.start_job(
            job_type="generate-preview",
            project_id=project_id,
            params={"compute_target": self._normalize_compute_target(compute_target)},
        )
        return {
            "project": planned["project"],
            "recommended_training_profile": planned["recommended_training_profile"],
            "job": job,
        }

    def start_dataset_pipeline_job(
        self,
        *,
        project_id: str,
        mio_base_url: str,
        use_preview_reference: bool = True,
    ) -> dict[str, Any]:
        return self.start_job(
            job_type="approve-preview",
            project_id=project_id,
            params={
                "mio_base_url": mio_base_url,
                "use_preview_reference": use_preview_reference,
            },
        )

    def start_one_click_job(
        self,
        *,
        style_instruction: str,
        gpu_memory_gb: int,
        mio_base_url: str,
        model_family: str = "piper",
        seed_voice_backend: str = "kizuna",
        compute_target: str = "auto",
    ) -> dict[str, Any]:
        planned = self.create_simple_project(
            style_instruction=style_instruction,
            gpu_memory_gb=gpu_memory_gb,
            model_family=model_family,
            seed_voice_backend=seed_voice_backend,
        )
        project_id = planned["project"]["project_id"]
        job = self.start_job(
            job_type="build-tts-product",
            project_id=project_id,
            params={
                "mio_base_url": mio_base_url,
                "use_preview_reference": True,
                "model_family": self._normalize_model_family(model_family),
                "compute_target": self._normalize_compute_target(compute_target),
            },
        )
        return {
            "project": planned["project"],
            "recommended_training_profile": planned["recommended_training_profile"],
            "job": job,
        }

    def start_build_tts_job(
        self,
        *,
        project_id: str,
        mio_base_url: str,
        model_family: str = "piper",
        compute_target: str = "auto",
    ) -> dict[str, Any]:
        spec = self.load_project(project_id)
        spec.target_model_family = self._normalize_model_family(model_family)
        self._save_project(spec)
        return self.start_job(
            job_type="build-tts-product",
            project_id=project_id,
            params={
                "mio_base_url": mio_base_url,
                "use_preview_reference": True,
                "skip_preview": True,
                "model_family": spec.target_model_family,
                "compute_target": self._normalize_compute_target(compute_target),
            },
        )

    def recommend_training_profile(self, spec: VoiceProjectSpec) -> dict[str, Any]:
        vram = max(4, int(spec.gpu_memory_gb))
        if vram <= 8:
            piper_batch_size = 1
            sbv2_batch_size = 1
        elif vram <= 12:
            piper_batch_size = 2
            sbv2_batch_size = 2
        elif vram <= 16:
            piper_batch_size = 8
            sbv2_batch_size = 3
        elif vram <= 24:
            piper_batch_size = 8
            sbv2_batch_size = 4
        else:
            piper_batch_size = 12
            sbv2_batch_size = 6
        piper_use_wavlm = vram >= 12
        return {
            "gpu_memory_gb": vram,
            "piper_batch_size": piper_batch_size,
            "sbv2_batch_size": sbv2_batch_size,
            "sbv2_preprocess_batch_size": max(2, sbv2_batch_size),
            "piper_use_wavlm_discriminator": piper_use_wavlm,
            "notes": [
                "Uses the same Piper-plus fine-tune recipe as the proven Mio experiment by default.",
                "The reference recipe is 200 utterances, batch size 8, WavLM discriminator enabled, and 300 epochs.",
                "If the selected GPU is smaller than the reference 16GB setup, the batch size is reduced automatically.",
                "WavLM stays enabled on 12GB-and-up GPUs unless VOICE_FACTORY_PIPER_DISABLE_WAVLM is set.",
            ],
        }

    def list_projects(self) -> list[str]:
        return sorted(path.name for path in self.projects_root.iterdir() if path.is_dir())

    def _save_project(self, spec: VoiceProjectSpec) -> Path:
        project_file = self.get_project_paths(spec.project_id).project_dir / "project.json"
        project_file.parent.mkdir(parents=True, exist_ok=True)
        project_file.write_text(json.dumps(spec.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return project_file

    def get_project_paths(self, project_id: str) -> ProjectPaths:
        project_dir = self.projects_root / project_id
        artifacts_dir = project_dir / "artifacts"
        scripts_dir = project_dir / "scripts"
        training_output_dir = project_dir / "training_output"
        output_dir = project_dir / "output"
        distribution_dir = artifacts_dir / "distribution"
        preview_dir = artifacts_dir / "voice_preview"
        raw_dataset_dir = artifacts_dir / "datasets" / "raw"
        piper_dir = artifacts_dir / "piper"
        piper_ljspeech_dir = piper_dir / "ljspeech_dataset"
        piper_preprocessed_dir = piper_dir / "preprocessed"
        sbv2_dir = artifacts_dir / "sbv2"
        sbv2_data_dir = sbv2_dir / "Data"
        return ProjectPaths(
            root=self.workspace_root,
            project_dir=project_dir,
            artifacts_dir=artifacts_dir,
            scripts_dir=scripts_dir,
            training_output_dir=training_output_dir,
            output_dir=output_dir,
            distribution_dir=distribution_dir,
            preview_dir=preview_dir,
            raw_dataset_dir=raw_dataset_dir,
            piper_dir=piper_dir,
            piper_ljspeech_dir=piper_ljspeech_dir,
            piper_preprocessed_dir=piper_preprocessed_dir,
            sbv2_dir=sbv2_dir,
            sbv2_data_dir=sbv2_data_dir,
        )

    def _preview_manifest_path(self, project_id: str) -> Path:
        return self.get_project_paths(project_id).preview_dir / "reference.json"

    def _preview_audio_path(self, project_id: str) -> Path:
        return self.get_project_paths(project_id).preview_dir / "reference.wav"

    def _dataset_manifest_path(self, project_id: str) -> Path:
        return self.get_project_paths(project_id).raw_dataset_dir / "manifest.json"

    def _package_manifest_path(self, project_id: str) -> Path:
        return self.get_project_paths(project_id).distribution_dir / "package.json"

    def _package_archive_path(self, project_id: str) -> Path:
        manifest = self.get_package_manifest(project_id)
        if manifest is None:
            return self.get_project_paths(project_id).distribution_dir / "package.zip"
        return Path(manifest["archive_path"])

    def _sbv2_package_manifest_path(self, project_id: str) -> Path:
        return self.get_project_paths(project_id).distribution_dir / "sbv2-package.json"

    def _miotts_package_manifest_path(self, project_id: str) -> Path:
        return self.get_project_paths(project_id).distribution_dir / "miotts-package.json"

    def _miotts_package_preview_manifest_path(self, project_id: str) -> Path:
        return self.get_project_paths(project_id).distribution_dir / "miotts-package-preview.json"

    def _package_preview_manifest_path(self, project_id: str) -> Path:
        return self.get_project_paths(project_id).distribution_dir / "package-preview.json"

    def _sbv2_package_preview_manifest_path(self, project_id: str) -> Path:
        return self.get_project_paths(project_id).distribution_dir / "sbv2-package-preview.json"

    def plan_project(self, spec: VoiceProjectSpec) -> dict[str, Any]:
        paths = self.get_project_paths(spec.project_id)
        for directory in (
            paths.project_dir,
            paths.artifacts_dir,
            paths.scripts_dir,
            paths.training_output_dir,
            paths.output_dir,
            paths.distribution_dir,
            paths.preview_dir,
            paths.raw_dataset_dir,
            paths.piper_ljspeech_dir / "wav",
            paths.sbv2_data_dir / spec.speaker_name / "raw",
        ):
            directory.mkdir(parents=True, exist_ok=True)

        project_file = self._save_project(spec)
        return {
            "project": spec.to_dict(),
            "paths": paths.to_dict(),
            "project_file": str(project_file),
        }

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _job_dir(self, job_id: str) -> Path:
        return self.jobs_root / job_id

    def _job_file(self, job_id: str) -> Path:
        return self._job_dir(job_id) / "job.json"

    def _job_result_file(self, job_id: str) -> Path:
        return self._job_dir(job_id) / "result.json"

    def _job_stdout_file(self, job_id: str) -> Path:
        return self._job_dir(job_id) / "stdout.log"

    def _job_stderr_file(self, job_id: str) -> Path:
        return self._job_dir(job_id) / "stderr.log"

    def _write_job_payload(self, job_id: str, payload: dict[str, Any]) -> None:
        self._job_dir(job_id).mkdir(parents=True, exist_ok=True)
        self._job_file(job_id).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _set_job_stage(
        self,
        payload: dict[str, Any],
        *,
        stage: str,
        stage_label: str,
    ) -> None:
        payload["stage"] = stage
        payload["stage_label_base"] = stage_label
        payload["stage_label"] = stage_label
        payload.pop("progress", None)
        payload.pop("progress_detail", None)
        payload.pop("progress_percent", None)
        self._write_job_payload(payload["job_id"], payload)

    def _set_job_progress(
        self,
        payload: dict[str, Any],
        *,
        current: int,
        total: int,
        detail: str | None = None,
    ) -> None:
        safe_total = max(1, int(total))
        safe_current = min(max(0, int(current)), safe_total)
        percent = int((safe_current / safe_total) * 100)
        payload["progress"] = {
            "current": safe_current,
            "total": safe_total,
        }
        payload["progress_detail"] = detail
        payload["progress_percent"] = percent
        base_label = payload.get("stage_label_base", payload.get("stage_label", "進行中です"))
        if detail:
            payload["stage_label"] = f"{base_label} ({safe_current}/{safe_total}) - {detail}"
        else:
            payload["stage_label"] = f"{base_label} ({safe_current}/{safe_total})"
        self._write_job_payload(payload["job_id"], payload)

    def list_jobs(self, project_id: str | None = None) -> list[dict[str, Any]]:
        jobs: list[dict[str, Any]] = []
        for job_dir in sorted(self.jobs_root.iterdir(), reverse=True):
            if not job_dir.is_dir():
                continue
            job_file = job_dir / "job.json"
            if not job_file.exists():
                continue
            payload = json.loads(job_file.read_text(encoding="utf-8"))
            if project_id and payload.get("project_id") != project_id:
                continue
            jobs.append(payload)
        return jobs

    def get_job(self, job_id: str) -> dict[str, Any]:
        payload = json.loads(self._job_file(job_id).read_text(encoding="utf-8"))
        result_file = self._job_result_file(job_id)
        if result_file.exists():
            payload["result"] = json.loads(result_file.read_text(encoding="utf-8"))
        payload["stdout_log"] = str(self._job_stdout_file(job_id))
        payload["stderr_log"] = str(self._job_stderr_file(job_id))
        return payload

    def start_job(
        self,
        *,
        job_type: str,
        project_id: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        job_id = f"{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid4().hex[:8]}"
        payload = {
            "job_id": job_id,
            "job_type": job_type,
            "project_id": project_id,
            "params": params or {},
            "status": "queued",
            "stage": "queued",
            "stage_label": "実行待ちです",
            "created_at": self._now_iso(),
        }
        self._write_job_payload(job_id, payload)

        env = os.environ.copy()
        self._apply_compute_target_to_env(env, (params or {}).get("compute_target"))
        src_path = str(REPO_ROOT / "src")
        env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else src_path + os.pathsep + env["PYTHONPATH"]
        env.setdefault("WANDB_MODE", "disabled")
        env.setdefault("WANDB_DISABLED", "true")
        env.setdefault("WANDB_SILENT", "true")

        stdout_handle = self._job_stdout_file(job_id).open("w", encoding="utf-8")
        stderr_handle = self._job_stderr_file(job_id).open("w", encoding="utf-8")
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "voice_factory.cli",
                "run-job",
                "--job-file",
                str(self._job_file(job_id)),
            ],
            cwd=str(REPO_ROOT),
            env=env,
            stdout=stdout_handle,
            stderr=stderr_handle,
            start_new_session=True,
        )
        stdout_handle.close()
        stderr_handle.close()

        payload["worker_pid"] = process.pid
        self._write_job_payload(job_id, payload)
        return self.get_job(job_id)

    def run_job_file(self, job_file: Path) -> dict[str, Any]:
        payload = json.loads(job_file.read_text(encoding="utf-8"))
        job_id = payload["job_id"]
        payload["status"] = "running"
        payload["started_at"] = self._now_iso()
        self._set_job_stage(payload, stage="starting", stage_label="処理を開始しています")

        try:
            params = payload.get("params", {})
            job_type = payload["job_type"]
            project_id = payload["project_id"]
            compute_target = self._normalize_compute_target(params.get("compute_target"))
            dataset_progress = lambda current, total, detail: self._set_job_progress(
                payload,
                current=current,
                total=total,
                detail=detail,
            )
            if job_type == "generate-preview":
                self._set_job_stage(payload, stage="preview", stage_label="種音声を作成中です")
                result = self.generate_preview(
                    project_id,
                    device=self._preview_device_for_compute_target(compute_target),
                    status_callback=lambda message: self._set_job_stage(
                        payload,
                        stage="preview",
                        stage_label=message,
                    ),
                )
            elif job_type == "approve-preview":
                self._set_job_stage(payload, stage="dataset", stage_label="学習データセットを作成中です")
                dataset_result = self.build_dataset(
                    project_id,
                    mio_base_url=params["mio_base_url"],
                    use_preview_reference=params.get("use_preview_reference", True),
                    progress_callback=dataset_progress,
                )
                self._set_job_stage(payload, stage="scripts", stage_label="学習設定を書き出しています")
                scripts_result = self.write_training_scripts(project_id)
                result = {
                    "project_id": project_id,
                    "dataset": dataset_result,
                    "training_scripts": scripts_result,
                    "recommended_training_profile": self.recommend_training_profile(self.load_project(project_id)),
                }
            elif job_type == "build-dataset":
                self._set_job_stage(payload, stage="dataset", stage_label="学習データセットを作成中です")
                result = self.build_dataset(
                    project_id,
                    mio_base_url=params["mio_base_url"],
                    use_preview_reference=params.get("use_preview_reference", True),
                    progress_callback=dataset_progress,
                )
            elif job_type == "bootstrap-project":
                self._set_job_stage(payload, stage="preview", stage_label="種音声を作成中です")
                preview_result = self.generate_preview(
                    project_id,
                    device=self._preview_device_for_compute_target(compute_target),
                    status_callback=lambda message: self._set_job_stage(
                        payload,
                        stage="preview",
                        stage_label=message,
                    ),
                )
                self._set_job_stage(payload, stage="dataset", stage_label="学習データセットを作成中です")
                dataset_result = self.build_dataset(
                    project_id,
                    mio_base_url=params["mio_base_url"],
                    use_preview_reference=params.get("use_preview_reference", True),
                    progress_callback=dataset_progress,
                )
                self._set_job_stage(payload, stage="scripts", stage_label="学習設定を書き出しています")
                scripts_result = self.write_training_scripts(project_id)
                result = {
                    "project_id": project_id,
                    "preview": preview_result,
                    "dataset": dataset_result,
                    "training_scripts": scripts_result,
                    "recommended_training_profile": self.recommend_training_profile(self.load_project(project_id)),
                }
            elif job_type == "build-tts-product":
                result = self.run_one_click_pipeline(
                    payload,
                    project_id=project_id,
                    mio_base_url=params["mio_base_url"],
                    use_preview_reference=params.get("use_preview_reference", True),
                    skip_preview=params.get("skip_preview", False),
                    model_family=params.get("model_family"),
                    compute_target=compute_target,
                )
            elif job_type == "prepare-piper":
                self._set_job_stage(payload, stage="preprocess", stage_label="学習の前処理をしています")
                result = self.prepare_piper(project_id)
            elif job_type == "train-piper":
                self._set_job_stage(payload, stage="training", stage_label="モデルを学習中です")
                result = self.train_piper(project_id, execute=True)
            elif job_type == "train-sbv2":
                self._set_job_stage(payload, stage="training", stage_label="SBV2 モデルを学習中です")
                result = self.train_sbv2(
                    project_id,
                    execute=True,
                    style_bert_vits2_root=params.get("style_bert_vits2_root"),
                )
            elif job_type == "build-sbv2-package":
                self._set_job_stage(payload, stage="package", stage_label="SBV2 Python パッケージをまとめています")
                result = self.build_installable_sbv2_package(
                    project_id,
                    style_bert_vits2_root=params.get("style_bert_vits2_root"),
                )
            elif job_type == "write-training-scripts":
                self._set_job_stage(payload, stage="scripts", stage_label="学習設定を書き出しています")
                result = self.write_training_scripts(project_id)
            else:
                raise ValueError(f"Unsupported job_type: {job_type}")

            self._job_result_file(job_id).write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            payload["status"] = "completed"
            payload["completed_at"] = self._now_iso()
            payload["stage"] = "completed"
            payload["stage_label"] = "完了しました"
            self._write_job_payload(job_id, payload)
            return result
        except Exception as exc:
            self._job_result_file(job_id).write_text(
                json.dumps(
                    {
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            payload["status"] = "failed"
            payload["completed_at"] = self._now_iso()
            payload["stage"] = "failed"
            payload["stage_label"] = "失敗しました"
            self._write_job_payload(job_id, payload)
            raise

    def load_project(self, project_id: str) -> VoiceProjectSpec:
        project_file = self.get_project_paths(project_id).project_dir / "project.json"
        return VoiceProjectSpec.from_dict(json.loads(project_file.read_text(encoding="utf-8")))

    def _translate_style_instruction(self, spec: VoiceProjectSpec) -> str:
        translated = (spec.style_instruction_zh or "").strip()
        if translated:
            return translated
        translated = self.translator.translate_ja_to_zh(
            spec.style_instruction,
            model_id=spec.translation_model_id,
        )
        spec.style_instruction_zh = translated
        self._save_project(spec)
        return translated

    def get_preview_manifest(self, project_id: str) -> dict[str, Any] | None:
        manifest_path = self._preview_manifest_path(project_id)
        if not manifest_path.exists():
            return None
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    def get_package_manifest(self, project_id: str) -> dict[str, Any] | None:
        manifest_path = self._package_manifest_path(project_id)
        if not manifest_path.exists():
            return None
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    def get_sbv2_package_manifest(self, project_id: str) -> dict[str, Any] | None:
        manifest_path = self._sbv2_package_manifest_path(project_id)
        if not manifest_path.exists():
            return None
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    def get_miotts_package_manifest(self, project_id: str) -> dict[str, Any] | None:
        manifest_path = self._miotts_package_manifest_path(project_id)
        if not manifest_path.exists():
            return None
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    def get_miotts_package_preview_manifest(self, project_id: str) -> dict[str, Any] | None:
        manifest_path = self._miotts_package_preview_manifest_path(project_id)
        if not manifest_path.exists():
            return None
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    def get_package_preview_manifest(self, project_id: str) -> dict[str, Any] | None:
        manifest_path = self._package_preview_manifest_path(project_id)
        if not manifest_path.exists():
            return None
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    def get_sbv2_package_preview_manifest(self, project_id: str) -> dict[str, Any] | None:
        manifest_path = self._sbv2_package_preview_manifest_path(project_id)
        if not manifest_path.exists():
            return None
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    def get_preview_audio_path(self, project_id: str) -> Path:
        return self._preview_audio_path(project_id)

    def describe_project(self, project_id: str) -> dict[str, Any]:
        spec = self.load_project(project_id)
        preview = self.get_preview_manifest(project_id)
        package_manifest = self.get_package_manifest(project_id)
        package_preview_manifest = self.get_package_preview_manifest(project_id)
        sbv2_package_manifest = self.get_sbv2_package_manifest(project_id)
        sbv2_package_preview_manifest = self.get_sbv2_package_preview_manifest(project_id)
        miotts_package_manifest = self.get_miotts_package_manifest(project_id)
        miotts_package_preview_manifest = self.get_miotts_package_preview_manifest(project_id)
        dataset_manifest_path = self._dataset_manifest_path(project_id)
        item_count = None
        if dataset_manifest_path.exists():
            dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
            item_count = len(dataset_manifest.get("items", []))
        return {
            "project": spec.to_dict(),
            "recommended_training_profile": self.recommend_training_profile(spec),
            "preview": preview,
            "dataset": {
                "ready": dataset_manifest_path.exists(),
                "manifest_path": str(dataset_manifest_path),
                "item_count": item_count,
            },
            "package": {
                "ready": package_manifest is not None,
                "manifest": package_manifest,
            },
            "package_preview": {
                "ready": package_preview_manifest is not None,
                "manifest": package_preview_manifest,
            },
            "sbv2_package": {
                "ready": sbv2_package_manifest is not None,
                "manifest": sbv2_package_manifest,
            },
            "sbv2_package_preview": {
                "ready": sbv2_package_preview_manifest is not None,
                "manifest": sbv2_package_preview_manifest,
            },
            "miotts_package": {
                "ready": miotts_package_manifest is not None,
                "manifest": miotts_package_manifest,
            },
            "miotts_package_preview": {
                "ready": miotts_package_preview_manifest is not None,
                "manifest": miotts_package_preview_manifest,
            },
            "jobs": self.list_jobs(project_id=project_id)[:10],
        }

    def _require_python_module(self, module_name: str, *, install_hint: str) -> None:
        if importlib.util.find_spec(module_name) is None:
            raise RuntimeError(install_hint)

    def _require_piper_training_runtime(self) -> None:
        custom_python = os.environ.get("VOICE_FACTORY_PIPER_PYTHON")
        if custom_python:
            if not Path(custom_python).expanduser().exists():
                raise RuntimeError(
                    f"VOICE_FACTORY_PIPER_PYTHON was set but not found: {custom_python}"
                )
            return
        self._require_python_module(
            "piper_train",
            install_hint="Piper training is not installed. Run `pip install -e .[train]` in this project first.",
        )

    def _require_sbv2_runtime(self) -> None:
        self._require_python_module(
            "style_bert_vits2",
            install_hint=(
                "Style-Bert-VITS2 runtime is not installed. "
                "Install it with `pip install git+https://github.com/litagin02/Style-Bert-VITS2.git`."
            ),
        )

    def _subprocess_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        env.setdefault("WANDB_MODE", "disabled")
        env.setdefault("WANDB_DISABLED", "true")
        env.setdefault("WANDB_SILENT", "true")
        src_path = str(REPO_ROOT / "src")
        env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else src_path + os.pathsep + env["PYTHONPATH"]
        return env

    def _run_subprocess(self, command: list[str], *, cwd: Path | None = None) -> None:
        env = self._subprocess_env()
        try:
            subprocess.run(command, check=True, cwd=str(cwd) if cwd else None, env=env)
        except FileNotFoundError:
            if command and command[0] == "piper-train":
                fallback = [sys.executable, "-m", "piper_train", *command[1:]]
                subprocess.run(fallback, check=True, cwd=str(cwd) if cwd else None, env=env)
                return
            raise

    def _run_subprocess_json(self, command: list[str], *, cwd: Path | None = None) -> dict[str, Any]:
        completed = subprocess.run(
            command,
            check=True,
            cwd=str(cwd) if cwd else None,
            env=self._subprocess_env(),
            capture_output=True,
            text=True,
        )
        stdout = completed.stdout.strip()
        if not stdout:
            raise RuntimeError(f"Subprocess produced no JSON output: {' '.join(command)}")
        try:
            return json.loads(stdout)
        except json.JSONDecodeError:
            for line in reversed(stdout.splitlines()):
                line = line.strip()
                if not line:
                    continue
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue
            raise

    def _piper_python_executable(self) -> str:
        return os.environ.get("VOICE_FACTORY_PIPER_PYTHON", sys.executable)

    def _package_runtime_python(self, family: str) -> str:
        normalized = family.strip().lower()
        env_name = {
            "piper": "VOICE_FACTORY_PIPER_RUNTIME_PYTHON",
            "sbv2": "VOICE_FACTORY_SBV2_RUNTIME_PYTHON",
            "miotts": "VOICE_FACTORY_MIOTTS_RUNTIME_PYTHON",
        }.get(normalized)
        if env_name:
            runtime = os.environ.get(env_name)
            if runtime:
                return runtime
        return sys.executable

    def run_one_click_pipeline(
        self,
        payload: dict[str, Any],
        *,
        project_id: str,
        mio_base_url: str,
        use_preview_reference: bool = True,
        skip_preview: bool = False,
        model_family: str | None = None,
        compute_target: str = "auto",
    ) -> dict[str, Any]:
        spec = self.load_project(project_id)
        resolved_model_family = self._normalize_model_family(model_family or spec.target_model_family)
        if spec.target_model_family != resolved_model_family:
            spec.target_model_family = resolved_model_family
            self._save_project(spec)

        if skip_preview:
            preview_result = self.get_preview_manifest(project_id)
            if preview_result is None:
                raise FileNotFoundError("Preview is not ready. Generate and confirm the seed voice first.")
            self._set_job_stage(payload, stage="preview", stage_label="確認済みの種音声を使います")
        else:
            self._set_job_stage(payload, stage="preview", stage_label="種音声を作成中です")
            preview_result = self.generate_preview(
                project_id,
                device=self._preview_device_for_compute_target(compute_target),
                status_callback=lambda message: self._set_job_stage(
                    payload,
                    stage="preview",
                    stage_label=message,
                ),
            )

        self._set_job_stage(payload, stage="dataset", stage_label="学習データセットを作成中です")
        dataset_result = self.build_dataset(
            project_id,
            mio_base_url=mio_base_url,
            use_preview_reference=use_preview_reference,
            progress_callback=lambda current, total, detail: self._set_job_progress(
                payload,
                current=current,
                total=total,
                detail=detail,
            ),
        )
        scripts_result = self.write_training_scripts(project_id)
        prepare_result = None
        training_result = None
        export_result = None
        package_result = None
        sbv2_training_result = None
        sbv2_package_result = None

        if resolved_model_family == "piper":
            self._require_piper_training_runtime()
            self._set_job_stage(payload, stage="preprocess", stage_label="Piper 学習の前処理をしています")
            prepare_result = self.prepare_piper(project_id)

            self._set_job_stage(payload, stage="training", stage_label="Piper モデルを学習中です")
            training_result = self.train_piper(project_id, execute=True)

            self._set_job_stage(payload, stage="export", stage_label="Piper モデルを配布形式に変換しています")
            export_result = self.export_latest_piper_onnx(project_id)

            self._set_job_stage(payload, stage="package", stage_label="Piper Python パッケージをまとめています")
            package_result = self.build_installable_package(
                project_id,
                onnx_path=Path(export_result["onnx_path"]),
                config_path=Path(export_result["config_path"]),
            )
        else:
            style_bert_vits2_root = str(self._resolve_style_bert_vits2_root())
            self._set_job_stage(payload, stage="training", stage_label="Style-Bert-VITS2 モデルを学習中です")
            sbv2_training_result = self.train_sbv2(
                project_id,
                execute=True,
                style_bert_vits2_root=style_bert_vits2_root,
            )

            self._set_job_stage(payload, stage="package", stage_label="Style-Bert-VITS2 Python パッケージをまとめています")
            sbv2_package_result = self.build_installable_sbv2_package(
                project_id,
                style_bert_vits2_root=style_bert_vits2_root,
            )

        return {
            "project_id": project_id,
            "model_family": resolved_model_family,
            "preview": preview_result,
            "dataset": dataset_result,
            "training_scripts": scripts_result,
            "prepare_piper": prepare_result,
            "training": training_result,
            "export": export_result,
            "package": package_result,
            "sbv2_training": sbv2_training_result,
            "sbv2_package": sbv2_package_result,
            "recommended_training_profile": self.recommend_training_profile(self.load_project(project_id)),
        }

    def _resolve_qwen_device(self, requested_device: str) -> str:
        if requested_device and requested_device != "auto":
            return requested_device
        try:
            import torch
        except ImportError:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _is_model_cached(self, model_id: str) -> bool:
        try:
            snapshot_download(repo_id=model_id, local_files_only=True)
            return True
        except Exception:
            return False

    def _is_hf_file_cached(self, repo_id: str, filename: str) -> bool:
        try:
            hf_hub_download(repo_id=repo_id, filename=filename, local_files_only=True)
            return True
        except Exception:
            return False

    def _voice_designer_assets_ready(self, spec: VoiceProjectSpec) -> bool:
        flow_model = VOICE_DESIGNER_CACHE_DIR / "models" / "flow" / "best_model.pt"
        gpt_entry = VOICE_DESIGNER_CACHE_DIR / "GPT-SoVITS" / "GPT_SoVITS" / "TTS_infer_pack" / "TTS.py"
        pretrained = VOICE_DESIGNER_CACHE_DIR / "GPT-SoVITS" / "GPT_SoVITS" / "pretrained_models" / "s1v3.ckpt"
        sv_checkpoint = (
            VOICE_DESIGNER_CACHE_DIR
            / "GPT-SoVITS"
            / "GPT_SoVITS"
            / "pretrained_models"
            / "sv"
            / "pretrained_eres2netv2w24s4ep4.ckpt"
        )
        gguf_cached = self._is_hf_file_cached(spec.seed_voice_gguf_model, spec.seed_voice_gguf_file)
        return flow_model.exists() and gpt_entry.exists() and pretrained.exists() and sv_checkpoint.exists() and gguf_cached

    def _seed_voice_backend_ready(self, spec: VoiceProjectSpec) -> bool:
        if spec.seed_voice_backend == "kizuna":
            return self._voice_designer_assets_ready(spec)
        return self._is_model_cached(spec.qwen_model_id) and self._is_model_cached(spec.translation_model_id)

    def _ensure_voice_designer_runtime_dirs(self) -> None:
        # fast-langdetect expects its cache parent to exist before it downloads the model.
        (
            VOICE_DESIGNER_CACHE_DIR
            / "GPT-SoVITS"
            / "GPT_SoVITS"
            / "pretrained_models"
            / "fast_langdetect"
        ).mkdir(parents=True, exist_ok=True)

    def _ensure_voice_designer_pretrained_assets(self) -> None:
        try:
            from kizuna_voice_designer.downloader import (
                ensure_flow_model,
                ensure_gpt_sovits,
                ensure_pretrained_models,
            )
        except ImportError:
            return

        ensure_gpt_sovits()
        ensure_flow_model()
        pretrained_root = ensure_pretrained_models()
        required_hubert_files = [
            "chinese-hubert-base/config.json",
            "chinese-hubert-base/preprocessor_config.json",
            "chinese-hubert-base/pytorch_model.bin",
        ]
        for filename in required_hubert_files:
            target = pretrained_root / filename
            if not target.exists():
                hf_hub_download(
                    "lj1995/GPT-SoVITS",
                    filename,
                    local_dir=str(pretrained_root),
                )
        sv_checkpoint = pretrained_root / "sv" / "pretrained_eres2netv2w24s4ep4.ckpt"
        if not sv_checkpoint.exists():
            hf_hub_download(
                "lj1995/GPT-SoVITS",
                "sv/pretrained_eres2netv2w24s4ep4.ckpt",
                local_dir=str(pretrained_root),
            )

    @contextmanager
    def _temporary_cwd(self, target: Path):
        original_cwd = Path.cwd()
        os.chdir(target)
        try:
            yield
        finally:
            os.chdir(original_cwd)

    def generate_preview(
        self,
        project_id: str,
        *,
        device: str = "auto",
        status_callback: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        spec = self.load_project(project_id)
        paths = self.get_project_paths(project_id)
        if status_callback:
            if self._seed_voice_backend_ready(spec):
                if spec.seed_voice_backend == "kizuna":
                    status_callback("音声デザイナーモデルを準備中です")
                else:
                    status_callback("翻訳モデルを準備中です")
            else:
                if spec.seed_voice_backend == "kizuna":
                    status_callback("音声デザイナーモデルをダウンロード中です")
                else:
                    status_callback("翻訳モデルをダウンロード中です")
        resolved_device = self._resolve_qwen_device(device)

        reference_path = paths.preview_dir / "reference.wav"
        manifest_path = paths.preview_dir / "reference.json"
        payload: dict[str, Any]
        if spec.seed_voice_backend == "kizuna":
            try:
                from kizuna_voice_designer import VoiceDesigner
            except ImportError as exc:
                raise RuntimeError(
                    "Preview generation requires kizuna-voice-designer with GGUF support."
                ) from exc

            self._ensure_voice_designer_runtime_dirs()
            self._ensure_voice_designer_pretrained_assets()
            voice_designer = VoiceDesigner(
                device=resolved_device,
                embedding_mode=spec.seed_voice_embedding_mode,
                gguf_model=spec.seed_voice_gguf_model,
                gguf_file=spec.seed_voice_gguf_file,
            )
            if status_callback:
                status_callback("種音声を生成中です")
            with self._temporary_cwd(VOICE_DESIGNER_CACHE_DIR):
                audio, sample_rate, embedding = voice_designer.generate(
                    prompt=spec.style_instruction,
                    text=spec.seed_text,
                    lang="all_ja",
                )
            embedding_path = paths.preview_dir / "reference_embedding.npy"
            voice_designer.save(str(reference_path), audio, sample_rate)
            voice_designer.save_embedding(str(embedding_path), embedding)
            payload = {
                "project_id": project_id,
                "reference_wav": str(reference_path),
                "reference_embedding": str(embedding_path),
                "sample_rate": int(sample_rate),
                "style_instruction": spec.style_instruction,
                "seed_voice_backend": spec.seed_voice_backend,
                "seed_voice_model": spec.seed_voice_model,
                "seed_voice_embedding_mode": spec.seed_voice_embedding_mode,
                "seed_voice_gguf_model": spec.seed_voice_gguf_model,
                "seed_voice_gguf_file": spec.seed_voice_gguf_file,
                "zero_shot_model": spec.mio_model_label,
                "seed_voice_device": resolved_device,
                "seed_text": spec.seed_text,
            }
        else:
            translated_instruction = self._translate_style_instruction(spec)
            try:
                import numpy as np
                import soundfile as sf
                import torch
                from qwen_tts import Qwen3TTSModel
            except ImportError as exc:
                raise RuntimeError("Preview generation requires qwen-tts, torch, numpy, soundfile") from exc

            model_dtype = torch.bfloat16 if resolved_device.startswith("cuda") else torch.float32
            if status_callback:
                if self._is_model_cached(spec.qwen_model_id):
                    status_callback("VoiceDesign モデルを準備中です")
                else:
                    status_callback("VoiceDesign モデルをダウンロード中です")
            model = Qwen3TTSModel.from_pretrained(
                spec.qwen_model_id,
                device_map=resolved_device,
                dtype=model_dtype,
            )
            if status_callback:
                status_callback("種音声を生成中です")
            audio_list, sample_rate = model.generate_voice_design(
                text=spec.seed_text,
                instruct=translated_instruction,
                language=spec.language,
                max_new_tokens=1024,
                temperature=0.85,
                top_p=0.95,
            )
            audio = np.concatenate(audio_list) if isinstance(audio_list, list) else audio_list
            if isinstance(audio, torch.Tensor):
                audio = audio.detach().cpu().numpy()
            audio = np.asarray(audio, dtype=np.float32).reshape(-1)
            sf.write(str(reference_path), audio, sample_rate)
            payload = {
                "project_id": project_id,
                "reference_wav": str(reference_path),
                "sample_rate": int(sample_rate),
                "style_instruction": spec.style_instruction,
                "style_instruction_zh": translated_instruction,
                "translation_model_id": spec.translation_model_id,
                "seed_voice_backend": spec.seed_voice_backend,
                "seed_voice_model": spec.qwen_model_id,
                "zero_shot_model": spec.mio_model_label,
                "seed_voice_device": resolved_device,
                "seed_text": spec.seed_text,
            }
        manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload

    def build_dataset(
        self,
        project_id: str,
        *,
        mio_base_url: str,
        use_preview_reference: bool = True,
        output_format: str = "wav",
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> dict[str, Any]:
        spec = self.load_project(project_id)
        paths = self.get_project_paths(project_id)
        manifest_items: list[dict[str, Any]] = []

        preview_wav = paths.preview_dir / "reference.wav"
        if use_preview_reference and not preview_wav.exists():
            raise FileNotFoundError(
                f"Preview reference not found: {preview_wav}. Run generate-preview first or disable preview reference."
            )
        if not use_preview_reference and not spec.mio_reference_preset_id:
            raise ValueError("mio_reference_preset_id is required when use_preview_reference is false")

        categories = spec.prompt_categories
        category_texts = {
            category: self._expand_category_texts(category, spec.items_per_category)
            for category in categories
        }
        total_items = sum(len(texts) for texts in category_texts.values())
        completed_items = 0
        if progress_callback:
            progress_callback(0, total_items, "データセット生成を開始しました")
        with httpx.Client(timeout=180.0) as client:
            for category in categories:
                texts = category_texts[category]
                category_dir = paths.raw_dataset_dir / category
                if category_dir.exists():
                    shutil.rmtree(category_dir)
                category_dir.mkdir(parents=True, exist_ok=True)

                for index, text in enumerate(texts, start=1):
                    item_id = f"{category[:3]}_{index:03d}"
                    output_wav = category_dir / f"{item_id}.wav"
                    if use_preview_reference:
                        files = {
                            "reference_audio": (
                                preview_wav.name,
                                preview_wav.read_bytes(),
                                "audio/wav",
                            )
                        }
                        data = {
                            "text": text,
                            "model": spec.mio_model_label,
                            "output_format": output_format,
                        }
                    else:
                        files = None
                        data = {
                            "text": text,
                            "reference_preset_id": spec.mio_reference_preset_id,
                            "model": spec.mio_model_label,
                            "output_format": output_format,
                        }

                    response = client.post(
                        f"{mio_base_url.rstrip('/')}/v1/tts/file",
                        data=data,
                        files=files,
                    )
                    response.raise_for_status()
                    output_wav.write_bytes(response.content)
                    manifest_items.append(
                        {
                            "id": item_id,
                            "category": category,
                            "text": text,
                            "wav_path": str(output_wav),
                        }
                    )
                    completed_items += 1
                    if progress_callback:
                        progress_callback(
                            completed_items,
                            total_items,
                            f"{category} / {item_id}",
                        )

        manifest_path = paths.raw_dataset_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps({"project_id": project_id, "items": manifest_items}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._write_piper_format(spec, paths, manifest_items)
        self._write_sbv2_format(spec, paths, manifest_items)
        return {
            "project_id": project_id,
            "item_count": len(manifest_items),
            "manifest_path": str(manifest_path),
            "piper_dataset_dir": str(paths.piper_ljspeech_dir),
            "sbv2_dataset_dir": str(paths.sbv2_data_dir / spec.speaker_name),
            "zero_shot_model": spec.mio_model_label,
        }

    def _expand_category_texts(self, category: str, target_count: int) -> list[str]:
        base_texts = [text.strip() for text in self.prompt_bank[category] if text.strip()]
        if not base_texts:
            raise ValueError(f"No prompts configured for category: {category}")

        suffixes = self._PROMPT_VARIANT_SUFFIXES.get(category, [""])
        expanded: list[str] = []
        seen: set[str] = set()
        variant_round = 0

        while len(expanded) < target_count:
            suffix = suffixes[variant_round % len(suffixes)]
            round_tail = self._PROMPT_VARIANT_ROUND_TAILS[
                min(variant_round // max(1, len(suffixes)), len(self._PROMPT_VARIANT_ROUND_TAILS) - 1)
            ]
            for text in base_texts:
                candidate = self._render_prompt_variant(text, suffix, round_tail)
                if candidate in seen:
                    continue
                seen.add(candidate)
                expanded.append(candidate)
                if len(expanded) >= target_count:
                    break
            variant_round += 1
        return expanded

    def _render_prompt_variant(self, text: str, suffix: str, round_tail: str = "") -> str:
        parts = [text.strip()]
        if suffix:
            parts.append(suffix.strip())
        if round_tail:
            parts.append(round_tail.strip())
        return " ".join(parts)

    def _write_piper_format(
        self, spec: VoiceProjectSpec, paths: ProjectPaths, manifest_items: list[dict[str, Any]]
    ) -> None:
        wav_dir = paths.piper_ljspeech_dir / "wav"
        if wav_dir.exists():
            shutil.rmtree(wav_dir)
        wav_dir.mkdir(parents=True, exist_ok=True)
        metadata_lines: list[str] = []
        for item in manifest_items:
            source_wav = Path(item["wav_path"])
            target_wav = wav_dir / f"{item['id']}.wav"
            self._resample_wav(source_wav, target_wav, target_sr=22050)
            metadata_lines.append(f"{item['id']}|{item['text']}")
        (paths.piper_ljspeech_dir / "metadata.csv").write_text(
            "\n".join(metadata_lines) + "\n", encoding="utf-8"
        )

    def _write_sbv2_format(
        self, spec: VoiceProjectSpec, paths: ProjectPaths, manifest_items: list[dict[str, Any]]
    ) -> None:
        raw_dir = paths.sbv2_data_dir / spec.speaker_name / "raw"
        if raw_dir.exists():
            shutil.rmtree(raw_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)
        esd_lines: list[str] = []
        for item in manifest_items:
            source_wav = Path(item["wav_path"])
            target_wav = raw_dir / f"{item['id']}.wav"
            target_wav.write_bytes(source_wav.read_bytes())
            esd_lines.append(f"{item['id']}.wav|{spec.speaker_name}|JP|{item['text']}")
        (paths.sbv2_data_dir / spec.speaker_name / "esd.list").write_text(
            "\n".join(esd_lines) + "\n",
            encoding="utf-8",
        )

    def _resample_wav(self, source_wav: Path, target_wav: Path, *, target_sr: int) -> None:
        with wave.open(str(source_wav), "rb") as reader:
            channels = reader.getnchannels()
            sample_width = reader.getsampwidth()
            source_sr = reader.getframerate()
            frames = reader.readframes(reader.getnframes())

        if channels > 1:
            frames = audioop.tomono(frames, sample_width, 0.5, 0.5)
            channels = 1
        if sample_width != 2:
            frames = audioop.lin2lin(frames, sample_width, 2)
            sample_width = 2
        if source_sr != target_sr:
            frames, _ = audioop.ratecv(frames, sample_width, channels, source_sr, target_sr, None)

        target_wav.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(target_wav), "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(target_sr)
            writer.writeframes(frames)

    def piper_training_commands(self, project_id: str) -> list[list[str]]:
        spec = self.load_project(project_id)
        paths = self.get_project_paths(project_id)
        profile = self.recommend_training_profile(spec)
        resume_checkpoint_path = self._preferred_piper_resume_checkpoint(project_id)
        base_hparams = self._load_piper_base_hparams(project_id)
        compute_target = self._normalize_compute_target(os.environ.get("VOICE_FACTORY_COMPUTE_TARGET"))
        use_gpu = compute_target != "cpu"
        max_epochs = max(1, self._env_int("VOICE_FACTORY_PIPER_MAX_EPOCHS", 300))
        checkpoint_epochs = max(
            1,
            min(max_epochs, self._env_int("VOICE_FACTORY_PIPER_CHECKPOINT_EPOCHS", 50)),
        )
        disable_wavlm = self._env_flag(
            "VOICE_FACTORY_PIPER_DISABLE_WAVLM",
            default=not use_gpu,
        )
        fine_tune_base_lr = float(os.environ.get("VOICE_FACTORY_PIPER_BASE_LR", "1e-4"))
        command = [
            self._piper_python_executable(),
            "-m",
            "voice_factory.piper_train_wrapper",
        ]
        command.extend(
            [
            "--dataset-dir",
            str(paths.piper_preprocessed_dir),
            "--quality",
            "medium",
            "--checkpoint-epochs",
            str(checkpoint_epochs),
            "--no-pin-memory",
            "--disable_auto_lr_scaling",
            "--base_lr",
            str(fine_tune_base_lr),
            "--precision",
            str(base_hparams.get("precision", "16-mixed") if use_gpu else "32-true"),
            "--max_epochs",
            str(max_epochs),
            "--default_root_dir",
            str(paths.training_output_dir / "piper"),
            "--resume_from_checkpoint",
            str(resume_checkpoint_path),
            "--batch-size",
            str(profile["piper_batch_size"]),
            "--validation-split",
            str(base_hparams.get("validation_split", 0.1)),
            "--num-test-examples",
            str(base_hparams.get("num_test_examples", 5)),
            "--hidden-channels",
            str(base_hparams.get("hidden_channels", 192)),
            "--inter-channels",
            str(base_hparams.get("inter_channels", 192)),
            "--filter-channels",
            str(base_hparams.get("filter_channels", 768)),
            "--n-layers",
            str(base_hparams.get("n_layers", 6)),
            "--n-heads",
            str(base_hparams.get("n_heads", 2)),
            "--gin-channels",
            str(base_hparams.get("gin_channels", 0)),
            "--prosody-dim",
            str(base_hparams.get("prosody_dim", 16)),
            "--num-workers",
            str(base_hparams.get("num_workers", 0)),
            "--seed",
            str(base_hparams.get("seed", 1234)),
            ]
        )
        if use_gpu:
            command.extend(["--accelerator", "gpu", "--devices", "1"])
        else:
            command.extend(["--accelerator", "cpu"])
        samples_per_speaker = int(base_hparams.get("samples_per_speaker", 0) or 0)
        if int(base_hparams.get("num_speakers", 0) or 0) > 1 and samples_per_speaker > 0:
            command.extend(["--samples-per-speaker", str(samples_per_speaker)])
        if not disable_wavlm:
            command.extend(
                [
                    "--wavlm-model-name",
                    str(base_hparams.get("wavlm_model_name", "microsoft/wavlm-base-plus")),
                    "--c-wavlm",
                    str(base_hparams.get("c_wavlm", 0.5)),
                ]
            )
        else:
            command.append("--disable-wavlm")
        return [command]

    def sbv2_training_commands(self, project_id: str) -> list[list[str]]:
        spec = self.load_project(project_id)
        profile = self.recommend_training_profile(spec)
        paths = self.get_project_paths(project_id)
        self._write_sbv2_auto_config(spec, paths)
        preprocess_epochs = max(1, self._env_int("VOICE_FACTORY_SBV2_PREPROCESS_EPOCHS", 100))
        return [
            [
                sys.executable,
                "-m",
                "voice_factory.sbv2_preprocess_wrapper",
                "-m",
                spec.speaker_name,
                "--use_jp_extra",
                "-b",
                str(profile["sbv2_preprocess_batch_size"]),
                "-e",
                str(preprocess_epochs),
            ],
            [
                sys.executable,
                "train_ms_jp_extra.py",
                "-m",
                str(Path("Data") / spec.speaker_name),
                "-c",
                str(paths.sbv2_dir / "config.auto.json"),
            ],
        ]

    def _write_sbv2_auto_config(self, spec: VoiceProjectSpec, paths: ProjectPaths) -> Path:
        profile = self.recommend_training_profile(spec)
        dataset_dir = paths.sbv2_data_dir / spec.speaker_name
        sbv2_epochs = max(1, self._env_int("VOICE_FACTORY_SBV2_EPOCHS", 1000))
        config = {
            "model_name": spec.speaker_name,
            "train": {
                "log_interval": 200,
                "eval_interval": 1000,
                "seed": 42,
                "epochs": sbv2_epochs,
                "learning_rate": 0.0001,
                "betas": [0.8, 0.99],
                "eps": 1e-9,
                "batch_size": profile["sbv2_batch_size"],
                "bf16_run": False,
                "fp16_run": False,
                "lr_decay": 0.99996,
                "segment_size": 16384,
                "init_lr_ratio": 1,
                "warmup_epochs": 0,
                "c_mel": 45,
                "c_kl": 1.0,
                "c_commit": 100,
                "skip_optimizer": False,
                "freeze_ZH_bert": False,
                "freeze_JP_bert": False,
                "freeze_EN_bert": False,
                "freeze_emo": False,
                "freeze_style": False,
                "freeze_decoder": False,
            },
            "data": {
                "use_jp_extra": True,
                "training_files": str(dataset_dir / "train.list"),
                "validation_files": str(dataset_dir / "val.list"),
                "max_wav_value": 32768.0,
                "sampling_rate": 44100,
                "filter_length": 2048,
                "hop_length": 512,
                "win_length": 2048,
                "n_mel_channels": 128,
                "mel_fmin": 0.0,
                "mel_fmax": None,
                "add_blank": True,
                "n_speakers": 1,
                "cleaned_text": True,
                "spk2id": {
                    spec.speaker_name: 0,
                },
                "num_styles": 1,
                "style2id": {
                    "Neutral": 0,
                },
            },
            "model": {
                "use_spk_conditioned_encoder": True,
                "use_noise_scaled_mas": True,
                "use_mel_posterior_encoder": False,
                "use_duration_discriminator": False,
                "use_wavlm_discriminator": True,
                "inter_channels": 192,
                "hidden_channels": 192,
                "filter_channels": 768,
                "n_heads": 2,
                "n_layers": 6,
                "kernel_size": 3,
                "p_dropout": 0.1,
                "resblock": "1",
                "resblock_kernel_sizes": [3, 7, 11],
                "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                "upsample_rates": [8, 8, 2, 2, 2],
                "upsample_initial_channel": 512,
                "upsample_kernel_sizes": [16, 16, 8, 2, 2],
                "n_layers_q": 3,
                "use_spectral_norm": False,
                "gin_channels": 512,
                "slm": {
                    "model": "./slm/wavlm-base-plus",
                    "sr": 16000,
                    "hidden": 768,
                    "nlayers": 13,
                    "initial_channel": 64,
                },
            },
            "version": "2.7.0-JP-Extra",
        }
        config_path = paths.sbv2_dir / "config.auto.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        return config_path

    def _materialize_sbv2_dataset(self, spec: VoiceProjectSpec, style_bert_vits2_root: str) -> Path:
        root = Path(style_bert_vits2_root).expanduser().resolve()
        source = self.get_project_paths(spec.project_id).sbv2_data_dir / spec.speaker_name
        target = root / "Data" / spec.speaker_name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            if target.is_symlink() and target.resolve() == source.resolve():
                return target
            raise RuntimeError(f"SBV2 dataset path already exists and is not the expected symlink: {target}")
        os.symlink(source, target, target_is_directory=True)
        return target

    def _ensure_piper_base_checkpoint(self, project_id: str) -> Path:
        spec = self.load_project(project_id)
        paths = self.get_project_paths(project_id)
        target_dir = paths.artifacts_dir / "pretrained" / "piper-base"
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / spec.piper_base_checkpoint_filename
        if target_path.exists():
            return target_path
        downloaded = hf_hub_download(
            repo_id=spec.piper_base_checkpoint_repo,
            filename=spec.piper_base_checkpoint_filename,
            local_dir=str(target_dir),
        )
        return Path(downloaded)

    def _load_piper_base_hparams(self, project_id: str) -> dict[str, Any]:
        checkpoint_path = self._ensure_piper_base_checkpoint(project_id)
        if os.name == "nt":
            original_posix_path = pathlib.PosixPath
            pathlib.PosixPath = pathlib.WindowsPath
            try:
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            finally:
                pathlib.PosixPath = original_posix_path
        else:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        hparams = dict(checkpoint.get("hyper_parameters", {}))
        state_dict = checkpoint.get("state_dict", {})

        has_gin_conditioning = any(
            key.startswith("model_g.dec.cond.")
            or ".enc.cond_layer." in key
            or key.startswith("model_g.dp.cond.")
            for key in state_dict
        )
        if not has_gin_conditioning:
            hparams["gin_channels"] = 0

        hparams["use_wavlm_discriminator"] = any(
            key.startswith("model_d_wavlm.") for key in state_dict
        )
        return hparams

    def _ensure_sbv2_pretrained_assets(self, project_id: str, style_bert_vits2_root: str) -> dict[str, str]:
        spec = self.load_project(project_id)
        root = Path(style_bert_vits2_root).expanduser().resolve()
        use_jp_extra = spec.sbv2_pretrained_variant == "jp_extra"
        pretrained_dir = root / ("pretrained_jp_extra" if use_jp_extra else "pretrained")
        required_files = (
            ("G_0.safetensors", "D_0.safetensors", "WD_0.safetensors")
            if use_jp_extra
            else ("G_0.safetensors", "D_0.safetensors", "DUR_0.safetensors")
        )
        missing = [name for name in required_files if not (pretrained_dir / name).exists()]
        if missing:
            init_args = [sys.executable, "initialize.py", "--skip_default_models"]
            subprocess.run(init_args, check=True, cwd=str(root))
            missing = [name for name in required_files if not (pretrained_dir / name).exists()]
            if missing:
                missing_names = ", ".join(missing)
                raise FileNotFoundError(
                    f"SBV2 pretrained assets are missing in {pretrained_dir}: {missing_names}"
                )
        model_dir = root / "model_assets" / spec.speaker_name
        model_dir.mkdir(parents=True, exist_ok=True)
        copied: dict[str, str] = {}
        for name in required_files:
            source = pretrained_dir / name
            target = model_dir / name
            if not target.exists():
                shutil.copy2(source, target)
            copied[name] = str(target)
        return copied

    def _resolve_style_bert_vits2_root(self, style_bert_vits2_root: str | None = None) -> Path:
        candidate = (
            style_bert_vits2_root
            or os.environ.get("VOICE_FACTORY_STYLE_BERT_VITS2_ROOT")
            or os.environ.get("STYLE_BERT_VITS2_ROOT")
            or str(REPO_ROOT / "vendor" / "Style-Bert-VITS2")
        )
        root = Path(candidate).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"Style-Bert-VITS2 root was not found: {root}")
        return root

    def _sbv2_model_dir(self, project_id: str, *, style_bert_vits2_root: str | None = None) -> Path:
        spec = self.load_project(project_id)
        root = self._resolve_style_bert_vits2_root(style_bert_vits2_root)
        model_dir = root / "model_assets" / spec.speaker_name
        if not model_dir.exists():
            raise FileNotFoundError(f"SBV2 model directory was not found: {model_dir}")
        return model_dir

    def _latest_sbv2_model_file(
        self,
        project_id: str,
        *,
        style_bert_vits2_root: str | None = None,
    ) -> Path:
        model_dir = self._sbv2_model_dir(project_id, style_bert_vits2_root=style_bert_vits2_root)
        model_files = sorted(
            model_dir.glob("*.safetensors"),
            key=lambda path: path.stat().st_mtime,
        )
        if not model_files:
            raise FileNotFoundError(f"No SBV2 safetensors model was found in {model_dir}")
        return model_files[-1]

    def write_training_scripts(self, project_id: str) -> dict[str, str]:
        spec = self.load_project(project_id)
        paths = self.get_project_paths(project_id)
        piper_script = paths.scripts_dir / "run_train_piper.sh"
        sbv2_script = paths.scripts_dir / "run_train_sbv2.sh"
        profile = self.recommend_training_profile(spec)
        profile_path = paths.scripts_dir / "training_profile.json"
        self._write_sbv2_auto_config(spec, paths)
        profile_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")

        piper_script.write_text(
            "\n".join(
                [
                    "#!/usr/bin/env bash",
                    "set -euo pipefail",
                    'PROJECT_ID="${1:-' + project_id + '}"',
                    f"# Auto-selected Piper batch size from GPU memory: {profile['piper_batch_size']}",
                    "python -m voice_factory.cli prepare-piper --project-id \"$PROJECT_ID\"",
                    "python -m voice_factory.cli train-piper --project-id \"$PROJECT_ID\" --execute",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        sbv2_script.write_text(
            "\n".join(
                [
                    "#!/usr/bin/env bash",
                    "set -euo pipefail",
                    'PROJECT_ID="${1:-' + project_id + '}"',
                    'STYLE_BERT_VITS2_ROOT="${STYLE_BERT_VITS2_ROOT:?set STYLE_BERT_VITS2_ROOT}"',
                    f"# Auto-selected SBV2 batch size from GPU memory: {profile['sbv2_batch_size']}",
                    "python -m voice_factory.cli train-sbv2 --project-id \"$PROJECT_ID\" --execute --style-bert-vits2-root \"$STYLE_BERT_VITS2_ROOT\"",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        if os.name != "nt":
            subprocess.run(["chmod", "+x", str(piper_script), str(sbv2_script)], check=False)
        return {
            "piper_script": str(piper_script),
            "sbv2_script": str(sbv2_script),
            "training_profile": str(profile_path),
            "recommended_training_profile": profile,
        }

    def prepare_piper(self, project_id: str) -> dict[str, str]:
        self._require_piper_training_runtime()
        paths = self.get_project_paths(project_id)
        base_hparams_path = paths.piper_dir / "base_hparams.json"
        base_hparams_path.parent.mkdir(parents=True, exist_ok=True)
        base_hparams_path.write_text(
            json.dumps(
                self._load_piper_base_hparams(project_id),
                ensure_ascii=True,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        return self._run_subprocess_json(
            [
                self._piper_python_executable(),
                str(PACKAGE_DIR / "piper_prepare.py"),
                "--project-dir",
                str(paths.project_dir),
                "--base-hparams-json",
                str(base_hparams_path),
            ],
            cwd=REPO_ROOT,
        )

    def _piper_preprocessed_ready(self, project_id: str) -> bool:
        paths = self.get_project_paths(project_id)
        return (paths.piper_preprocessed_dir / "config.json").exists() and (
            paths.piper_preprocessed_dir / "dataset.jsonl"
        ).exists()

    def train_piper(self, project_id: str, *, execute: bool = False) -> dict[str, Any]:
        commands = self.piper_training_commands(project_id)
        if execute:
            self._require_piper_training_runtime()
            if not self._piper_preprocessed_ready(project_id):
                self.prepare_piper(project_id)
            self._run_subprocess(commands[0])
        return {"commands": commands, "executed": execute}

    def train_sbv2(
        self,
        project_id: str,
        *,
        execute: bool = False,
        style_bert_vits2_root: str | None = None,
    ) -> dict[str, Any]:
        commands = self.sbv2_training_commands(project_id)
        if execute:
            resolved_root = self._resolve_style_bert_vits2_root(style_bert_vits2_root)
            spec = self.load_project(project_id)
            self._materialize_sbv2_dataset(spec, str(resolved_root))
            pretrained_assets = self._ensure_sbv2_pretrained_assets(project_id, str(resolved_root))
            self._write_sbv2_auto_config(spec, self.get_project_paths(project_id))
            for command in commands:
                subprocess.run(command, check=True, cwd=str(resolved_root))
        else:
            pretrained_assets = None
        return {"commands": commands, "executed": execute, "pretrained_assets": pretrained_assets}

    def export_piper_module(
        self,
        *,
        onnx_path: Path,
        config_path: Path,
        output_path: Path,
        module_name: str = "portable_voice",
        class_name: str = "PortablePiperVoice",
    ) -> dict[str, str]:
        written = export_standalone_piper_module(
            onnx_path=onnx_path,
            config_path=config_path,
            output_path=output_path,
            module_name=module_name,
            class_name=class_name,
        )
        return {"output_path": str(written)}

    def _latest_piper_checkpoint(self, project_id: str) -> Path:
        checkpoints = sorted(
            (self.get_project_paths(project_id).training_output_dir / "piper").glob("lightning_logs/version_*/checkpoints/*.ckpt"),
            key=lambda path: path.stat().st_mtime,
        )
        if not checkpoints:
            raise FileNotFoundError("No Piper checkpoints were found. Run training first.")
        return checkpoints[-1]

    def _preferred_piper_resume_checkpoint(self, project_id: str) -> Path:
        try:
            latest_checkpoint = self._latest_piper_checkpoint(project_id)
        except FileNotFoundError:
            latest_checkpoint = None
        if latest_checkpoint is not None:
            return latest_checkpoint
        return self._ensure_piper_base_checkpoint(project_id)

    def export_latest_piper_onnx(self, project_id: str) -> dict[str, str]:
        self._require_piper_training_runtime()
        paths = self.get_project_paths(project_id)
        checkpoint_path = self._latest_piper_checkpoint(project_id)
        config_path = paths.piper_preprocessed_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Piper config was not found: {config_path}")
        onnx_path = paths.output_dir / f"{project_id}.onnx"
        self._run_subprocess(
            [
                self._piper_python_executable(),
                "-m",
                "voice_factory.piper_export_onnx_wrapper",
                "--simplify",
                "--stochastic",
                str(checkpoint_path),
                str(onnx_path),
            ]
        )
        return {
            "checkpoint_path": str(checkpoint_path),
            "config_path": str(config_path),
            "onnx_path": str(onnx_path),
        }

    def build_installable_package(
        self,
        project_id: str,
        *,
        onnx_path: Path | None = None,
        config_path: Path | None = None,
    ) -> dict[str, str]:
        paths = self.get_project_paths(project_id)
        spec = self.load_project(project_id)
        if onnx_path is None or config_path is None:
            export_result = self.export_latest_piper_onnx(project_id)
            onnx_path = Path(export_result["onnx_path"])
            config_path = Path(export_result["config_path"])
        else:
            export_result = {
                "onnx_path": str(onnx_path),
                "config_path": str(config_path),
            }

        package_name = self._PIPER_PACKAGE_NAME
        module_name = self._PIPER_MODULE_NAME
        package_dir = paths.distribution_dir / package_name
        if package_dir.exists():
            shutil.rmtree(package_dir)

        src_dir = package_dir / "src" / module_name
        src_dir.mkdir(parents=True, exist_ok=True)
        module_path = src_dir / "voice.py"
        export_standalone_piper_module(
            onnx_path=onnx_path,
            config_path=config_path,
            output_path=module_path,
            module_name=module_name,
            class_name="PortableVoice",
        )
        (src_dir / "__init__.py").write_text(
            "\n".join(
                [
                    '"""Portable generated voice package."""',
                    "",
                    "from .voice import PortableVoice, load_voice",
                    "",
                    '__all__ = ["PortableVoice", "load_voice"]',
                    "",
                ]
            ),
            encoding="utf-8",
        )
        (package_dir / "README.md").write_text(
            "\n".join(
                [
                    f"# {package_name}",
                    "",
                    "Generated by Voice Factory.",
                    "",
                    "## Install",
                    "",
                    "```bash",
                    f"pip install {package_name}.zip",
                    "```",
                    "",
                    "## Usage",
                    "",
                    "```python",
                    f"from {module_name} import load_voice",
                    "",
                    "voice = load_voice()",
                    'voice.save_wav("こんにちは。よろしくお願いします。", "sample.wav")',
                    "```",
                    "",
                    f"- Source prompt: {spec.style_instruction}",
                    f"- Seed voice backend: {spec.seed_voice_backend}",
                    f"- Seed voice model: {spec.seed_voice_model if spec.seed_voice_backend == 'kizuna' else spec.qwen_model_id}",
                    f"- MioTTS zero-shot model: {spec.mio_model_label}",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        (package_dir / "pyproject.toml").write_text(
            "\n".join(
                [
                    "[build-system]",
                    'requires = ["setuptools>=68"]',
                    'build-backend = "setuptools.build_meta"',
                    "",
                    "[project]",
                    f'name = "{package_name}"',
                    'version = "0.1.0"',
                    'description = "Portable generated Japanese voice package."',
                    'readme = "README.md"',
                    'requires-python = ">=3.10"',
                    "dependencies = [",
                    '  "numpy>=1.26.0,<2",',
                    '  "onnxruntime>=1.17.0",',
                    '  "pyopenjtalk>=0.4.1",',
                    '  "piper-train @ https://github.com/ayutaz/piper-plus/archive/refs/heads/dev.zip#subdirectory=src/python",',
                    "]",
                    "",
                    "[tool.setuptools]",
                    'package-dir = {"" = "src"}',
                    "",
                    "[tool.setuptools.packages.find]",
                    'where = ["src"]',
                    "",
                ]
            ),
            encoding="utf-8",
        )
        archive_base = paths.distribution_dir / package_name
        archive_path = Path(
            shutil.make_archive(
                str(archive_base),
                "zip",
                root_dir=paths.distribution_dir,
                base_dir=package_name,
            )
        )
        manifest = {
            "project_id": project_id,
            "package_name": package_name,
            "module_name": module_name,
            "package_dir": str(package_dir),
            "archive_path": str(archive_path),
            "pip_install_example": f"pip install {archive_path.name}",
            "onnx_path": export_result["onnx_path"],
            "config_path": export_result["config_path"],
        }
        self._package_manifest_path(project_id).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return manifest

    def build_installable_miotts_package(
        self,
        project_id: str,
        *,
        mio_base_url: str | None = None,
        model_id: str | None = None,
    ) -> dict[str, str]:
        paths = self.get_project_paths(project_id)
        spec = self.load_project(project_id)
        preview_manifest = self.get_preview_manifest(project_id)
        reference_audio_path = self.get_preview_audio_path(project_id)
        if preview_manifest is None or not reference_audio_path.exists():
            raise FileNotFoundError("Preview is not ready. Generate the seed voice first.")

        resolved_mio_base_url = (mio_base_url or self.default_mio_base_url()).rstrip("/")
        resolved_model_id = model_id or spec.mio_model_label
        package_name = self._MIOTTS_PACKAGE_NAME
        module_name = self._MIOTTS_MODULE_NAME
        package_dir = paths.distribution_dir / package_name
        if package_dir.exists():
            shutil.rmtree(package_dir)

        src_dir = package_dir / "src" / module_name
        assets_dir = src_dir / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(reference_audio_path, assets_dir / "reference.wav")

        (src_dir / "__init__.py").write_text(
            "\n".join(
                [
                    '"""Portable MioTTS reference voice package."""',
                    "",
                    "from .voice import PortableMioVoice, load_voice",
                    "",
                    '__all__ = ["PortableMioVoice", "load_voice"]',
                    "",
                ]
            ),
            encoding="utf-8",
        )
        (src_dir / "voice.py").write_text(
            "\n".join(
                [
                    "from __future__ import annotations",
                    "",
                    "import os",
                    "from importlib import resources",
                    "from pathlib import Path",
                    "",
                    "import httpx",
                    "",
                    f'DEFAULT_API_BASE = "{resolved_mio_base_url}"',
                    f'DEFAULT_MODEL_ID = "{resolved_model_id}"',
                    "",
                    "",
                    "def _reference_audio_path() -> Path:",
                    '    resource = resources.files(__package__) / "assets" / "reference.wav"',
                    "    return Path(resource)",
                    "",
                    "",
                    "class PortableMioVoice:",
                    "    def __init__(",
                    "        self,",
                    "        *,",
                    "        api_base_url: str | None = None,",
                    "        model_id: str | None = None,",
                    "        reference_audio_path: str | os.PathLike[str] | None = None,",
                    "        timeout: float = 120.0,",
                    "    ) -> None:",
                    '        env_api_base = os.environ.get("MIOTTS_API_BASE", "").strip()',
                    "        self.api_base_url = (api_base_url or env_api_base or DEFAULT_API_BASE).rstrip('/')",
                    "        self.model_id = model_id or DEFAULT_MODEL_ID",
                    "        self.reference_audio_path = (",
                    "            Path(reference_audio_path).expanduser().resolve()",
                    "            if reference_audio_path is not None",
                    "            else _reference_audio_path()",
                    "        )",
                    "        self.timeout = timeout",
                    "",
                    "    def synthesize(self, text: str) -> bytes:",
                    "        if not text or not text.strip():",
                    '            raise ValueError("text is required")',
                    "        with self.reference_audio_path.open('rb') as reference_audio:",
                    "            files = {",
                    "                'reference_audio': (self.reference_audio_path.name, reference_audio, 'audio/wav')",
                    "            }",
                    "            data = {",
                    "                'text': text,",
                    "                'model': self.model_id,",
                    "                'output_format': 'wav',",
                    "            }",
                    "            with httpx.Client(timeout=self.timeout) as client:",
                    "                response = client.post(",
                    "                    f'{self.api_base_url}/v1/tts/file',",
                    "                    data=data,",
                    "                    files=files,",
                    "                )",
                    "                response.raise_for_status()",
                    "                return response.content",
                    "",
                    "    def synthesize_to_file(self, text: str, output_path: str | os.PathLike[str]) -> str:",
                    "        output = Path(output_path)",
                    "        output.parent.mkdir(parents=True, exist_ok=True)",
                    "        output.write_bytes(self.synthesize(text))",
                    "        return str(output)",
                    "",
                    "    def save_wav(self, text: str, output_path: str | os.PathLike[str]) -> str:",
                    "        return self.synthesize_to_file(text, output_path)",
                    "",
                    "",
                    "def load_voice(",
                    "    *,",
                    "    api_base_url: str | None = None,",
                    "    model_id: str | None = None,",
                    "    reference_audio_path: str | os.PathLike[str] | None = None,",
                    "    timeout: float = 120.0,",
                    ") -> PortableMioVoice:",
                    "    return PortableMioVoice(",
                    "        api_base_url=api_base_url,",
                    "        model_id=model_id,",
                    "        reference_audio_path=reference_audio_path,",
                    "        timeout=timeout,",
                    "    )",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        (package_dir / "README.md").write_text(
            "\n".join(
                [
                    f"# {package_name}",
                    "",
                    "Generated by Kizuna Voice Studio.",
                    "",
                    "この zip には学習済みモデルは含まれません。",
                    "プレビューで確定した `reference.wav` を同封し、MioTTS zero-shot API を使って音声生成します。",
                    "",
                    "## Install",
                    "",
                    "```bash",
                    f"pip install {package_name}.zip",
                    "```",
                    "",
                    "## Usage",
                    "",
                    "```python",
                    f"from {module_name} import load_voice",
                    "",
                    "voice = load_voice()",
                    'voice.save_wav("こんにちは。よろしくお願いします。", "sample.wav")',
                    "```",
                    "",
                    "必要なら API ベース URL を切り替えられます。",
                    "",
                    "```python",
                    f"from {module_name} import load_voice",
                    "",
                    'voice = load_voice(api_base_url=\"https://your-miotts-api.example.com\")',
                    "```",
                    "",
                    f"- Source prompt: {spec.style_instruction}",
                    f"- Seed voice backend: {spec.seed_voice_backend}",
                    f"- Seed voice model: {spec.seed_voice_model if spec.seed_voice_backend == 'kizuna' else spec.qwen_model_id}",
                    f"- MioTTS model: {resolved_model_id}",
                    f"- MioTTS API base: {resolved_mio_base_url}",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        (package_dir / "pyproject.toml").write_text(
            "\n".join(
                [
                    "[build-system]",
                    'requires = ["setuptools>=68"]',
                    'build-backend = "setuptools.build_meta"',
                    "",
                    "[project]",
                    f'name = "{package_name}"',
                    'version = "0.1.0"',
                    'description = "Portable MioTTS reference voice package."',
                    'readme = "README.md"',
                    'requires-python = ">=3.10"',
                    "dependencies = [",
                    '  "httpx>=0.27.0",',
                    "]",
                    "",
                    "[tool.setuptools]",
                    'package-dir = {"" = "src"}',
                    'include-package-data = true',
                    "",
                    "[tool.setuptools.packages.find]",
                    'where = ["src"]',
                    "",
                    "[tool.setuptools.package-data]",
                    f'"{module_name}" = ["assets/*.wav"]',
                    "",
                ]
            ),
            encoding="utf-8",
        )
        archive_base = paths.distribution_dir / package_name
        archive_path = Path(
            shutil.make_archive(
                str(archive_base),
                "zip",
                root_dir=paths.distribution_dir,
                base_dir=package_name,
            )
        )
        manifest = {
            "project_id": project_id,
            "package_name": package_name,
            "module_name": module_name,
            "package_dir": str(package_dir),
            "archive_path": str(archive_path),
            "pip_install_example": f"pip install {archive_path.name}",
            "reference_audio_path": str(reference_audio_path),
            "mio_base_url": resolved_mio_base_url,
            "mio_model_id": resolved_model_id,
        }
        self._miotts_package_manifest_path(project_id).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return manifest

    def build_miotts_package_previews(
        self,
        project_id: str,
        *,
        texts: list[str] | None = None,
    ) -> dict[str, Any]:
        manifest = self.get_miotts_package_manifest(project_id)
        if manifest is None:
            raise FileNotFoundError("MioTTS package is not ready. Build it first.")

        package_dir = Path(manifest["package_dir"])
        module_name = manifest["module_name"]
        src_root = package_dir / "src"
        if not (src_root / module_name / "__init__.py").exists():
            raise FileNotFoundError(f"MioTTS package module not found: {src_root / module_name}")

        sys.path.insert(0, str(src_root))
        try:
            package = __import__(module_name, fromlist=["load_voice"])
            voice = package.load_voice(
                api_base_url=manifest["mio_base_url"],
                model_id=manifest["mio_model_id"],
            )
        finally:
            if sys.path and sys.path[0] == str(src_root):
                sys.path.pop(0)
        preview_dir = Path(manifest["package_dir"]) / "previews"
        preview_dir.mkdir(parents=True, exist_ok=True)

        text_items = texts or [item[1] for item in self._MIOTTS_PREVIEW_TEXTS]
        default_ids = [item[0] for item in self._MIOTTS_PREVIEW_TEXTS]
        samples: list[dict[str, str]] = []
        for index, text in enumerate(text_items):
            sample_id = default_ids[index] if index < len(default_ids) else f"sample_{index + 1:02d}"
            output_path = preview_dir / f"{sample_id}.wav"
            voice.save_wav(text, output_path)
            samples.append(
                {
                    "id": sample_id,
                    "text": text,
                    "audio_path": str(output_path),
                }
            )

        preview_manifest = {
            "project_id": project_id,
            "package_name": manifest["package_name"],
            "module_name": module_name,
            "samples": samples,
        }
        self._miotts_package_preview_manifest_path(project_id).write_text(
            json.dumps(preview_manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return preview_manifest

    def build_generated_package_previews(
        self,
        project_id: str,
        *,
        family: str,
        texts: list[str] | None = None,
        compute_target: str = "auto",
    ) -> dict[str, Any]:
        normalized_family = self._normalize_model_family(family)
        manifest = (
            self.get_package_manifest(project_id)
            if normalized_family == "piper"
            else self.get_sbv2_package_manifest(project_id)
        )
        if manifest is None:
            raise FileNotFoundError(f"{normalized_family} package is not ready. Build it first.")

        package_dir = Path(manifest["package_dir"])
        module_name = manifest["module_name"]
        src_root = package_dir / "src"
        if not (src_root / module_name / "__init__.py").exists():
            raise FileNotFoundError(f"Package module not found: {src_root / module_name}")

        text_items = texts or [item[1] for item in self._PACKAGE_PREVIEW_TEXTS]
        preview_dir = package_dir / "previews"
        preview_dir.mkdir(parents=True, exist_ok=True)

        command = [
            self._package_runtime_python(normalized_family),
            "-m",
            "voice_factory.package_runtime_runner",
            "--family",
            normalized_family,
            "--package-dir",
            str(package_dir),
            "--module-name",
            module_name,
            "--output-dir",
            str(preview_dir),
            "--texts-json",
            json.dumps(text_items, ensure_ascii=False),
        ]
        if normalized_family == "sbv2":
            # Packaged SBV2 preview is more stable on CPU across managed runtimes.
            device = "cpu"
            command.extend(["--device", device])

        env = self._subprocess_env()
        self._apply_compute_target_to_env(env, compute_target)
        preview_manifest = self._run_subprocess_json(command)
        preview_manifest["project_id"] = project_id
        preview_manifest["package_name"] = manifest["package_name"]
        preview_manifest["module_name"] = module_name

        manifest_path = (
            self._package_preview_manifest_path(project_id)
            if normalized_family == "piper"
            else self._sbv2_package_preview_manifest_path(project_id)
        )
        manifest_path.write_text(
            json.dumps(preview_manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return preview_manifest

    def build_installable_sbv2_package(
        self,
        project_id: str,
        *,
        style_bert_vits2_root: str | None = None,
    ) -> dict[str, str]:
        self._require_sbv2_runtime()
        paths = self.get_project_paths(project_id)
        spec = self.load_project(project_id)
        model_dir = self._sbv2_model_dir(project_id, style_bert_vits2_root=style_bert_vits2_root)
        model_path = self._latest_sbv2_model_file(project_id, style_bert_vits2_root=style_bert_vits2_root)
        config_path = model_dir / "config.json"
        style_vec_path = model_dir / "style_vectors.npy"
        for required_path in (config_path, style_vec_path):
            if not required_path.exists():
                raise FileNotFoundError(f"Required SBV2 asset was not found: {required_path}")

        package_name = self._SBV2_PACKAGE_NAME
        module_name = self._SBV2_MODULE_NAME
        package_dir = paths.distribution_dir / package_name
        if package_dir.exists():
            shutil.rmtree(package_dir)

        src_dir = package_dir / "src" / module_name
        assets_dir = src_dir / "assets" / "model"
        assets_dir.mkdir(parents=True, exist_ok=True)
        copied_model_path = assets_dir / model_path.name
        shutil.copy2(model_path, copied_model_path)
        shutil.copy2(config_path, assets_dir / "config.json")
        shutil.copy2(style_vec_path, assets_dir / "style_vectors.npy")

        (src_dir / "__init__.py").write_text(
            "\n".join(
                [
                    '"""Installable Style-Bert-VITS2 voice package."""',
                    "",
                    "from .voice import PortableSBV2Voice, load_voice",
                    "",
                    '__all__ = ["PortableSBV2Voice", "load_voice"]',
                    "",
                ]
            ),
            encoding="utf-8",
        )
        (src_dir / "voice.py").write_text(
            "\n".join(
                [
                    '"""Portable Style-Bert-VITS2 voice wrapper."""',
                    "",
                    "from __future__ import annotations",
                    "",
                    "import wave",
                    "from importlib import resources",
                    "from pathlib import Path",
                    "",
                    "import pyopenjtalk",
                    "",
                    "if not hasattr(pyopenjtalk, 'unset_user_dict'):",
                    "    def _unset_user_dict() -> None:",
                    "        return None",
                    "",
                    "    pyopenjtalk.unset_user_dict = _unset_user_dict",
                    "",
                    "from style_bert_vits2.constants import Languages",
                    "from style_bert_vits2.tts_model import TTSModel",
                    "",
                    "_MODEL_DIR = resources.files(__package__) / 'assets' / 'model'",
                    f"_MODEL_FILE = '{model_path.name}'",
                    "",
                    "",
                    "class PortableSBV2Voice:",
                    "    def __init__(self, device: str = 'cpu') -> None:",
                    "        self.device = device",
                    "        self.model_dir = Path(str(_MODEL_DIR))",
                    "        self.model = TTSModel(",
                    "            model_path=self.model_dir / _MODEL_FILE,",
                    "            config_path=self.model_dir / 'config.json',",
                    "            style_vec_path=self.model_dir / 'style_vectors.npy',",
                    "            device=device,",
                    "        )",
                    "",
                    "    @property",
                    "    def available_styles(self) -> list[str]:",
                    "        return sorted(self.model.style2id.keys(), key=self.model.style2id.get)",
                    "",
                    "    @property",
                    "    def available_speakers(self) -> list[str]:",
                    "        return [self.model.id2spk[idx] for idx in sorted(self.model.id2spk)]",
                    "",
                    "    def synthesize(",
                    "        self,",
                    "        text: str,",
                    "        *,",
                    "        style: str = 'Neutral',",
                    "        style_weight: float = 1.0,",
                    "        speaker_id: int = 0,",
                    "        length: float = 1.0,",
                    "        sdp_ratio: float = 0.2,",
                    "        noise: float = 0.6,",
                    "        noise_w: float = 0.8,",
                    "        language: Languages = Languages.JP,",
                    "    ):",
                    "        return self.model.infer(",
                    "            text=text,",
                    "            language=language,",
                    "            speaker_id=speaker_id,",
                    "            style=style,",
                    "            style_weight=style_weight,",
                    "            length=length,",
                    "            sdp_ratio=sdp_ratio,",
                    "            noise=noise,",
                    "            noise_w=noise_w,",
                    "        )",
                    "",
                    "    def save_wav(self, text: str, output_path: str | Path, **kwargs) -> Path:",
                    "        sample_rate, audio = self.synthesize(text, **kwargs)",
                    "        output_path = Path(output_path)",
                    "        output_path.parent.mkdir(parents=True, exist_ok=True)",
                    "        with wave.open(str(output_path), 'wb') as wf:",
                    "            wf.setnchannels(1)",
                    "            wf.setsampwidth(2)",
                    "            wf.setframerate(sample_rate)",
                    "            wf.writeframes(audio.tobytes())",
                    "        return output_path",
                    "",
                    "",
                    "def load_voice(device: str = 'cpu') -> PortableSBV2Voice:",
                    "    return PortableSBV2Voice(device=device)",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        (package_dir / "README.md").write_text(
            "\n".join(
                [
                    f"# {package_name}",
                    "",
                    "Generated by Voice Factory.",
                    "",
                    "## Install",
                    "",
                    "```bash",
                    f"pip install {package_name}.zip",
                    "```",
                    "",
                    "## Usage",
                    "",
                    "```python",
                    f"from {module_name} import load_voice",
                    "",
                    "voice = load_voice(device='cuda')",
                    'voice.save_wav("こんにちは。よろしくお願いします。", "sample.wav")',
                    "```",
                    "",
                    f"- Source prompt: {spec.style_instruction}",
                    f"- SBV2 speaker name: {spec.speaker_name}",
                    f"- Packed model: {model_path.name}",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        (package_dir / "pyproject.toml").write_text(
            "\n".join(
                [
                    "[build-system]",
                    'requires = ["setuptools>=68"]',
                    'build-backend = "setuptools.build_meta"',
                    "",
                    "[project]",
                    f'name = "{package_name}"',
                    'version = "0.1.0"',
                    'description = "Installable Style-Bert-VITS2 voice package."',
                    'readme = "README.md"',
                    'requires-python = ">=3.10"',
                    "dependencies = [",
                    '  "numpy<2",',
                    '  "style-bert-vits2[torch] @ git+https://github.com/litagin02/Style-Bert-VITS2.git",',
                    "]",
                    "",
                    "[tool.setuptools]",
                    'package-dir = {"" = "src"}',
                    'include-package-data = true',
                    "",
                    "[tool.setuptools.packages.find]",
                    'where = ["src"]',
                    "",
                    "[tool.setuptools.package-data]",
                    f'"{module_name}" = ["assets/model/*"]',
                    "",
                ]
            ),
            encoding="utf-8",
        )
        archive_base = paths.distribution_dir / package_name
        archive_path = Path(
            shutil.make_archive(
                str(archive_base),
                "zip",
                root_dir=paths.distribution_dir,
                base_dir=package_name,
            )
        )
        manifest = {
            "project_id": project_id,
            "package_name": package_name,
            "module_name": module_name,
            "package_dir": str(package_dir),
            "archive_path": str(archive_path),
            "pip_install_example": f"pip install {archive_path.name}",
            "model_dir": str(model_dir),
            "model_path": str(model_path),
            "config_path": str(config_path),
            "style_vectors_path": str(style_vec_path),
        }
        self._sbv2_package_manifest_path(project_id).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return manifest
