from __future__ import annotations

from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .models import VoiceProjectSpec
from .service import VoiceFactoryService

app = FastAPI(title="Voice Factory API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
service = VoiceFactoryService()


class PlanProjectRequest(BaseModel):
    project: dict


class QuickStartRequest(BaseModel):
    style_instruction: str
    gpu_memory_gb: int = 16
    model_family: str = "piper"
    seed_voice_backend: str = "kizuna"
    compute_target: str = "auto"


class OneClickRequest(BaseModel):
    style_instruction: str
    gpu_memory_gb: int = 16
    mio_base_url: str = service.default_mio_base_url()
    model_family: str = "piper"
    seed_voice_backend: str = "kizuna"
    compute_target: str = "auto"


class BuildDatasetRequest(BaseModel):
    mio_base_url: str
    use_preview_reference: bool = True


class BuildTTSRequest(BaseModel):
    mio_base_url: str = service.default_mio_base_url()
    model_family: str = "piper"
    compute_target: str = "auto"


class StartJobRequest(BaseModel):
    job_type: str
    project_id: str
    params: dict = {}


class ExportPiperRequest(BaseModel):
    onnx_path: str
    config_path: str
    output_path: str
    module_name: str = "portable_voice"
    class_name: str = "PortablePiperVoice"


class BuildSBV2PackageRequest(BaseModel):
    style_bert_vits2_root: str | None = None


class BuildMioTtsPackageRequest(BaseModel):
    mio_base_url: str = service.default_mio_base_url()
    model_id: str | None = None


class BuildMioTtsPackagePreviewRequest(BaseModel):
    texts: list[str] | None = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "projects": service.list_projects()}


@app.get("/v1/system/compute-targets")
def list_compute_targets() -> dict:
    return {"targets": service.list_compute_targets()}


@app.get("/v1/projects")
def list_projects() -> dict:
    return {"projects": service.list_projects()}


@app.get("/v1/projects/{project_id}")
def describe_project(project_id: str) -> dict:
    return service.describe_project(project_id)


@app.get("/v1/jobs")
def list_jobs(project_id: str | None = None) -> dict:
    return {"jobs": service.list_jobs(project_id=project_id)}


@app.get("/v1/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    return service.get_job(job_id)


@app.post("/v1/projects/plan")
def plan_project(payload: PlanProjectRequest) -> dict:
    spec = VoiceProjectSpec.from_dict(payload.project)
    return service.plan_project(spec)


@app.post("/v1/quick-start")
def quick_start(payload: QuickStartRequest) -> dict:
    return service.start_simple_preview_job(
        style_instruction=payload.style_instruction,
        gpu_memory_gb=payload.gpu_memory_gb,
        model_family=payload.model_family,
        seed_voice_backend=payload.seed_voice_backend,
        compute_target=payload.compute_target,
    )


@app.post("/v1/one-click")
def one_click(payload: OneClickRequest) -> dict:
    return service.start_one_click_job(
        style_instruction=payload.style_instruction,
        gpu_memory_gb=payload.gpu_memory_gb,
        mio_base_url=payload.mio_base_url,
        model_family=payload.model_family,
        seed_voice_backend=payload.seed_voice_backend,
        compute_target=payload.compute_target,
    )


@app.post("/v1/projects/{project_id}/preview")
def generate_preview(project_id: str) -> dict:
    return service.generate_preview(project_id)


@app.get("/v1/projects/{project_id}/preview")
def get_preview(project_id: str) -> dict:
    payload = service.get_preview_manifest(project_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Preview not ready")
    return payload


@app.get("/v1/projects/{project_id}/preview/audio")
def get_preview_audio(project_id: str) -> FileResponse:
    preview_path = service.get_preview_audio_path(project_id)
    if not preview_path.exists():
        raise HTTPException(status_code=404, detail="Preview audio not ready")
    return FileResponse(preview_path, media_type="audio/wav", filename=f"{project_id}-preview.wav")


@app.get("/v1/projects/{project_id}/package")
def get_package(project_id: str) -> dict:
    payload = service.get_package_manifest(project_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Package not ready")
    return payload


@app.get("/v1/projects/{project_id}/package/download")
def download_package(project_id: str) -> FileResponse:
    manifest = service.get_package_manifest(project_id)
    if manifest is None:
        raise HTTPException(status_code=404, detail="Package not ready")
    archive_path = Path(manifest["archive_path"])
    if not archive_path.exists():
        raise HTTPException(status_code=404, detail="Package archive not found")
    return FileResponse(archive_path, media_type="application/zip", filename=archive_path.name)


@app.get("/v1/projects/{project_id}/package/sbv2")
def get_sbv2_package(project_id: str) -> dict:
    payload = service.get_sbv2_package_manifest(project_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="SBV2 package not ready")
    return payload


@app.get("/v1/projects/{project_id}/package/sbv2/download")
def download_sbv2_package(project_id: str) -> FileResponse:
    manifest = service.get_sbv2_package_manifest(project_id)
    if manifest is None:
        raise HTTPException(status_code=404, detail="SBV2 package not ready")
    archive_path = Path(manifest["archive_path"])
    if not archive_path.exists():
        raise HTTPException(status_code=404, detail="SBV2 package archive not found")
    return FileResponse(archive_path, media_type="application/zip", filename=archive_path.name)


@app.get("/v1/projects/{project_id}/package/miotts")
def get_miotts_package(project_id: str) -> dict:
    payload = service.get_miotts_package_manifest(project_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="MioTTS package not ready")
    return payload


@app.get("/v1/projects/{project_id}/package/miotts/download")
def download_miotts_package(project_id: str) -> FileResponse:
    manifest = service.get_miotts_package_manifest(project_id)
    if manifest is None:
        raise HTTPException(status_code=404, detail="MioTTS package not ready")
    archive_path = Path(manifest["archive_path"])
    if not archive_path.exists():
        raise HTTPException(status_code=404, detail="MioTTS package archive not found")
    return FileResponse(archive_path, media_type="application/zip", filename=archive_path.name)


@app.get("/v1/projects/{project_id}/package/miotts/preview")
def get_miotts_package_preview(project_id: str) -> dict:
    payload = service.get_miotts_package_preview_manifest(project_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="MioTTS package preview is not ready")
    return payload


@app.get("/v1/projects/{project_id}/package/miotts/preview/{sample_id}/audio")
def get_miotts_package_preview_audio(project_id: str, sample_id: str) -> FileResponse:
    manifest = service.get_miotts_package_preview_manifest(project_id)
    if manifest is None:
        raise HTTPException(status_code=404, detail="MioTTS package preview is not ready")
    sample = next((item for item in manifest["samples"] if item["id"] == sample_id), None)
    if sample is None:
        raise HTTPException(status_code=404, detail="MioTTS preview sample not found")
    audio_path = Path(sample["audio_path"])
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="MioTTS preview audio not found")
    return FileResponse(audio_path, media_type="audio/wav", filename=audio_path.name)


@app.post("/v1/projects/{project_id}/dataset")
def build_dataset(project_id: str, payload: BuildDatasetRequest) -> dict:
    return service.build_dataset(
        project_id,
        mio_base_url=payload.mio_base_url,
        use_preview_reference=payload.use_preview_reference,
    )


@app.post("/v1/projects/{project_id}/approve-preview")
def approve_preview(project_id: str, payload: BuildDatasetRequest) -> dict:
    return service.start_dataset_pipeline_job(
        project_id=project_id,
        mio_base_url=payload.mio_base_url,
        use_preview_reference=payload.use_preview_reference,
    )


@app.post("/v1/projects/{project_id}/build-tts")
def build_tts(project_id: str, payload: BuildTTSRequest) -> dict:
    return service.start_build_tts_job(
        project_id=project_id,
        mio_base_url=payload.mio_base_url,
        model_family=payload.model_family,
        compute_target=payload.compute_target,
    )


@app.post("/v1/projects/{project_id}/training-scripts")
def write_training_scripts(project_id: str) -> dict:
    return service.write_training_scripts(project_id)


@app.post("/v1/jobs")
def start_job(payload: StartJobRequest) -> dict:
    return service.start_job(
        job_type=payload.job_type,
        project_id=payload.project_id,
        params=payload.params,
    )


@app.post("/v1/projects/{project_id}/prepare-piper")
def prepare_piper(project_id: str) -> dict:
    return service.prepare_piper(project_id)


@app.post("/v1/projects/{project_id}/train-piper")
def train_piper(project_id: str) -> dict:
    return service.train_piper(project_id, execute=False)


@app.post("/v1/projects/{project_id}/train-sbv2")
def train_sbv2(project_id: str) -> dict:
    return service.train_sbv2(project_id, execute=False)


@app.post("/v1/projects/{project_id}/package/sbv2")
def build_sbv2_package(project_id: str, payload: BuildSBV2PackageRequest) -> dict:
    return service.build_installable_sbv2_package(
        project_id,
        style_bert_vits2_root=payload.style_bert_vits2_root,
    )


@app.post("/v1/projects/{project_id}/package/miotts")
def build_miotts_package(project_id: str, payload: BuildMioTtsPackageRequest) -> dict:
    return service.build_installable_miotts_package(
        project_id,
        mio_base_url=payload.mio_base_url,
        model_id=payload.model_id,
    )


@app.post("/v1/projects/{project_id}/package/miotts/preview")
def build_miotts_package_preview(project_id: str, payload: BuildMioTtsPackagePreviewRequest) -> dict:
    return service.build_miotts_package_previews(
        project_id,
        texts=payload.texts,
    )


@app.post("/v1/export/piper")
def export_piper(payload: ExportPiperRequest) -> dict:
    return service.export_piper_module(
        onnx_path=Path(payload.onnx_path),
        config_path=Path(payload.config_path),
        output_path=Path(payload.output_path),
        module_name=payload.module_name,
        class_name=payload.class_name,
    )


def main() -> None:
    uvicorn.run(app, host="127.0.0.1", port=7861)


if __name__ == "__main__":
    main()
