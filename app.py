from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import gradio as gr

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = Path(os.environ.get("SPACE_DATA_DIR", "/data")).expanduser()


def _resolve_data_root() -> Path:
    if DEFAULT_DATA_ROOT.exists() and os.access(DEFAULT_DATA_ROOT, os.W_OK):
        return DEFAULT_DATA_ROOT
    if DEFAULT_DATA_ROOT.parent.exists() and os.access(DEFAULT_DATA_ROOT.parent, os.W_OK):
        return DEFAULT_DATA_ROOT
    fallback = REPO_ROOT / ".space-data"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


DATA_ROOT = _resolve_data_root()

os.environ.setdefault("VOICE_FACTORY_WORKSPACE_ROOT", str(DATA_ROOT / "voice-factory-workspace"))
os.environ.setdefault("HF_HOME", str(DATA_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("XDG_CACHE_HOME", str(DATA_ROOT / ".cache"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(DATA_ROOT / ".cache" / "huggingface" / "transformers"))
os.environ.setdefault("GRADIO_SERVER_NAME", "0.0.0.0")
os.environ.setdefault("GRADIO_SERVER_PORT", "7860")

src_path = REPO_ROOT / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from voice_factory.service import VoiceFactoryService

service = VoiceFactoryService()


def start_preview_ui(style_instruction: str) -> tuple[str, str]:
    payload = service.start_simple_preview_job(
        style_instruction=style_instruction,
        gpu_memory_gb=16,
        model_family="piper",
        seed_voice_backend="kizuna",
    )
    return payload["project"]["project_id"], json.dumps(payload, ensure_ascii=False, indent=2)


def project_status_ui(project_id: str) -> tuple[str | None, str]:
    payload = service.describe_project(project_id)
    preview = payload.get("preview") or {}
    preview_path = preview.get("reference_wav")
    return preview_path, json.dumps(payload, ensure_ascii=False, indent=2)


with gr.Blocks(title="Kizuna Voice Studio Space") as demo:
    gr.Markdown("# Kizuna Voice Studio")
    gr.Markdown(
        "\n".join(
            [
        "This Space focuses on seed voice preview generation with Kizuna Voice Designer.",
        "Full TTS training and package export are intended for the desktop app or a paid GPU runtime.",
            ]
        )
    )
    style_instruction = gr.Textbox(
        label="Describe the voice you want",
        lines=5,
        value="20代後半の女性。落ち着いていて明るすぎず、ニュースを自然に読める聞き取りやすい声。",
    )
    project_id = gr.Textbox(label="Project ID")
    launch_output = gr.Code(label="Started Job", language="json")
    preview_audio = gr.Audio(label="Voice Preview", type="filepath")
    preview_status = gr.Code(label="Project Status", language="json")

    gr.Button("1. Generate Seed Voice Preview").click(
        start_preview_ui,
        inputs=[style_instruction],
        outputs=[project_id, launch_output],
    )
    gr.Button("2. Refresh Preview Status").click(
        project_status_ui,
        inputs=[project_id],
        outputs=[preview_audio, preview_status],
    )


if __name__ == "__main__":
    demo.launch(
        server_name=os.environ["GRADIO_SERVER_NAME"],
        server_port=int(os.environ["GRADIO_SERVER_PORT"]),
    )
