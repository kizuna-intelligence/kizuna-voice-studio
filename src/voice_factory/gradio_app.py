from __future__ import annotations

import json

import gradio as gr

from .service import VoiceFactoryService

service = VoiceFactoryService()


def start_preview_ui(
    style_instruction: str,
    gpu_memory_gb: float,
    model_family: str,
    seed_voice_backend: str,
) -> tuple[str, str]:
    payload = service.start_simple_preview_job(
        style_instruction=style_instruction,
        gpu_memory_gb=int(gpu_memory_gb),
        model_family=model_family,
        seed_voice_backend=seed_voice_backend,
    )
    return payload["project"]["project_id"], json.dumps(payload, ensure_ascii=False, indent=2)


def build_tts_ui(project_id: str, mio_base_url: str, model_family: str) -> str:
    payload = service.start_build_tts_job(
        project_id=project_id,
        mio_base_url=mio_base_url,
        model_family=model_family,
    )
    return json.dumps(payload, ensure_ascii=False, indent=2)


def project_status_ui(project_id: str) -> tuple[str | None, str]:
    payload = service.describe_project(project_id)
    preview = payload.get("preview") or {}
    preview_path = None
    if preview.get("reference_wav"):
        preview_path = preview["reference_wav"]
    return preview_path, json.dumps(payload, ensure_ascii=False, indent=2)


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Voice Factory") as demo:
        gr.Markdown("# Voice Factory")
        gr.Markdown("まず種音声を作って確認し、気に入ったらその声で TTS を作ります。")
        style_instruction = gr.Textbox(
            label="どんな声を作りたいですか",
            lines=5,
            value="20代後半の女性。落ち着いていて明るすぎず、ニュースを自然に読める聞き取りやすい声。",
        )
        gpu_memory_gb = gr.Number(label="GPU Memory (GB)", value=16, precision=0)
        model_family = gr.Radio(
            label="作るモデル",
            choices=[("Piper TTS", "piper"), ("Style-Bert-VITS2", "sbv2")],
            value="piper",
        )
        seed_voice_backend = gr.Radio(
            label="種音声の作り方",
            choices=[("Kizuna Voice Designer", "kizuna"), ("Qwen Voice Designer", "qwen")],
            value="kizuna",
        )
        mio_base_url = gr.Textbox(label="Mio Base URL", value=service.default_mio_base_url())
        project_id = gr.Textbox(label="Project ID")
        launch_output = gr.Code(label="Started Job", language="json")
        preview_audio = gr.Audio(label="Voice Preview", type="filepath")
        preview_status = gr.Code(label="Project Status", language="json")

        gr.Button("1. 種音声を作成").click(
            start_preview_ui,
            inputs=[style_instruction, gpu_memory_gb, model_family, seed_voice_backend],
            outputs=[project_id, launch_output],
        )
        gr.Button("現在の状態を見る").click(
            project_status_ui,
            inputs=[project_id],
            outputs=[preview_audio, preview_status],
        )
        gr.Button("2. この声で TTS を作る").click(
            build_tts_ui,
            inputs=[project_id, mio_base_url, model_family],
            outputs=[launch_output],
        )
    return demo


def main() -> None:
    build_demo().launch(server_name="127.0.0.1", server_port=7862)


if __name__ == "__main__":
    main()
