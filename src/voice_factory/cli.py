from __future__ import annotations

import argparse
import json
from pathlib import Path

from .models import VoiceProjectSpec
from .service import VoiceFactoryService

DEFAULT_MIO_BASE_URL = VoiceFactoryService().default_mio_base_url()


def _print(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(prog="voice-factory")
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan-project")
    plan_parser.add_argument("--spec", type=Path, required=True)

    quick_parser = subparsers.add_parser("quick-start")
    quick_parser.add_argument("--style-instruction", required=True)
    quick_parser.add_argument("--gpu-memory-gb", type=int, default=16)
    quick_parser.add_argument("--model-family", choices=["piper", "sbv2"], default="piper")
    quick_parser.add_argument("--seed-voice-backend", choices=["kizuna", "qwen"], default="kizuna")

    one_click_parser = subparsers.add_parser("one-click")
    one_click_parser.add_argument("--style-instruction", required=True)
    one_click_parser.add_argument("--gpu-memory-gb", type=int, default=16)
    one_click_parser.add_argument("--mio-base-url", default=DEFAULT_MIO_BASE_URL)
    one_click_parser.add_argument("--model-family", choices=["piper", "sbv2"], default="piper")
    one_click_parser.add_argument("--seed-voice-backend", choices=["kizuna", "qwen"], default="kizuna")

    preview_parser = subparsers.add_parser("generate-preview")
    preview_parser.add_argument("--project-id", required=True)

    approve_parser = subparsers.add_parser("approve-preview")
    approve_parser.add_argument("--project-id", required=True)
    approve_parser.add_argument("--mio-base-url", required=True)
    approve_parser.add_argument(
        "--no-preview-reference",
        action="store_false",
        dest="use_preview_reference",
        default=True,
    )

    dataset_parser = subparsers.add_parser("build-dataset")
    dataset_parser.add_argument("--project-id", required=True)
    dataset_parser.add_argument("--mio-base-url", required=True)
    dataset_parser.add_argument(
        "--no-preview-reference",
        action="store_false",
        dest="use_preview_reference",
        default=True,
    )

    scripts_parser = subparsers.add_parser("write-training-scripts")
    scripts_parser.add_argument("--project-id", required=True)

    prepare_piper_parser = subparsers.add_parser("prepare-piper")
    prepare_piper_parser.add_argument("--project-id", required=True)

    train_piper_parser = subparsers.add_parser("train-piper")
    train_piper_parser.add_argument("--project-id", required=True)
    train_piper_parser.add_argument("--execute", action="store_true")

    train_sbv2_parser = subparsers.add_parser("train-sbv2")
    train_sbv2_parser.add_argument("--project-id", required=True)
    train_sbv2_parser.add_argument("--execute", action="store_true")
    train_sbv2_parser.add_argument("--style-bert-vits2-root")

    export_parser = subparsers.add_parser("export-piper-module")
    export_parser.add_argument("--onnx-path", type=Path, required=True)
    export_parser.add_argument("--config-path", type=Path, required=True)
    export_parser.add_argument("--output-path", type=Path, required=True)
    export_parser.add_argument("--module-name", default="portable_voice")
    export_parser.add_argument("--class-name", default="PortablePiperVoice")

    package_parser = subparsers.add_parser("build-installable-package")
    package_parser.add_argument("--project-id", required=True)

    sbv2_package_parser = subparsers.add_parser("build-installable-sbv2-package")
    sbv2_package_parser.add_argument("--project-id", required=True)
    sbv2_package_parser.add_argument("--style-bert-vits2-root")

    start_job_parser = subparsers.add_parser("start-job")
    start_job_parser.add_argument("--job-type", required=True)
    start_job_parser.add_argument("--project-id", required=True)
    start_job_parser.add_argument("--params-json", default="{}")

    get_job_parser = subparsers.add_parser("get-job")
    get_job_parser.add_argument("--job-id", required=True)

    list_jobs_parser = subparsers.add_parser("list-jobs")
    list_jobs_parser.add_argument("--project-id")

    run_job_parser = subparsers.add_parser("run-job")
    run_job_parser.add_argument("--job-file", type=Path, required=True)

    serve_parser = subparsers.add_parser("serve-api")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=7861)

    gradio_parser = subparsers.add_parser("launch-gradio")
    gradio_parser.add_argument("--host", default="127.0.0.1")
    gradio_parser.add_argument("--port", type=int, default=7862)

    args = parser.parse_args()
    service = VoiceFactoryService()

    if args.command == "plan-project":
        spec = VoiceProjectSpec.from_dict(json.loads(args.spec.read_text(encoding="utf-8")))
        _print(service.plan_project(spec))
        return
    if args.command == "quick-start":
        _print(
            service.start_simple_preview_job(
                style_instruction=args.style_instruction,
                gpu_memory_gb=args.gpu_memory_gb,
                model_family=args.model_family,
                seed_voice_backend=args.seed_voice_backend,
            )
        )
        return
    if args.command == "one-click":
        _print(
            service.start_one_click_job(
                style_instruction=args.style_instruction,
                gpu_memory_gb=args.gpu_memory_gb,
                mio_base_url=args.mio_base_url,
                model_family=args.model_family,
                seed_voice_backend=args.seed_voice_backend,
            )
        )
        return
    if args.command == "generate-preview":
        _print(service.generate_preview(args.project_id))
        return
    if args.command == "approve-preview":
        _print(
            service.start_dataset_pipeline_job(
                project_id=args.project_id,
                mio_base_url=args.mio_base_url,
                use_preview_reference=args.use_preview_reference,
            )
        )
        return
    if args.command == "build-dataset":
        _print(
            service.build_dataset(
                args.project_id,
                mio_base_url=args.mio_base_url,
                use_preview_reference=args.use_preview_reference,
            )
        )
        return
    if args.command == "write-training-scripts":
        _print(service.write_training_scripts(args.project_id))
        return
    if args.command == "prepare-piper":
        _print(service.prepare_piper(args.project_id))
        return
    if args.command == "train-piper":
        _print(service.train_piper(args.project_id, execute=args.execute))
        return
    if args.command == "train-sbv2":
        _print(
            service.train_sbv2(
                args.project_id,
                execute=args.execute,
                style_bert_vits2_root=args.style_bert_vits2_root,
            )
        )
        return
    if args.command == "export-piper-module":
        _print(
            service.export_piper_module(
                onnx_path=args.onnx_path,
                config_path=args.config_path,
                output_path=args.output_path,
                module_name=args.module_name,
                class_name=args.class_name,
            )
        )
        return
    if args.command == "build-installable-package":
        _print(service.build_installable_package(args.project_id))
        return
    if args.command == "build-installable-sbv2-package":
        _print(
            service.build_installable_sbv2_package(
                args.project_id,
                style_bert_vits2_root=args.style_bert_vits2_root,
            )
        )
        return
    if args.command == "start-job":
        _print(
            service.start_job(
                job_type=args.job_type,
                project_id=args.project_id,
                params=json.loads(args.params_json),
            )
        )
        return
    if args.command == "get-job":
        _print(service.get_job(args.job_id))
        return
    if args.command == "list-jobs":
        _print({"jobs": service.list_jobs(project_id=args.project_id)})
        return
    if args.command == "run-job":
        _print(service.run_job_file(args.job_file))
        return
    if args.command == "serve-api":
        import uvicorn
        from .server import app

        uvicorn.run(app, host=args.host, port=args.port)
        return
    if args.command == "launch-gradio":
        from .gradio_app import build_demo

        build_demo().launch(server_name=args.host, server_port=args.port)
        return


if __name__ == "__main__":
    main()
