from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _load_package(src_root: Path, module_name: str):
    sys.path.insert(0, str(src_root))
    try:
        return __import__(module_name, fromlist=["load_voice"])
    finally:
        if sys.path and sys.path[0] == str(src_root):
            sys.path.pop(0)


def _voice_for_family(package, args: argparse.Namespace):
    if args.family == "piper":
        return package.load_voice()
    if args.family == "sbv2":
        return package.load_voice(device=args.device)
    if args.family == "miotts":
        return package.load_voice(
            api_base_url=args.mio_base_url,
            model_id=args.mio_model_id,
        )
    raise ValueError(f"Unsupported family: {args.family}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True, choices=["piper", "sbv2", "miotts"])
    parser.add_argument("--package-dir", required=True)
    parser.add_argument("--module-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--texts-json", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mio-base-url")
    parser.add_argument("--mio-model-id")
    args = parser.parse_args()

    package_dir = Path(args.package_dir)
    src_root = package_dir / "src"
    package = _load_package(src_root, args.module_name)
    voice = _voice_for_family(package, args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    texts = [text.strip() for text in json.loads(args.texts_json) if str(text).strip()]

    samples: list[dict[str, str]] = []
    for index, text in enumerate(texts, start=1):
        sample_id = f"sample_{index:02d}"
        output_path = output_dir / f"{sample_id}.wav"
        voice.save_wav(text, output_path)
        samples.append(
            {
                "id": sample_id,
                "text": text,
                "audio_path": str(output_path),
            }
        )

    sys.stdout.write(
        json.dumps(
            {
                "family": args.family,
                "samples": samples,
            },
            ensure_ascii=False,
        )
        + "\n"
    )
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
