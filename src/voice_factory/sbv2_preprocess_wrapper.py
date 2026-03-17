from __future__ import annotations

import argparse
from multiprocessing import cpu_count
from pathlib import Path


def main() -> None:
    import numpy as np
    from gradio_tabs.train import preprocess_all

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", "-m", type=str, required=True)
    parser.add_argument("--batch_size", "-b", type=int, default=2)
    parser.add_argument("--epochs", "-e", type=int, default=100)
    parser.add_argument("--save_every_steps", "-s", type=int, default=1000)
    parser.add_argument("--num_processes", type=int, default=cpu_count() // 2)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--trim", action="store_true")
    parser.add_argument("--freeze_EN_bert", action="store_true")
    parser.add_argument("--freeze_JP_bert", action="store_true")
    parser.add_argument("--freeze_ZH_bert", action="store_true")
    parser.add_argument("--freeze_style", action="store_true")
    parser.add_argument("--freeze_decoder", action="store_true")
    parser.add_argument("--use_jp_extra", action="store_true")
    parser.add_argument("--val_per_lang", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=200)
    parser.add_argument("--yomi_error", type=str, default="raise")
    args = parser.parse_args()

    preprocess_all(
        model_name=args.model_name,
        batch_size=args.batch_size,
        epochs=args.epochs,
        save_every_steps=args.save_every_steps,
        num_processes=args.num_processes,
        normalize=args.normalize,
        trim=args.trim,
        freeze_EN_bert=args.freeze_EN_bert,
        freeze_JP_bert=args.freeze_JP_bert,
        freeze_ZH_bert=args.freeze_ZH_bert,
        freeze_style=args.freeze_style,
        freeze_decoder=args.freeze_decoder,
        use_jp_extra=args.use_jp_extra,
        val_per_lang=args.val_per_lang,
        log_interval=args.log_interval,
        yomi_error=args.yomi_error,
    )

    wav_dir = Path("Data") / args.model_name / "wavs"
    if wav_dir.exists():
        placeholder = np.zeros((256,), dtype=np.float32)
        for wav_path in wav_dir.rglob("*.wav"):
            npy_path = Path(f"{wav_path}.npy")
            if not npy_path.exists():
                np.save(npy_path, placeholder)


if __name__ == "__main__":
    main()
