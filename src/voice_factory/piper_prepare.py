from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path

import librosa
import torch
from piper_train.phonemize.japanese import phonemize_japanese_with_prosody
from piper_train.phonemize.jp_id_map import get_japanese_id_map
from piper_train.vits.mel_processing import spectrogram_torch
from tqdm import tqdm


def preprocess_project(
    project_dir: Path,
    sample_rate: int = 22050,
    *,
    base_hparams: dict | None = None,
) -> Path:
    """Create Piper preprocessed files for a generated LJSpeech dataset."""
    ljspeech_dir = project_dir / "artifacts" / "piper" / "ljspeech_dataset"
    output_dir = project_dir / "artifacts" / "piper" / "preprocessed"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache" / str(sample_rate)
    cache_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = ljspeech_dir / "metadata.csv"
    entries: list[tuple[str, str]] = []
    for line in metadata_path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.split("|", 1)
        if len(parts) == 2:
            entries.append((parts[0].strip(), parts[1].strip()))

    japanese_id_map = get_japanese_id_map()
    base_hparams = base_hparams or {}
    config = {
        "dataset": project_dir.name,
        "audio": {"sample_rate": sample_rate, "quality": "medium"},
        "espeak": {"voice": "ja"},
        "language": {"code": "ja"},
        "inference": {"noise_scale": 0.667, "length_scale": 1, "noise_w": 0.8},
        "phoneme_type": "openjtalk",
        "phoneme_map": {},
        "phoneme_id_map": japanese_id_map,
        "num_symbols": len(japanese_id_map),
        "num_speakers": max(1, int(base_hparams.get("num_speakers", 1) or 1)),
        "speaker_id_map": {},
        "piper_version": "1.6.0",
        "prosody_num_symbols": 11,
        "prosody_id_map": {str(i): [i] for i in range(11)},
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=True, indent=2), encoding="utf-8"
    )

    dataset_path = output_dir / "dataset.jsonl"
    filter_length = 1024
    hop_length = 256
    win_length = 1024

    with dataset_path.open("w", encoding="utf-8") as handle:
        for file_id, text in tqdm(entries, desc="Preparing Piper dataset"):
            wav_path = ljspeech_dir / "wav" / f"{file_id}.wav"
            phonemes, prosody_info_list = phonemize_japanese_with_prosody(text)
            phoneme_ids: list[int] = []
            prosody_features: list[dict | None] = []
            for phoneme, prosody_info in zip(phonemes, prosody_info_list):
                if phoneme not in japanese_id_map:
                    continue
                ids = japanese_id_map[phoneme]
                phoneme_ids.extend(ids)
                for _ in ids:
                    if prosody_info is None:
                        prosody_features.append(None)
                    else:
                        prosody_features.append(
                            {"a1": prosody_info.a1, "a2": prosody_info.a2, "a3": prosody_info.a3}
                        )

            audio_cache_id = sha256(str(wav_path.absolute()).encode()).hexdigest()
            audio_norm_path = cache_dir / f"{audio_cache_id}.pt"
            audio_spec_path = cache_dir / f"{audio_cache_id}.spec.pt"

            if not audio_norm_path.exists():
                audio_array, _ = librosa.load(path=str(wav_path), sr=sample_rate)
                audio_norm_tensor = torch.FloatTensor(audio_array).unsqueeze(0)
                torch.save(audio_norm_tensor, audio_norm_path)
            else:
                audio_norm_tensor = torch.load(audio_norm_path)

            if not audio_spec_path.exists():
                audio_spec_tensor = spectrogram_torch(
                    y=audio_norm_tensor,
                    n_fft=filter_length,
                    sampling_rate=sample_rate,
                    hop_size=hop_length,
                    win_size=win_length,
                    center=False,
                ).squeeze(0)
                torch.save(audio_spec_tensor, audio_spec_path)

            handle.write(
                json.dumps(
                    {
                        "phoneme_ids": phoneme_ids,
                        "audio_norm_path": str(audio_norm_path.resolve()),
                        "audio_spec_path": str(audio_spec_path.resolve()),
                        "text": text,
                        "prosody_features": prosody_features,
                        "prosody_ids": [],
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )

    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--base-hparams-json", type=Path)
    args = parser.parse_args()

    base_hparams = None
    if args.base_hparams_json:
        base_hparams = json.loads(args.base_hparams_json.read_text(encoding="utf-8"))

    output_dir = preprocess_project(
        args.project_dir.expanduser().resolve(),
        sample_rate=args.sample_rate,
        base_hparams=base_hparams,
    )
    print(json.dumps({"preprocessed_dir": str(output_dir)}))


if __name__ == "__main__":
    main()
