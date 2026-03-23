from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from multiprocessing import cpu_count
from pathlib import Path


_BERT_GEN_PRELUDE = """
try:
    import transformers.utils.import_utils as _transformers_import_utils
except Exception:
    _transformers_import_utils = None
else:
    if hasattr(_transformers_import_utils, "check_torch_load_is_safe"):
        _transformers_import_utils.check_torch_load_is_safe = lambda *args, **kwargs: None

try:
    import transformers.modeling_utils as _transformers_modeling_utils
except Exception:
    _transformers_modeling_utils = None
else:
    if hasattr(_transformers_modeling_utils, "check_torch_load_is_safe"):
        _transformers_modeling_utils.check_torch_load_is_safe = lambda *args, **kwargs: None
"""

_BERT_GEN_EXTRACT_OVERRIDE = """
def extract_bert_feature(
    text,
    word2ph,
    language,
    device,
    assist_text=None,
    assist_text_weight=0.7,
):
    if language != Languages.JP:
        return _original_extract_bert_feature(
            text,
            word2ph,
            language,
            device,
            assist_text,
            assist_text_weight,
        )

    import torch
    from style_bert_vits2.nlp import bert_models
    from style_bert_vits2.nlp.japanese.g2p import text_to_sep_kata

    text = "".join(text_to_sep_kata(text, raise_yomi_error=False)[0])
    if assist_text:
        assist_text = "".join(text_to_sep_kata(assist_text, raise_yomi_error=False)[0])

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    model = bert_models.load_model(Languages.JP, device_map=device)
    bert_models.transfer_model(Languages.JP, device)

    style_res_mean = None
    with torch.no_grad():
        tokenizer = bert_models.load_tokenizer(Languages.JP)
        inputs = tokenizer(text, return_tensors="pt")
        for key in inputs:
            inputs[key] = inputs[key].to(device)
        res = model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()
        if assist_text:
            style_inputs = tokenizer(assist_text, return_tensors="pt")
            for key in style_inputs:
                style_inputs[key] = style_inputs[key].to(device)
            style_res = model(**style_inputs, output_hidden_states=True)
            style_res = torch.cat(style_res["hidden_states"][-3:-2], -1)[0].cpu()
            style_res_mean = style_res.mean(0)

    assert len(word2ph) == res.shape[0], (len(word2ph), int(res.shape[0]), text)
    phone_level_feature = []
    for index, phone_count in enumerate(word2ph):
        if assist_text:
            assert style_res_mean is not None
            repeat_feature = (
                res[index].repeat(phone_count, 1) * (1 - assist_text_weight)
                + style_res_mean.repeat(phone_count, 1) * assist_text_weight
            )
        else:
            repeat_feature = res[index].repeat(phone_count, 1)
        phone_level_feature.append(repeat_feature)

    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    return phone_level_feature.T
"""

_STYLE_GEN_PRELUDE = """
import huggingface_hub

_original_hf_hub_download = huggingface_hub.hf_hub_download


def _hf_hub_download_compat(*args, use_auth_token=None, token=None, **kwargs):
    if token is None and use_auth_token is not None:
        token = use_auth_token
    return _original_hf_hub_download(*args, token=token, **kwargs)


huggingface_hub.hf_hub_download = _hf_hub_download_compat
"""


def _patch_once(source: str, old: str, new: str, *, label: str) -> str:
    if old not in source:
        raise RuntimeError(f"Expected snippet not found while patching {label}: {old[:80]!r}")
    return source.replace(old, new, 1)


def _rebuild_word2ph_lists(*, train_path: Path, val_path: Path) -> None:
    from style_bert_vits2.constants import Languages
    from style_bert_vits2.nlp import bert_models
    from style_bert_vits2.nlp.japanese import g2p as japanese_g2p

    handle_long = getattr(japanese_g2p, "__handle_long")
    kata_to_phoneme_list = getattr(japanese_g2p, "__kata_to_phoneme_list")
    distribute_phone = getattr(japanese_g2p, "__distribute_phone")
    tokenizer = bert_models.load_tokenizer(Languages.JP)

    def rebuild_line(line: str, *, path: Path, line_number: int) -> str:
        parts = line.rstrip("\n").split("|")
        if len(parts) != 7:
            raise RuntimeError(f"Unexpected SBV2 line format in {path}:{line_number}: {line.rstrip()}")
        wav_path, speaker, language, text, phones, tones, _word2ph = parts
        sep_text, sep_kata = japanese_g2p.text_to_sep_kata(text, raise_yomi_error=False)
        sep_phonemes = handle_long([kata_to_phoneme_list(kata) for kata in sep_kata])
        if len(sep_text) != len(sep_phonemes):
            raise RuntimeError(
                f"Mismatch while rebuilding word2ph in {path}:{line_number}: {len(sep_text)} text units vs {len(sep_phonemes)} phoneme groups"
            )
        char_level_word2ph: list[int] = []
        for text_unit, phoneme_group in zip(sep_text, sep_phonemes):
            char_count = len(text_unit)
            if char_count <= 0:
                raise RuntimeError(f"Empty text unit while rebuilding word2ph in {path}:{line_number}: {text}")
            char_level_word2ph.extend(distribute_phone(len(phoneme_group), char_count))
        joined_text = "".join(sep_text)
        encoded = tokenizer(joined_text, return_offsets_mapping=True, add_special_tokens=True)
        offset_mapping = encoded.get("offset_mapping")
        if offset_mapping is None:
            raise RuntimeError("Tokenizer did not return offset_mapping; cannot rebuild SBV2 word2ph safely")
        rebuilt_word2ph = [1]
        for start, end in offset_mapping:
            if start == end == 0:
                continue
            rebuilt_word2ph.append(sum(char_level_word2ph[start:end]))
        rebuilt_word2ph.append(1)
        phone_count = len(phones.split())
        if sum(rebuilt_word2ph) != phone_count:
            raise RuntimeError(
                f"Rebuilt word2ph does not match phone count in {path}:{line_number}: {sum(rebuilt_word2ph)} != {phone_count}"
            )
        input_ids = encoded.get("input_ids")
        if input_ids is not None and len(rebuilt_word2ph) != len(input_ids):
            raise RuntimeError(
                f"Rebuilt word2ph length does not match tokenizer input length in {path}:{line_number}: {len(rebuilt_word2ph)} != {len(input_ids)}"
            )
        return "|".join([wav_path, speaker, language, text, phones, tones, " ".join(str(v) for v in rebuilt_word2ph)]) + "\n"

    for path in (train_path, val_path):
        if not path.exists() or path.stat().st_size == 0:
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        rebuilt_lines = [rebuild_line(line, path=path, line_number=index) for index, line in enumerate(lines, start=1)]
        path.write_text("".join(rebuilt_lines), encoding="utf-8")


def main() -> None:
    from gradio_tabs.train import initialize

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

    ok, message = initialize(
        model_name=args.model_name,
        batch_size=args.batch_size,
        epochs=args.epochs,
        save_every_steps=args.save_every_steps,
        freeze_EN_bert=args.freeze_EN_bert,
        freeze_JP_bert=args.freeze_JP_bert,
        freeze_ZH_bert=args.freeze_ZH_bert,
        freeze_style=args.freeze_style,
        freeze_decoder=args.freeze_decoder,
        use_jp_extra=args.use_jp_extra,
        log_interval=args.log_interval,
    )
    if not ok:
        raise RuntimeError(message)

    config_path = Path("Data") / args.model_name / "config.json"
    esd_path = Path("Data") / args.model_name / "esd.list"
    train_path = Path("Data") / args.model_name / "train.list"
    val_path = Path("Data") / args.model_name / "val.list"
    raw_path = Path("Data") / args.model_name / "raw"
    wav_path = Path("Data") / args.model_name / "wavs"

    resample_command = [
        sys.executable,
        "resample.py",
        "-i",
        str(raw_path),
        "-o",
        str(wav_path),
        "--num_processes",
        str(args.num_processes),
        "--sr",
        "44100",
    ]
    if args.normalize:
        resample_command.append("--normalize")
    if args.trim:
        resample_command.append("--trim")
    subprocess.run(resample_command, check=True)

    preprocess_source = Path("preprocess_text.py").read_text(encoding="utf-8")
    patched_preprocess = _patch_once(
        preprocess_source,
        "pyopenjtalk_worker.initialize_worker()",
        "# Worker startup disabled by voice_factory.sbv2_preprocess_wrapper",
        label="preprocess_text.py",
    )
    bert_gen_source = Path("bert_gen.py").read_text(encoding="utf-8")
    patched_bert_gen = _patch_once(
        bert_gen_source,
        "from style_bert_vits2.nlp import cleaned_text_to_sequence, extract_bert_feature\n",
        "from style_bert_vits2.nlp import cleaned_text_to_sequence, extract_bert_feature as _original_extract_bert_feature\n",
        label="bert_gen.py",
    )
    patched_bert_gen = _patch_once(
        patched_bert_gen,
        "from tqdm import tqdm\n",
        "from tqdm import tqdm\n" + _BERT_GEN_PRELUDE + "\n",
        label="bert_gen.py",
    )
    patched_bert_gen = _patch_once(
        patched_bert_gen,
        "from style_bert_vits2.utils.stdout_wrapper import SAFE_STDOUT\n",
        "from style_bert_vits2.utils.stdout_wrapper import SAFE_STDOUT\n\n" + _BERT_GEN_EXTRACT_OVERRIDE + "\n",
        label="bert_gen.py",
    )
    patched_bert_gen = _patch_once(
        patched_bert_gen,
        "pyopenjtalk_worker.initialize_worker()",
        "# Worker startup disabled by voice_factory.sbv2_preprocess_wrapper",
        label="bert_gen.py",
    )
    style_gen_source = Path("style_gen.py").read_text(encoding="utf-8")
    patched_style_gen = _patch_once(
        style_gen_source,
        "from pyannote.audio import Inference, Model\n",
        _STYLE_GEN_PRELUDE + "\nfrom pyannote.audio import Inference, Model\n",
        label="style_gen.py",
    )
    with tempfile.TemporaryDirectory(prefix="sbv2-preprocess-") as tmpdir:
        patched_preprocess_path = Path(tmpdir) / "preprocess_text_noworker.py"
        patched_preprocess_path.write_text(patched_preprocess, encoding="utf-8")
        patched_bert_gen_path = Path(tmpdir) / "bert_gen_nocheck.py"
        patched_bert_gen_path.write_text(patched_bert_gen, encoding="utf-8")
        patched_style_gen_path = Path(tmpdir) / "style_gen_compat.py"
        patched_style_gen_path.write_text(patched_style_gen, encoding="utf-8")
        env = os.environ.copy()
        cwd = os.getcwd()
        env["PYTHONPATH"] = cwd if not env.get("PYTHONPATH") else cwd + os.pathsep + env["PYTHONPATH"]
        preprocess_command = [
            sys.executable,
            str(patched_preprocess_path),
            "--config-path",
            str(config_path),
            "--transcription-path",
            str(esd_path),
            "--train-path",
            str(train_path),
            "--val-path",
            str(val_path),
            "--val-per-lang",
            str(args.val_per_lang),
            "--yomi_error",
            args.yomi_error,
            "--correct_path",
        ]
        if args.use_jp_extra:
            preprocess_command.append("--use_jp_extra")
        subprocess.run(preprocess_command, check=True, env=env)
        _rebuild_word2ph_lists(train_path=train_path, val_path=val_path)
        subprocess.run(
            [sys.executable, str(patched_bert_gen_path), "--config", str(config_path)],
            check=True,
            env=env,
        )
        subprocess.run(
            [
                sys.executable,
                str(patched_style_gen_path),
                "--config",
                str(config_path),
                "--num_processes",
                str(args.num_processes),
            ],
            check=True,
            env=env,
        )

if __name__ == "__main__":
    main()
