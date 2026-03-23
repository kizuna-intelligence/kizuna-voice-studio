from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path


_PRELUDE_TEMPLATE = """
import sys
sys.path.insert(0, {style_bert_vits2_root!r})

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

import torch_directml


class _IdentityDDP(torch.nn.Module):
    def __init__(self, module, device_ids=None):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def __getattr__(self, name):
        if name == "module":
            return super().__getattr__(name)
        return getattr(self.module, name)


DDP = _IdentityDDP
_DIRECTML_DEVICE_INDEX = int(os.environ.get("VOICE_FACTORY_DIRECTML_DEVICE_INDEX", "0"))
_DIRECTML_DEVICE = torch_directml.device(_DIRECTML_DEVICE_INDEX)
device = _DIRECTML_DEVICE


def _tensor_cuda(self, device=None, non_blocking=False, memory_format=None):
    return self.to(_DIRECTML_DEVICE)


def _module_cuda(self, device=None):
    return self.to(_DIRECTML_DEVICE)


torch.Tensor.cuda = _tensor_cuda
torch.nn.Module.cuda = _module_cuda
torch.cuda.set_device = lambda *args, **kwargs: None

import mel_processing as _mel_processing

_original_spectrogram_torch = _mel_processing.spectrogram_torch
_original_mel_spectrogram_torch = _mel_processing.mel_spectrogram_torch
_original_torch_gather = torch.gather
_original_tensor_gather = torch.Tensor.gather


def _cpu_spectrogram_torch(y, *args, **kwargs):
    output_device = y.device
    spec = _original_spectrogram_torch(y.to("cpu"), *args, **kwargs)
    return spec.to(output_device)


def _cpu_mel_spectrogram_torch(y, *args, **kwargs):
    output_device = y.device
    mel = _original_mel_spectrogram_torch(y.to("cpu"), *args, **kwargs)
    return mel.to(output_device)


_mel_processing.spectrogram_torch = _cpu_spectrogram_torch
_mel_processing.mel_spectrogram_torch = _cpu_mel_spectrogram_torch


def _cpu_gather(input, dim, index, *, sparse_grad=False, out=None):
    if input.device.type == "cpu":
        return _original_torch_gather(input, dim, index, sparse_grad=sparse_grad, out=out)
    if out is not None:
        raise RuntimeError("DirectML compatibility gather does not support the out parameter")
    gathered = _original_torch_gather(
        input.to("cpu"),
        dim,
        index.to("cpu"),
        sparse_grad=sparse_grad,
    )
    return gathered.to(input.device)


def _tensor_cpu_gather(self, dim, index, *, sparse_grad=False):
    return _cpu_gather(self, dim, index, sparse_grad=sparse_grad)


torch.gather = _cpu_gather
torch.Tensor.gather = _tensor_cpu_gather

import style_bert_vits2.models.models_jp_extra as _models_jp_extra
import losses as _losses

_original_stochastic_duration_predictor_forward = (
    _models_jp_extra.StochasticDurationPredictor.forward
)
_original_duration_predictor_forward = _models_jp_extra.DurationPredictor.forward
_original_synthesizer_forward = _models_jp_extra.SynthesizerTrn.forward
_original_synthesizer_infer = _models_jp_extra.SynthesizerTrn.infer
_original_kl_loss = _losses.kl_loss
_original_wavlm_forward = _losses.WavLMLoss.forward
_original_wavlm_generator = _losses.WavLMLoss.generator
_original_wavlm_discriminator = _losses.WavLMLoss.discriminator
_original_wavlm_discriminator_forward = _losses.WavLMLoss.discriminator_forward


def _stochastic_duration_predictor_forward_cpu(
    self,
    x,
    x_mask,
    w=None,
    g=None,
    reverse=False,
    noise_scale=1.0,
):
    output_device = x.device
    result = _original_stochastic_duration_predictor_forward(
        self,
        x.to("cpu"),
        x_mask.to("cpu"),
        None if w is None else w.to("cpu"),
        None if g is None else g.to("cpu"),
        reverse=reverse,
        noise_scale=noise_scale,
    )
    return result.to(output_device)


def _duration_predictor_forward_cpu(self, x, x_mask, g=None):
    output_device = x.device
    result = _original_duration_predictor_forward(
        self,
        x.to("cpu"),
        x_mask.to("cpu"),
        None if g is None else g.to("cpu"),
    )
    return result.to(output_device)


_models_jp_extra.StochasticDurationPredictor.forward = (
    _stochastic_duration_predictor_forward_cpu
)
_models_jp_extra.DurationPredictor.forward = _duration_predictor_forward_cpu


def _cpu_kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
    output_device = z_p.device
    return _original_kl_loss(
        z_p.contiguous().to("cpu"),
        logs_q.contiguous().to("cpu"),
        m_p.contiguous().to("cpu"),
        logs_p.contiguous().to("cpu"),
        z_mask.contiguous().to("cpu"),
    ).to(output_device)


_losses.kl_loss = _cpu_kl_loss


def _wavlm_forward_cpu(self, wav, y_rec):
    output_device = y_rec.device
    with torch.no_grad():
        wav_16 = self.resample(wav.to("cpu"))
        wav_embeddings = self.wavlm(
            input_values=wav_16, output_hidden_states=True
        ).hidden_states
    y_rec_16 = self.resample(y_rec.to("cpu"))
    y_rec_embeddings = self.wavlm(
        input_values=y_rec_16, output_hidden_states=True
    ).hidden_states
    floss = 0
    for er, eg in zip(wav_embeddings, y_rec_embeddings):
        floss += torch.mean(torch.abs(er - eg))
    return floss.mean().to(output_device)


def _wavlm_generator_cpu(self, y_rec):
    y_rec_16 = self.resample(y_rec.to("cpu"))
    y_rec_embeddings = self.wavlm(
        input_values=y_rec_16, output_hidden_states=True
    ).hidden_states
    y_rec_embeddings = (
        torch.stack(y_rec_embeddings, dim=1)
        .transpose(-1, -2)
        .flatten(start_dim=1, end_dim=2)
    )
    y_df_hat_g = self.wd(y_rec_embeddings.to(_DIRECTML_DEVICE))
    return torch.mean((1 - y_df_hat_g) ** 2)


def _wavlm_discriminator_cpu(self, wav, y_rec):
    with torch.no_grad():
        wav_16 = self.resample(wav.to("cpu"))
        wav_embeddings = self.wavlm(
            input_values=wav_16, output_hidden_states=True
        ).hidden_states
        y_rec_16 = self.resample(y_rec.to("cpu"))
        y_rec_embeddings = self.wavlm(
            input_values=y_rec_16, output_hidden_states=True
        ).hidden_states
        y_embeddings = (
            torch.stack(wav_embeddings, dim=1)
            .transpose(-1, -2)
            .flatten(start_dim=1, end_dim=2)
        )
        y_rec_embeddings = (
            torch.stack(y_rec_embeddings, dim=1)
            .transpose(-1, -2)
            .flatten(start_dim=1, end_dim=2)
        )
    y_d_rs = self.wd(y_embeddings.to(_DIRECTML_DEVICE))
    y_d_gs = self.wd(y_rec_embeddings.to(_DIRECTML_DEVICE))
    return (torch.mean((1 - y_d_rs) ** 2) + torch.mean(y_d_gs**2)).mean()


def _wavlm_discriminator_forward_cpu(self, wav):
    with torch.no_grad():
        wav_16 = self.resample(wav.to("cpu"))
        wav_embeddings = self.wavlm(
            input_values=wav_16, output_hidden_states=True
        ).hidden_states
        y_embeddings = (
            torch.stack(wav_embeddings, dim=1)
            .transpose(-1, -2)
            .flatten(start_dim=1, end_dim=2)
        )
    return self.wd(y_embeddings.to(_DIRECTML_DEVICE))


_losses.WavLMLoss.forward = _wavlm_forward_cpu
_losses.WavLMLoss.generator = _wavlm_generator_cpu
_losses.WavLMLoss.discriminator = _wavlm_discriminator_cpu
_losses.WavLMLoss.discriminator_forward = _wavlm_discriminator_forward_cpu


def _synthesizer_forward_split(
    self,
    x,
    x_lengths,
    y,
    y_lengths,
    sid,
    tone,
    language,
    bert,
    style_vec,
):
    model_device = _DIRECTML_DEVICE
    if self.n_speakers > 0:
        g = self.emb_g(sid.to(model_device)).unsqueeze(-1)
    else:
        g = self.ref_enc(y.to(model_device).transpose(1, 2)).unsqueeze(-1)
    g_cpu = g.to("cpu")
    x_cpu = x.to("cpu")
    x_lengths_cpu = x_lengths.to("cpu")
    y_cpu = y.to("cpu")
    y_lengths_cpu = y_lengths.to("cpu")
    tone_cpu = tone.to("cpu")
    language_cpu = language.to("cpu")
    bert_cpu = bert.to("cpu")
    style_vec_cpu = style_vec.to("cpu")

    hidden_x, m_p, logs_p, x_mask = self.enc_p(
        x_cpu, x_lengths_cpu, tone_cpu, language_cpu, bert_cpu, style_vec_cpu, g=g_cpu
    )
    z, m_q, logs_q, y_mask = self.enc_q(y_cpu, y_lengths_cpu, g=g_cpu)
    z_p = self.flow(z, y_mask, g=g_cpu)

    with torch.no_grad():
        s_p_sq_r = torch.exp(-2 * logs_p)
        neg_cent1 = torch.sum(
            -0.5 * _models_jp_extra.math.log(2 * _models_jp_extra.math.pi) - logs_p,
            [1],
            keepdim=True,
        )
        neg_cent2 = torch.matmul(-0.5 * (z_p**2).transpose(1, 2), s_p_sq_r)
        neg_cent3 = torch.matmul(z_p.transpose(1, 2), (m_p * s_p_sq_r))
        neg_cent4 = torch.sum(-0.5 * (m_p**2) * s_p_sq_r, [1], keepdim=True)
        neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4
        if self.use_noise_scaled_mas:
            epsilon = (
                torch.std(neg_cent)
                * torch.randn_like(neg_cent)
                * self.current_mas_noise_scale
            )
            neg_cent = neg_cent + epsilon

        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = (
            _models_jp_extra.monotonic_alignment.maximum_path(
                neg_cent, attn_mask.squeeze(1)
            )
            .unsqueeze(1)
            .detach()
        )

    w = attn.sum(2)
    l_length_sdp = self.sdp(hidden_x, x_mask, w, g=g_cpu)
    l_length_sdp = l_length_sdp / torch.sum(x_mask)
    logw_ = torch.log(w + 1e-6) * x_mask
    logw = self.dp(hidden_x, x_mask, g=g_cpu)
    l_length_dp = torch.sum((logw - logw_) ** 2, [1, 2]) / torch.sum(x_mask)
    l_length = l_length_dp + l_length_sdp

    m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
    logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)
    z_slice, ids_slice = _models_jp_extra.commons.rand_slice_segments(
        z, y_lengths_cpu, self.segment_size
    )
    o = self.dec(z_slice.to(model_device), g=g)
    return (
        o,
        l_length.to(model_device),
        attn.to(model_device),
        ids_slice.to(model_device),
        x_mask.to(model_device),
        y_mask.to(model_device),
        (
            z.to(model_device),
            z_p.to(model_device),
            m_p.to(model_device),
            logs_p.to(model_device),
            m_q.to(model_device),
            logs_q.to(model_device),
        ),
        (
            hidden_x.to(model_device),
            logw.to(model_device),
            logw_.to(model_device),
        ),
        g,
    )


def _synthesizer_infer_split(
    self,
    x,
    x_lengths,
    sid,
    tone,
    language,
    bert,
    style_vec,
    noise_scale=0.667,
    length_scale=1.0,
    noise_scale_w=0.8,
    max_len=None,
    sdp_ratio=0.0,
    y=None,
):
    model_device = _DIRECTML_DEVICE
    if self.n_speakers > 0:
        g = self.emb_g(sid.to(model_device)).unsqueeze(-1)
    else:
        assert y is not None
        g = self.ref_enc(y.to(model_device).transpose(1, 2)).unsqueeze(-1)
    g_cpu = g.to("cpu")
    x_cpu = x.to("cpu")
    x_lengths_cpu = x_lengths.to("cpu")
    tone_cpu = tone.to("cpu")
    language_cpu = language.to("cpu")
    bert_cpu = bert.to("cpu")
    style_vec_cpu = style_vec.to("cpu")

    hidden_x, m_p, logs_p, x_mask = self.enc_p(
        x_cpu, x_lengths_cpu, tone_cpu, language_cpu, bert_cpu, style_vec_cpu, g=g_cpu
    )
    logw = self.sdp(
        hidden_x, x_mask, g=g_cpu, reverse=True, noise_scale=noise_scale_w
    ) * sdp_ratio + self.dp(hidden_x, x_mask, g=g_cpu) * (1 - sdp_ratio)
    w = torch.exp(logw) * x_mask * length_scale
    w_ceil = torch.ceil(w)
    y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
    y_mask = torch.unsqueeze(
        _models_jp_extra.commons.sequence_mask(y_lengths, None), 1
    ).to(x_mask.dtype)
    attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
    attn = _models_jp_extra.commons.generate_path(w_ceil, attn_mask)
    m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
    logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)
    z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
    z = self.flow(z_p, y_mask, g=g_cpu, reverse=True)
    o = self.dec((z * y_mask)[:, :, :max_len].to(model_device), g=g)
    return (
        o,
        attn.to(model_device),
        y_mask.to(model_device),
        (
            z.to(model_device),
            z_p.to(model_device),
            m_p.to(model_device),
            logs_p.to(model_device),
        ),
    )


_models_jp_extra.SynthesizerTrn.forward = _synthesizer_forward_split
_models_jp_extra.SynthesizerTrn.infer = _synthesizer_infer_split
"""


def _replace_once(text: str, old: str, new: str) -> str:
    if old not in text:
        raise RuntimeError(f"Expected snippet not found while patching SBV2 trainer: {old[:80]!r}")
    return text.replace(old, new, 1)


def _patch_train_script(
    source: str,
    *,
    style_bert_vits2_root: str,
    debug_duration_component: str | None,
    debug_directml: bool,
) -> str:
    patched = source
    prelude = _PRELUDE_TEMPLATE.format(style_bert_vits2_root=style_bert_vits2_root)
    patched = _replace_once(
        patched,
        "import default_style\n",
        prelude + "\nimport default_style\n",
    )
    patched = _replace_once(
        patched,
        """    backend = "nccl"
    if platform.system() == "Windows":
        backend = "gloo"  # If Windows,switch to gloo backend.
    dist.init_process_group(
        backend=backend,
        init_method="env://",
        timeout=datetime.timedelta(seconds=300),
    )  # Use torchrun instead of mp.spawn
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    n_gpus = dist.get_world_size()
""",
        """    rank = 0
    local_rank = 0
    n_gpus = 1
    device = _DIRECTML_DEVICE
    logger.info(f"Using DirectML device: {torch_directml.device_name(_DIRECTML_DEVICE_INDEX)}")
""",
    )
    patched = _replace_once(patched, "    torch.cuda.set_device(local_rank)\n", "    device = _DIRECTML_DEVICE\n")
    patched = patched.replace(".cuda(local_rank, non_blocking=True)", ".to(device)")
    patched = patched.replace(".cuda(local_rank)", ".to(device)")
    patched = patched.replace(".cuda()", ".to(device)")
    patched = patched.replace(".to(local_rank)", ".to(device)")
    patched = patched.replace("GradScaler(enabled=hps.train.bf16_run)", "GradScaler(enabled=False)")
    patched = patched.replace(
        "with autocast(enabled=hps.train.bf16_run, dtype=torch.bfloat16):",
        "with autocast(enabled=False, dtype=torch.bfloat16):",
    )
    patched = patched.replace("torch.cuda.empty_cache()", "pass")
    patched = patched.replace(
        "                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl\n",
        "                loss_kl = kl_loss(z_p.contiguous(), logs_q.contiguous(), m_p.contiguous(), logs_p.contiguous(), z_mask.contiguous()) * hps.train.c_kl\n",
    )
    patched = _replace_once(
        patched,
        """        wl = WavLMLoss(
            hps.model.slm.model,
            net_wd,
            hps.data.sampling_rate,
            hps.model.slm.sr,
        ).to(device)
""",
        """        wl = WavLMLoss(
            hps.model.slm.model,
            net_wd,
            hps.data.sampling_rate,
            hps.model.slm.sr,
        ).to(device)
        wl.wavlm = wl.wavlm.to("cpu")
        wl.resample = wl.resample.to("cpu")
""",
    )
    patched = _replace_once(
        patched,
        "    if getattr(hps.train, \"freeze_JP_bert\", False):\n",
        "    net_g.enc_p = net_g.enc_p.to(\"cpu\")\n    net_g.enc_q = net_g.enc_q.to(\"cpu\")\n    net_g.flow = net_g.flow.to(\"cpu\")\n    net_g.sdp = net_g.sdp.to(\"cpu\")\n    net_g.dp = net_g.dp.to(\"cpu\")\n    if getattr(hps.train, \"freeze_JP_bert\", False):\n",
    )
    if debug_directml:
        patched = patched.replace(
            "                scaler.scale(loss_dur_disc_all).backward()\n",
            '                print("BACKWARD dur_disc start", flush=True)\n                scaler.scale(loss_dur_disc_all).backward()\n                print("BACKWARD dur_disc done", flush=True)\n',
        )
        patched = patched.replace(
            "                scaler.scale(loss_slm).backward()\n",
            '                print("BACKWARD wavlm_disc start", flush=True)\n                scaler.scale(loss_slm).backward()\n                print("BACKWARD wavlm_disc done", flush=True)\n',
        )
        patched = patched.replace(
            "    initial_step = global_step\n",
            '    initial_step = global_step\n    print(f"TRAIN LOOP READY batches={len(train_loader)}", flush=True)\n',
        )
        patched = patched.replace(
            "    ) in enumerate(train_loader):\n",
            '    ) in enumerate(train_loader):\n        print(f"BATCH {batch_idx} fetched", flush=True)\n',
        )
        patched = patched.replace(
            "        style_vec = style_vec.cuda(local_rank, non_blocking=True)\n",
            '        style_vec = style_vec.cuda(local_rank, non_blocking=True)\n        print(f"BATCH {batch_idx} moved_to_device", flush=True)\n',
        )
        patched = _replace_once(
            patched,
            "        with autocast(enabled=False, dtype=torch.bfloat16):\n            (\n                y_hat,\n",
            '        with autocast(enabled=False, dtype=torch.bfloat16):\n            print(f"BATCH {batch_idx} net_g start", flush=True)\n            (\n                y_hat,\n',
        )
        patched = patched.replace(
            "            mel = spec_to_mel_torch(\n",
            '            print(f"BATCH {batch_idx} net_g done", flush=True)\n            mel = spec_to_mel_torch(\n',
        )
        patched = patched.replace(
            "        scaler.scale(loss_disc_all).backward()\n",
            '        print("BACKWARD disc start", flush=True)\n        scaler.scale(loss_disc_all).backward()\n        print("BACKWARD disc done", flush=True)\n',
        )
        patched = patched.replace(
            "        scaler.scale(loss_gen_all).backward()\n",
            '        if global_step == 0:\n'
            '            _vf_loss_checks = [\n'
            '                ("loss_gen", loss_gen),\n'
            '                ("loss_fm", loss_fm),\n'
            '                ("loss_mel", loss_mel),\n'
            '                ("loss_dur", loss_dur),\n'
            '                ("loss_kl", loss_kl),\n'
            '            ]\n'
            '            if net_dur_disc is not None:\n'
            '                _vf_loss_checks.append(("loss_dur_gen", loss_dur_gen))\n'
            '            if net_wd is not None:\n'
            '                _vf_loss_checks.extend([\n'
            '                    ("loss_lm", loss_lm),\n'
            '                    ("loss_lm_gen", loss_lm_gen),\n'
            '                ])\n'
            '            for _vf_name, _vf_loss in _vf_loss_checks:\n'
            '                optim_g.zero_grad()\n'
            '                print(f"DEBUG gen component start {_vf_name}", flush=True)\n'
            '                scaler.scale(_vf_loss).backward(retain_graph=True)\n'
            '                print(f"DEBUG gen component done {_vf_name}", flush=True)\n'
            '            optim_g.zero_grad()\n'
            '        print("BACKWARD gen start", flush=True)\n'
            '        scaler.scale(loss_gen_all).backward()\n'
            '        print("BACKWARD gen done", flush=True)\n',
        )
    if debug_duration_component == "dp":
        patched = _replace_once(
            patched,
            "        l_length = l_length_dp + l_length_sdp\n",
            "        l_length = l_length_dp\n",
        )
    elif debug_duration_component == "sdp":
        patched = _replace_once(
            patched,
            "        l_length = l_length_dp + l_length_sdp\n",
            "        l_length = l_length_sdp\n",
        )
    return patched


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--style-bert-vits2-root", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--config-path", required=True)
    args = parser.parse_args()

    root = Path(args.style_bert_vits2_root).expanduser().resolve()
    source_path = root / "train_ms_jp_extra.py"
    debug_duration_component = os.environ.get("VOICE_FACTORY_SBV2_DEBUG_DURATION_COMPONENT")
    debug_directml = os.environ.get("VOICE_FACTORY_SBV2_DEBUG_DIRECTML", "").strip() == "1"
    patched_source = _patch_train_script(
        source_path.read_text(encoding="utf-8"),
        style_bert_vits2_root=str(root),
        debug_duration_component=debug_duration_component,
        debug_directml=debug_directml,
    )

    with tempfile.TemporaryDirectory(prefix="sbv2-directml-") as tmpdir:
        patched_path = Path(tmpdir) / "train_ms_jp_extra_directml.py"
        patched_path.write_text(patched_source, encoding="utf-8")
        command = [
            sys.executable,
            str(patched_path),
            "-m",
            args.model_dir,
            "-c",
            args.config_path,
        ]
        subprocess.run(command, check=True, cwd=str(root))


if __name__ == "__main__":
    main()
