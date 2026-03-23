from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path


def _patch_directml_compatibility(device_index: int) -> object:
    import torch
    import torch_directml
    import piper_train.vits.lightning as light_mod
    import piper_train.vits.mel_processing as melp
    import piper_train.vits.models as models_mod
    from piper_train.vits import commons, monotonic_align

    device = torch_directml.device(device_index)

    orig_spec_to_mel = light_mod.spec_to_mel_torch
    orig_mel_spec = light_mod.mel_spectrogram_torch

    def cpu_spec_to_mel_torch(spec, *args, **kwargs):
        output_device = spec.device
        mel = orig_spec_to_mel(spec.to("cpu"), *args, **kwargs)
        return mel.to(output_device)

    def cpu_mel_spectrogram_torch(y, *args, **kwargs):
        output_device = y.device
        mel = orig_mel_spec(y.to("cpu"), *args, **kwargs)
        return mel.to(output_device)

    melp.spec_to_mel_torch = cpu_spec_to_mel_torch
    melp.mel_spectrogram_torch = cpu_mel_spectrogram_torch
    light_mod.spec_to_mel_torch = cpu_spec_to_mel_torch
    light_mod.mel_spectrogram_torch = cpu_mel_spectrogram_torch

    def split_forward(self, x, x_lengths, y, y_lengths, sid=None, prosody_features=None):
        x_cpu = x.to("cpu")
        x_lengths_cpu = x_lengths.to("cpu")
        y_cpu = y.to("cpu")
        y_lengths_cpu = y_lengths.to("cpu")
        x_enc, m_p, logs_p, x_mask = self.enc_p(x_cpu, x_lengths_cpu)
        if self.n_speakers > 1:
            assert sid is not None, "Missing speaker id"
            g = self.emb_g(sid.to("cpu")).unsqueeze(-1)
        else:
            g = None
        z, m_q, logs_q, y_mask = self.enc_q(y_cpu, y_lengths_cpu, g=g)
        z_p = self.flow(z, y_mask, g=g)

        with torch.no_grad():
            s_p_sq_r = torch.exp(-2 * logs_p)
            neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True)
            neg_cent2 = torch.matmul(-0.5 * (z_p**2).transpose(1, 2), s_p_sq_r)
            neg_cent3 = torch.matmul(z_p.transpose(1, 2), (m_p * s_p_sq_r))
            neg_cent4 = torch.sum(-0.5 * (m_p**2) * s_p_sq_r, [1], keepdim=True)
            neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4
            attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
            attn = (
                monotonic_align.maximum_path(neg_cent, attn_mask.squeeze(1))
                .unsqueeze(1)
                .detach()
            )

        prosody_cpu = None if prosody_features is None else prosody_features.to("cpu")
        x_dp = self._prepare_prosody_input(x_enc, x_mask, prosody_cpu)
        w = attn.sum(2)
        if self.use_sdp:
            l_length = self.dp(x_dp, x_mask, w, g=g)
            l_length = l_length / torch.sum(x_mask)
        else:
            logw_ = torch.log(w + 1e-6) * x_mask
            logw = self.dp(x_dp, x_mask, g=g)
            l_length = torch.sum((logw - logw_) ** 2, [1, 2]) / torch.sum(x_mask)

        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)
        z_slice, ids_slice = commons.rand_slice_segments(z, y_lengths_cpu, self.segment_size)
        decoder_g = None if g is None else g.to(device)
        o = self.dec(z_slice.to(device), g=decoder_g)
        return (
            o,
            l_length.to(device),
            attn.to(device),
            ids_slice.to(device),
            x_mask.to(device),
            y_mask.to(device),
            (
                z.to(device),
                z_p.to(device),
                m_p.to(device),
                logs_p.to(device),
                m_q.to(device),
                logs_q.to(device),
            ),
        )

    def split_infer(
        self,
        x,
        x_lengths,
        sid=None,
        noise_scale=0.667,
        length_scale=1.0,
        noise_scale_w=0.8,
        max_len=None,
        prosody_features=None,
    ):
        x_cpu = x.to("cpu")
        x_lengths_cpu = x_lengths.to("cpu")
        x_enc, m_p, logs_p, x_mask = self.enc_p(x_cpu, x_lengths_cpu)
        if self.n_speakers > 1:
            assert sid is not None, "Missing speaker id"
            g = self.emb_g(sid.to("cpu")).unsqueeze(-1)
        else:
            g = None
        prosody_cpu = None if prosody_features is None else prosody_features.to("cpu")
        x_dp = self._prepare_prosody_input(x_enc, x_mask, prosody_cpu)
        if self.use_sdp:
            logw = self.dp(x_dp, x_mask, g=g, reverse=True, noise_scale=noise_scale_w)
        else:
            logw = self.dp(x_dp, x_mask, g=g)
        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, y_lengths.max()), 1).type_as(x_mask)
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = commons.generate_path(w_ceil, attn_mask)
        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)
        if getattr(self, "onnx_export_mode", False):
            z_p = m_p
        else:
            z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.flow(z_p, y_mask, g=g, reverse=True)
        decoder_g = None if g is None else g.to(device)
        o = self.dec((z * y_mask).to(device)[:, :, :max_len], g=decoder_g)
        return (
            o,
            attn.to(device),
            y_mask.to(device),
            (z.to(device), z_p.to(device), m_p.to(device), logs_p.to(device)),
        )

    models_mod.SynthesizerTrn.forward = split_forward
    models_mod.SynthesizerTrn.infer = split_infer
    return device


def _save_checkpoint(model, checkpoint_path: Path, epoch: int, global_step: int) -> None:
    import torch

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "state_dict": model.state_dict(),
        "hyper_parameters": dict(model.hparams),
        "pytorch-lightning_version": "2.6.1",
        "epoch": epoch,
        "global_step": global_step,
    }
    torch.save(ckpt, checkpoint_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--checkpoint-epochs", type=int, default=1)
    parser.add_argument("--quality", default="medium", choices=("x-low", "medium", "high"))
    parser.add_argument("--no-pin-memory", action="store_true")
    parser.add_argument("--disable_auto_lr_scaling", action="store_true")
    parser.add_argument("--base_lr", type=float, default=2e-4)
    parser.add_argument("--precision", default="32-true")
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--default_root_dir", default=None)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--samples-per-speaker", type=int, default=0)
    parser.add_argument("--wavlm-model-name", default="microsoft/wavlm-base-plus")
    parser.add_argument("--c-wavlm", type=float, default=0.0)
    parser.add_argument("--disable-wavlm", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--devices", default="1")

    from piper_train.vits.lightning import VitsModel

    VitsModel.add_model_specific_args(parser)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    default_root_dir = Path(args.default_root_dir) if args.default_root_dir else dataset_dir

    import json
    import torch

    device_index = int((__import__("os").environ.get("VOICE_FACTORY_DIRECTML_DEVICE_INDEX") or "0").strip() or "0")
    device = _patch_directml_compatibility(device_index)

    config_path = dataset_dir / "config.json"
    dataset_path = dataset_dir / "dataset.jsonl"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    num_symbols = int(config["num_symbols"])
    num_speakers = int(config["num_speakers"])
    sample_rate = int(config["audio"]["sample_rate"])

    dict_args = vars(args).copy()
    dict_args["dataset"] = [dataset_path]
    dict_args["num_symbols"] = num_symbols
    dict_args["num_speakers"] = num_speakers
    dict_args["sample_rate"] = sample_rate
    dict_args["learning_rate"] = args.base_lr
    dict_args["use_wavlm_discriminator"] = not args.disable_wavlm
    dict_args["default_root_dir"] = str(default_root_dir)
    dict_args["no_pin_memory"] = bool(args.no_pin_memory)
    dict_args.pop("dataset_dir", None)
    dict_args.pop("checkpoint_epochs", None)
    dict_args.pop("quality", None)
    dict_args.pop("disable_auto_lr_scaling", None)
    dict_args.pop("base_lr", None)
    dict_args.pop("precision", None)
    dict_args.pop("default_root_dir", None)
    dict_args.pop("resume_from_checkpoint", None)
    dict_args.pop("disable_wavlm", None)
    dict_args.pop("accelerator", None)
    dict_args.pop("devices", None)

    torch.manual_seed(args.seed)

    if args.quality == "x-low":
        dict_args.setdefault("hidden_channels", 96)
        dict_args.setdefault("inter_channels", 96)
        dict_args.setdefault("filter_channels", 384)
    elif args.quality == "high":
        dict_args.setdefault("resblock", "1")
        dict_args.setdefault("resblock_kernel_sizes", (3, 7, 11))
        dict_args.setdefault("resblock_dilation_sizes", ((1, 3, 5), (1, 3, 5), (1, 3, 5)))
        dict_args.setdefault("upsample_rates", (8, 8, 2, 2))
        dict_args.setdefault("upsample_initial_channel", 512)
        dict_args.setdefault("upsample_kernel_sizes", (16, 16, 4, 4))

    model = VitsModel(**dict_args)
    model._log_with_batch_info = lambda *unused_args, **unused_kwargs: None

    model.model_g.enc_p = model.model_g.enc_p.to("cpu")
    model.model_g.enc_q = model.model_g.enc_q.to("cpu")
    model.model_g.flow = model.model_g.flow.to("cpu")
    model.model_g.dp = model.model_g.dp.to("cpu")
    if hasattr(model.model_g, "emb_g"):
        model.model_g.emb_g = model.model_g.emb_g.to("cpu")
    model.model_g.dec = model.model_g.dec.to(device)
    model.model_d = model.model_d.to(device)
    if model.model_d_wavlm is not None:
        model.model_d_wavlm = model.model_d_wavlm.to("cpu")

    optimizers, schedulers = model.configure_optimizers()
    opt_g, opt_d = optimizers
    scheduler_g, scheduler_d = schedulers

    start_epoch = 0
    global_step = 0
    if args.resume_from_checkpoint:
        checkpoint = torch.load(args.resume_from_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        start_epoch = int(checkpoint.get("epoch", 0))
        global_step = int(checkpoint.get("global_step", 0))

    checkpoints_dir = default_root_dir / "lightning_logs" / "version_0" / "checkpoints"
    last_checkpoint = checkpoints_dir / "last.ckpt"

    for epoch in range(start_epoch, args.max_epochs):
        loader = model.train_dataloader()
        model.train()
        for batch_idx, batch in enumerate(loader):
            for name in (
                "phoneme_ids",
                "phoneme_lengths",
                "audios",
                "audio_lengths",
                "spectrograms",
                "spectrogram_lengths",
                "speaker_ids",
                "prosody_features",
            ):
                value = getattr(batch, name)
                if value is not None:
                    setattr(batch, name, value.to(device))

            opt_g.zero_grad(set_to_none=True)
            loss_g = model.training_step_g(batch)
            if torch.isnan(loss_g.detach()).any():
                raise RuntimeError("DirectML Piper training produced NaN generator loss.")
            loss_g.backward()
            opt_g.step()

            opt_d.zero_grad(set_to_none=True)
            loss_d = model.training_step_d(batch)
            if torch.isnan(loss_d.detach()).any():
                raise RuntimeError("DirectML Piper training produced NaN discriminator loss.")
            loss_d.backward()
            opt_d.step()

            global_step += 1
            print(
                f"epoch={epoch} batch={batch_idx} "
                f"loss_g={float(loss_g.detach().to('cpu')):.6f} "
                f"loss_d={float(loss_d.detach().to('cpu')):.6f}",
                flush=True,
            )

        scheduler_g.step()
        scheduler_d.step()
        if ((epoch + 1) % max(1, args.checkpoint_epochs)) == 0:
            epoch_checkpoint = checkpoints_dir / f"epoch={epoch}-step={global_step}.ckpt"
            _save_checkpoint(model, epoch_checkpoint, epoch, global_step)
        _save_checkpoint(model, last_checkpoint, epoch, global_step)


if __name__ == "__main__":
    main()
