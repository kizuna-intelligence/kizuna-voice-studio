---
title: Kizuna Voice Studio
emoji: 🎙️
colorFrom: indigo
colorTo: red
sdk: gradio
sdk_version: 5.21.0
app_file: app.py
python_version: "3.11"
suggested_hardware: zero-a10g
short_description: Generate seed voice previews for Kizuna Voice Studio.
---

# Kizuna Voice Studio

Kizuna Voice Studio is deployed here as a seed voice preview Space for:

- generating a seed voice from Japanese instructions
- previewing the generated voice

This Space is configured as a Gradio Space.

## Runtime notes

- The current organization plan runs on `cpu-basic`; paid GPU or ZeroGPU is not available from this repo today.
- The app stores projects and model caches under `/data` when persistent storage is enabled.
- This Space uses `Kizuna Voice Designer` for seed voice preview generation.
- Full TTS training and package export should be done from the desktop app or another runtime with paid GPU support.

## Environment defaults

- `VOICE_FACTORY_WORKSPACE_ROOT=/data/voice-factory-workspace`
- `HF_HOME=/data/.cache/huggingface`
- `TRANSFORMERS_CACHE=/data/.cache/huggingface/transformers`

## Source

- GitHub: https://github.com/kizuna-intelligence/kizuna-voice-studio
