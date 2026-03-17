from __future__ import annotations

import os
import sys
import types
import importlib.machinery
import importlib.util


def main() -> None:
    disable_wavlm_flag = False
    if "--disable-wavlm" in sys.argv:
        disable_wavlm_flag = True
        sys.argv = [arg for arg in sys.argv if arg != "--disable-wavlm"]

    disable_wandb = os.environ.get("VOICE_FACTORY_DISABLE_WANDB", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if disable_wandb and "wandb" not in sys.modules:
        # Make pytorch_lightning treat wandb as unavailable. This avoids importing
        # a broken local wandb install when Piper only needs TensorBoard logging.
        fake_wandb = types.ModuleType("wandb")
        fake_wandb.__spec__ = importlib.machinery.ModuleSpec("wandb", loader=None)
        sys.modules["wandb"] = fake_wandb
        original_find_spec = importlib.util.find_spec

        def patched_find_spec(name, package=None):
            if name == "wandb":
                return None
            return original_find_spec(name, package)

        importlib.util.find_spec = patched_find_spec

    import piper_train.__main__ as piper_main
    from piper_train.vits.lightning import VitsModel
    from piper_train.vits.models import WavLMDiscriminator

    disable_wavlm = disable_wavlm_flag or (
        os.environ.get("VOICE_FACTORY_PIPER_DISABLE_WAVLM", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
    )

    original_wavlm_init = WavLMDiscriminator.__init__

    def patched_wavlm_init(self, *args, **kwargs):
        original_wavlm_init(self, *args, **kwargs)
        feature_extractor = getattr(getattr(self, "wavlm", None), "feature_extractor", None)
        if feature_extractor is not None and hasattr(feature_extractor, "_freeze_parameters"):
            # transformers>=4.57 can raise on non-leaf requires_grad assignment
            # unless the frozen feature extractor also flips its internal flag.
            feature_extractor._freeze_parameters()

    WavLMDiscriminator.__init__ = patched_wavlm_init

    piper_main.WANDB_AVAILABLE = False

    if disable_wavlm:
        class WrappedVitsModel:
            add_model_specific_args = staticmethod(VitsModel.add_model_specific_args)
            load_from_checkpoint = staticmethod(VitsModel.load_from_checkpoint)

            def __new__(cls, *args, **kwargs):
                kwargs["use_wavlm_discriminator"] = False
                kwargs["c_wavlm"] = 0.0
                return VitsModel(*args, **kwargs)

        piper_main.VitsModel = WrappedVitsModel

        if "--c-wavlm" not in sys.argv:
            sys.argv.extend(["--c-wavlm", "0.0"])

    piper_main.main()


if __name__ == "__main__":
    main()
