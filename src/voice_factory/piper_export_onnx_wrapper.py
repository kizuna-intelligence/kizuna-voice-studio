from __future__ import annotations

import sys


def main() -> None:
    import torch
    import piper_train.export_onnx as export_onnx
    from piper_train.vits.lightning import VitsModel

    class WrappedVitsModel:
        add_model_specific_args = staticmethod(VitsModel.add_model_specific_args)
        load_from_checkpoint = staticmethod(
            lambda checkpoint_path, *args, **kwargs: VitsModel.load_from_checkpoint(
                checkpoint_path,
                *args,
                map_location=torch.device("cpu"),
                **kwargs,
            )
        )

    export_onnx.VitsModel = WrappedVitsModel
    sys.argv = [sys.argv[0], *sys.argv[1:]]
    export_onnx.main()


if __name__ == "__main__":
    main()
