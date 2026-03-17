from __future__ import annotations

from typing import Any

from .wandb_run import Run, RunDisabled

__version__ = "0.0.0"
run: Run | RunDisabled | None = None


def require(*args, **kwargs) -> None:
    return None


def init(**kwargs: Any) -> RunDisabled:
    global run
    run = RunDisabled(
        name=kwargs.get("name"),
        project=kwargs.get("project"),
        dir=kwargs.get("dir"),
        id=kwargs.get("id") or RunDisabled().id,
    )
    return run


def finish() -> None:
    global run
    if run is not None:
        run.finish()
    run = None


def _attach(*args, **kwargs) -> RunDisabled:
    return run or init(**kwargs)


class Artifact:
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs


class Api:
    def artifact(self, *args, **kwargs):
        return None


class Table:
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs


class Image:
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs


class Audio:
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs


__all__ = [
    "Api",
    "Artifact",
    "Audio",
    "Image",
    "Run",
    "RunDisabled",
    "Table",
    "finish",
    "init",
    "require",
    "run",
]
