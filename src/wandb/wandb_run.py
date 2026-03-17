from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


class _Config(dict):
    def update(self, *args, **kwargs):  # type: ignore[override]
        kwargs.pop("allow_val_change", None)
        return super().update(*args, **kwargs)


@dataclass
class _BaseRun:
    name: str | None = None
    project: str | None = None
    dir: str | None = None
    id: str = field(default_factory=lambda: uuid4().hex[:8])
    _attach_id: str = field(default_factory=lambda: uuid4().hex)

    def __post_init__(self) -> None:
        self.config = _Config()

    def define_metric(self, *args, **kwargs) -> None:
        return None

    def log(self, *args, **kwargs) -> None:
        return None

    def watch(self, *args, **kwargs) -> None:
        return None

    def unwatch(self, *args, **kwargs) -> None:
        return None

    def use_artifact(self, *args, **kwargs):
        return None

    def log_artifact(self, *args, **kwargs) -> None:
        return None

    def finish(self) -> None:
        return None


class Run(_BaseRun):
    pass


class RunDisabled(_BaseRun):
    pass
