from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class VoiceFactoryConfig:
    workspace_root: Path | None = None

    def resolved_workspace_root(self) -> Path | None:
        if self.workspace_root is None:
            return None
        return Path(self.workspace_root).expanduser().resolve()
