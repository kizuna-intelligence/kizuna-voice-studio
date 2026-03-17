"""Runtime compatibility patches for third-party training tools."""

from __future__ import annotations

try:
    import pyopenjtalk
except Exception:  # pragma: no cover - optional dependency during import
    pyopenjtalk = None

if pyopenjtalk is not None and not hasattr(pyopenjtalk, "unset_user_dict"):
    def _unset_user_dict() -> None:
        return None

    pyopenjtalk.unset_user_dict = _unset_user_dict
