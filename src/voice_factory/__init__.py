"""Voice Factory package."""

__all__ = ["VoiceFactoryService"]


def __getattr__(name: str):
    if name == "VoiceFactoryService":
        from .service import VoiceFactoryService

        return VoiceFactoryService
    raise AttributeError(name)
