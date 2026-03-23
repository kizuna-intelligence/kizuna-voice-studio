"""Voice Factory package."""

__all__ = ["VoiceFactory", "VoiceFactoryConfig", "VoiceFactoryService", "VoiceProjectSpec"]


def __getattr__(name: str):
    if name == "VoiceFactory":
        from .sdk import VoiceFactory

        return VoiceFactory
    if name == "VoiceFactoryConfig":
        from .config import VoiceFactoryConfig

        return VoiceFactoryConfig
    if name == "VoiceFactoryService":
        from .service import VoiceFactoryService

        return VoiceFactoryService
    if name == "VoiceProjectSpec":
        from .models import VoiceProjectSpec

        return VoiceProjectSpec
    raise AttributeError(name)
