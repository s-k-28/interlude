"""Runtime configuration, loaded from the environment.

Every external dependency is declared here so a missing key fails loudly at
startup rather than three steps into a long pipeline run.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache


class ConfigError(RuntimeError):
    """Raised when required configuration is absent or malformed."""


def _require(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ConfigError(
            f"{name} is not set. Copy .env.example to .env and fill it in."
        )
    return value


def _optional(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


@dataclass(frozen=True, slots=True)
class B2Config:
    """Backblaze B2, addressed through its S3-compatible API."""

    key_id: str
    app_key: str
    bucket: str
    endpoint: str

    @property
    def region(self) -> str:
        """Derive the region from the endpoint host.

        B2 endpoints look like ``s3.us-west-004.backblazeb2.com``; boto3 wants
        ``us-west-004``. Falls back to ``us-west-004`` when the host does not
        match that shape (e.g. a local MinIO used for testing).
        """
        host = self.endpoint.removeprefix("https://").removeprefix("http://")
        parts = host.split(".")
        if len(parts) >= 2 and parts[0] == "s3":
            return parts[1]
        return "us-west-004"

    @classmethod
    def from_env(cls) -> B2Config:
        return cls(
            key_id=_require("B2_KEY_ID"),
            app_key=_require("B2_APP_KEY"),
            bucket=_require("B2_BUCKET"),
            endpoint=_require("B2_ENDPOINT"),
        )


@dataclass(frozen=True, slots=True)
class ProviderConfig:
    """API keys for the model providers used by the pipeline."""

    google_api_key: str = ""
    assemblyai_api_key: str = ""
    elevenlabs_api_key: str = ""
    paritok_api_key: str = ""

    @classmethod
    def from_env(cls) -> ProviderConfig:
        return cls(
            google_api_key=_optional("GOOGLE_API_KEY"),
            assemblyai_api_key=_optional("ASSEMBLYAI_API_KEY"),
            elevenlabs_api_key=_optional("ELEVENLABS_API_KEY"),
            paritok_api_key=_optional("PARITOK_API_KEY"),
        )

    def missing(self) -> list[str]:
        """Provider keys that are absent. Used by /health."""
        names = {
            "GOOGLE_API_KEY": self.google_api_key,
            "ASSEMBLYAI_API_KEY": self.assemblyai_api_key,
            "ELEVENLABS_API_KEY": self.elevenlabs_api_key,
        }
        return sorted(k for k, v in names.items() if not v)


@dataclass(frozen=True, slots=True)
class Settings:
    b2: B2Config
    providers: ProviderConfig
    max_step_retries: int = 3
    paritok_enabled: bool = True

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            b2=B2Config.from_env(),
            providers=ProviderConfig.from_env(),
            max_step_retries=int(_optional("MAX_STEP_RETRIES", "3")),
            paritok_enabled=_optional("PARITOK_ENABLED", "1") not in {"0", "false", "False"},
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings accessor. Raises ConfigError on first call if invalid."""
    return Settings.from_env()
