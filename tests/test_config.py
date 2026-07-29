"""Tests for environment configuration."""

from __future__ import annotations

import pytest

from app.config import B2Config, ConfigError, ProviderConfig


class TestB2Config:
    def test_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("B2_KEY_ID", "k")
        monkeypatch.setenv("B2_APP_KEY", "s")
        monkeypatch.setenv("B2_BUCKET", "b")
        monkeypatch.setenv("B2_ENDPOINT", "https://s3.us-west-004.backblazeb2.com")
        cfg = B2Config.from_env()
        assert cfg.bucket == "b"

    def test_missing_key_raises_with_actionable_message(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("B2_KEY_ID", raising=False)
        with pytest.raises(ConfigError, match="B2_KEY_ID is not set"):
            B2Config.from_env()

    def test_blank_key_treated_as_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("B2_KEY_ID", "   ")
        with pytest.raises(ConfigError):
            B2Config.from_env()

    @pytest.mark.parametrize(
        ("endpoint", "expected"),
        [
            ("https://s3.us-west-004.backblazeb2.com", "us-west-004"),
            ("s3.eu-central-003.backblazeb2.com", "eu-central-003"),
            ("http://localhost:9000", "us-west-004"),
        ],
    )
    def test_region_derivation(self, endpoint: str, expected: str) -> None:
        cfg = B2Config(key_id="k", app_key="s", bucket="b", endpoint=endpoint)
        assert cfg.region == expected


class TestProviderConfig:
    def test_missing_lists_absent_keys(self) -> None:
        cfg = ProviderConfig(google_api_key="g")
        assert cfg.missing() == ["ASSEMBLYAI_API_KEY", "ELEVENLABS_API_KEY"]

    def test_missing_empty_when_all_present(self) -> None:
        cfg = ProviderConfig(
            google_api_key="g", assemblyai_api_key="a", elevenlabs_api_key="e"
        )
        assert cfg.missing() == []
