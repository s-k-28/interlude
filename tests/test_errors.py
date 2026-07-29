"""Tests for duck-typed S3 error inspection."""

from __future__ import annotations

from app.storage.errors import error_code, is_not_found, is_retryable


class _BotoShaped(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


class TestErrorCode:
    def test_extracts_code(self) -> None:
        assert error_code(_BotoShaped("NoSuchKey")) == "NoSuchKey"

    def test_plain_exception_returns_none(self) -> None:
        # None means "not botocore-shaped" -> caller must re-raise, never
        # silently treat an unrelated failure as a missing object.
        assert error_code(ValueError("boom")) is None

    def test_malformed_response_returns_none(self) -> None:
        exc = Exception()
        exc.response = "not a dict"  # type: ignore[attr-defined]
        assert error_code(exc) is None

    def test_missing_error_key_returns_none(self) -> None:
        exc = Exception()
        exc.response = {"ResponseMetadata": {}}  # type: ignore[attr-defined]
        assert error_code(exc) is None


class TestClassifiers:
    def test_404_is_not_found(self) -> None:
        assert is_not_found(_BotoShaped("404"))

    def test_nosuchkey_is_not_found(self) -> None:
        assert is_not_found(_BotoShaped("NoSuchKey"))

    def test_access_denied_is_not_not_found(self) -> None:
        # Critical: a permissions failure must never be mistaken for absence,
        # or the pipeline would silently regenerate assets it cannot read.
        assert not is_not_found(_BotoShaped("AccessDenied"))

    def test_slowdown_is_retryable(self) -> None:
        assert is_retryable(_BotoShaped("SlowDown"))

    def test_access_denied_is_not_retryable(self) -> None:
        assert not is_retryable(_BotoShaped("AccessDenied"))
