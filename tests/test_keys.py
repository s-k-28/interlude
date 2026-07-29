"""Tests for content-addressed key construction."""

from __future__ import annotations

import hashlib

import pytest

from app.storage.keys import content_key, manifest_key, sha256_hex


class TestSha256Hex:
    def test_matches_hashlib(self) -> None:
        assert sha256_hex(b"hello") == hashlib.sha256(b"hello").hexdigest()

    def test_empty_input(self) -> None:
        assert len(sha256_hex(b"")) == 64


class TestContentKey:
    def test_is_deterministic(self) -> None:
        assert content_key(b"x", namespace="source") == content_key(b"x", namespace="source")

    def test_differs_by_content(self) -> None:
        assert content_key(b"a", namespace="source") != content_key(b"b", namespace="source")

    def test_differs_by_namespace(self) -> None:
        assert content_key(b"a", namespace="source") != content_key(b"a", namespace="mixed")

    def test_shape(self) -> None:
        key = content_key(b"hello", namespace="audio", extension=".mp3")
        digest = hashlib.sha256(b"hello").hexdigest()
        assert key == f"audio/{digest[:2]}/{digest[2:4]}/{digest}.mp3"

    def test_no_extension(self) -> None:
        assert content_key(b"hello", namespace="source").endswith(
            hashlib.sha256(b"hello").hexdigest()
        )

    def test_rejects_empty_namespace(self) -> None:
        with pytest.raises(ValueError, match="slash-free"):
            content_key(b"x", namespace="")

    def test_rejects_slash_in_namespace(self) -> None:
        with pytest.raises(ValueError, match="slash-free"):
            content_key(b"x", namespace="a/b")

    def test_rejects_extension_without_dot(self) -> None:
        with pytest.raises(ValueError, match="start with a dot"):
            content_key(b"x", namespace="a", extension="mp3")

    def test_deduplication_property(self) -> None:
        # The whole point: same bytes generated twice occupy one object.
        generated_once = content_key(b"identical output", namespace="description")
        generated_again = content_key(b"identical output", namespace="description")
        assert generated_once == generated_again


class TestManifestKey:
    def test_shape(self) -> None:
        assert manifest_key("run-123") == "manifests/run-123.json"

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            manifest_key("")
