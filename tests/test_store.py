"""Tests for the B2 storage layer, against an in-memory fake."""

from __future__ import annotations

import pytest

from app.config import B2Config
from app.storage.errors import StorageError
from app.storage.keys import content_key
from app.storage.store import B2Store
from tests.fakes import FakeClientError, FakeS3Client


@pytest.fixture
def config() -> B2Config:
    return B2Config(
        key_id="k",
        app_key="s",
        bucket="test-bucket",
        endpoint="https://s3.us-west-004.backblazeb2.com",
    )


@pytest.fixture
def client() -> FakeS3Client:
    return FakeS3Client()


@pytest.fixture
def store(config: B2Config, client: FakeS3Client) -> B2Store:
    return B2Store(config, client=client)


class TestExists:
    def test_false_for_absent_key(self, store: B2Store) -> None:
        assert not store.exists("nope")

    def test_true_after_put(self, store: B2Store) -> None:
        stored = store.put(b"data", namespace="source")
        assert store.exists(stored.key)

    def test_access_denied_raises_rather_than_returning_false(
        self, store: B2Store, client: FakeS3Client
    ) -> None:
        # Treating a permissions failure as "absent" would cause the pipeline
        # to silently regenerate assets it simply cannot read.
        def denied(**_: object) -> None:
            raise FakeClientError("AccessDenied")

        client.head_object = denied  # type: ignore[assignment]
        with pytest.raises(StorageError, match="AccessDenied"):
            store.exists("k")

    def test_non_s3_exception_propagates(
        self, store: B2Store, client: FakeS3Client
    ) -> None:
        def boom(**_: object) -> None:
            raise TimeoutError("network down")

        client.head_object = boom  # type: ignore[assignment]
        with pytest.raises(TimeoutError):
            store.exists("k")


class TestPut:
    def test_returns_content_addressed_key(self, store: B2Store) -> None:
        stored = store.put(b"hello", namespace="source", extension=".mp4")
        assert stored.key == content_key(b"hello", namespace="source", extension=".mp4")

    def test_records_size_and_digest(self, store: B2Store) -> None:
        stored = store.put(b"hello", namespace="source")
        assert stored.size == 5
        assert len(stored.sha256) == 64

    def test_uri_property(self, store: B2Store) -> None:
        stored = store.put(b"hello", namespace="source")
        assert stored.uri == f"b2://{stored.key}"

    def test_deduplicates_identical_content(
        self, store: B2Store, client: FakeS3Client
    ) -> None:
        first = store.put(b"same", namespace="source")
        second = store.put(b"same", namespace="source")
        assert first.key == second.key
        assert not first.deduplicated
        assert second.deduplicated
        assert client.put_calls == 1  # the second upload never happened

    def test_digest_stored_in_metadata(
        self, store: B2Store, client: FakeS3Client
    ) -> None:
        stored = store.put(b"hello", namespace="source")
        assert client.objects[stored.key]["Metadata"]["sha256"] == stored.sha256

    def test_custom_metadata_merged(self, store: B2Store, client: FakeS3Client) -> None:
        stored = store.put(b"x", namespace="source", metadata={"run_id": "r1"})
        meta = client.objects[stored.key]["Metadata"]
        assert meta["run_id"] == "r1"
        assert "sha256" in meta

    def test_failure_wrapped_in_storage_error(
        self, store: B2Store, client: FakeS3Client
    ) -> None:
        def fail(**_: object) -> None:
            raise FakeClientError("InternalError")

        client.put_object = fail  # type: ignore[assignment]
        with pytest.raises(StorageError, match="put_object failed"):
            store.put(b"x", namespace="source")


class TestGet:
    def test_round_trip(self, store: B2Store) -> None:
        stored = store.put(b"payload", namespace="source")
        assert store.get(stored.key) == b"payload"

    def test_missing_key_raises(self, store: B2Store) -> None:
        with pytest.raises(StorageError, match="get_object failed"):
            store.get("absent")

    def test_detects_corruption(self, store: B2Store, client: FakeS3Client) -> None:
        # Bytes changed underneath us without the digest being updated.
        stored = store.put(b"original", namespace="source")
        client.corrupt(stored.key, b"tampered")
        with pytest.raises(StorageError, match="integrity check failed"):
            store.get(stored.key)


class TestPutText:
    def test_encodes_utf8(self, store: B2Store) -> None:
        stored = store.put_text("café", namespace="transcript")
        assert store.get(stored.key).decode("utf-8") == "café"


class TestPutAtKey:
    def test_uses_explicit_key(self, store: B2Store) -> None:
        stored = store.put_at_key('{"a":1}', key="manifests/run-1.json")
        assert stored.key == "manifests/run-1.json"

    def test_content_retrievable(self, store: B2Store) -> None:
        store.put_at_key('{"a":1}', key="manifests/run-1.json")
        assert store.get("manifests/run-1.json") == b'{"a":1}'


class TestPresignedUrl:
    def test_returns_url(self, store: B2Store) -> None:
        stored = store.put(b"x", namespace="source")
        assert stored.key in store.presigned_url(stored.key)

    def test_rejects_nonpositive_expiry(self, store: B2Store) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            store.presigned_url("k", expires_in=0)
