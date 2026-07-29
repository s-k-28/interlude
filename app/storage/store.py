"""Backblaze B2 object storage via the S3-compatible API.

Design notes:

* Path-style addressing. B2's S3 endpoints do not reliably support virtual-host
  style for every bucket name, so ``addressing_style`` is pinned to ``"path"``.
* Idempotent writes. Keys are content-addressed, so a put whose key already
  exists is a no-op; identical bytes are never uploaded or billed twice.
* Retries are delegated to botocore's adaptive mode, which backs off correctly
  under B2's rate limiting.
* Errors are duck-typed via :mod:`app.storage.errors` rather than
  isinstance-checked against botocore, so the whole layer is exercisable
  against an in-memory fake with no credentials and no network.

Supersedes the earlier ``b2.py``; ``bootstrap.sh`` installs this as the storage
module of record.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from app.config import B2Config
from app.storage.errors import StorageError, error_code, is_not_found
from app.storage.keys import content_key, sha256_hex

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class StoredObject:
    """The result of persisting a blob."""

    key: str
    sha256: str
    size: int
    deduplicated: bool

    @property
    def uri(self) -> str:
        return f"b2://{self.key}"


class B2Store:
    """Content-addressed object storage on Backblaze B2."""

    def __init__(self, config: B2Config, client: Any | None = None) -> None:
        self._config = config
        self._client = client if client is not None else self._build_client(config)

    @staticmethod
    def _build_client(config: B2Config) -> Any:
        import boto3
        from botocore.config import Config as BotoConfig

        return boto3.client(
            "s3",
            endpoint_url=config.endpoint,
            aws_access_key_id=config.key_id,
            aws_secret_access_key=config.app_key,
            region_name=config.region,
            config=BotoConfig(
                signature_version="s3v4",
                s3={"addressing_style": "path"},
                retries={"max_attempts": 5, "mode": "adaptive"},
                connect_timeout=15,
                read_timeout=120,
            ),
        )

    @property
    def bucket(self) -> str:
        return self._config.bucket

    def exists(self, key: str) -> bool:
        """True when an object is already present under ``key``."""
        try:
            self._client.head_object(Bucket=self.bucket, Key=key)
        except Exception as exc:  # noqa: BLE001 - botocore errors are duck-typed
            if error_code(exc) is None:
                raise  # not an S3 error; never swallow it
            if is_not_found(exc):
                return False
            raise StorageError(f"head_object failed for {key}: {error_code(exc)}") from exc
        return True

    def put(
        self,
        data: bytes,
        *,
        namespace: str,
        extension: str = "",
        content_type: str = "application/octet-stream",
        metadata: dict[str, str] | None = None,
    ) -> StoredObject:
        """Store ``data`` under its content-addressed key.

        When the content is already present the upload is skipped and
        ``deduplicated=True`` is returned.
        """
        key = content_key(data, namespace=namespace, extension=extension)
        digest = sha256_hex(data)

        if self.exists(key):
            logger.debug("dedup hit for %s", key)
            return StoredObject(key=key, sha256=digest, size=len(data), deduplicated=True)

        try:
            self._client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=data,
                ContentType=content_type,
                # Round-trip the digest so integrity is re-verifiable later
                # without re-downloading and re-hashing the payload.
                Metadata={"sha256": digest, **(metadata or {})},
            )
        except Exception as exc:  # noqa: BLE001
            raise StorageError(f"put_object failed for {key}: {exc}") from exc

        logger.info("stored %s (%d bytes)", key, len(data))
        return StoredObject(key=key, sha256=digest, size=len(data), deduplicated=False)

    def get(self, key: str) -> bytes:
        """Fetch an object, verifying integrity against the stored digest."""
        try:
            response = self._client.get_object(Bucket=self.bucket, Key=key)
            body: bytes = response["Body"].read()
        except Exception as exc:  # noqa: BLE001
            raise StorageError(f"get_object failed for {key}: {exc}") from exc

        expected = response.get("Metadata", {}).get("sha256")
        if expected:
            actual = sha256_hex(body)
            if actual != expected:
                raise StorageError(
                    f"integrity check failed for {key}: expected {expected}, got {actual}"
                )
        return body

    def put_text(self, text: str, *, namespace: str, extension: str = ".txt") -> StoredObject:
        return self.put(
            text.encode("utf-8"),
            namespace=namespace,
            extension=extension,
            content_type="text/plain; charset=utf-8",
        )

    def put_at_key(self, payload: str, *, key: str) -> StoredObject:
        """Store a document at an explicit key rather than a content hash.

        Manifests must be locatable by run id before their content hash is
        known, so they are the one artifact addressed by identity.
        """
        data = payload.encode("utf-8")
        digest = sha256_hex(data)
        try:
            self._client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=data,
                ContentType="application/json",
                Metadata={"sha256": digest},
            )
        except Exception as exc:  # noqa: BLE001
            raise StorageError(f"put_object failed for {key}: {exc}") from exc
        return StoredObject(key=key, sha256=digest, size=len(data), deduplicated=False)

    def presigned_url(self, key: str, *, expires_in: int = 3600) -> str:
        """Time-limited read URL, used to serve assets to the review UI."""
        if expires_in <= 0:
            raise ValueError(f"expires_in must be positive, got {expires_in}")
        return self._client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=expires_in,
        )
