"""Filesystem-backed store with the same contract as B2Store.

Interlude's storage layer is injected everywhere, so the pipeline does not care
whether artifacts land in Backblaze or on disk. This implementation exists for
two honest reasons:

1. The full pipeline — including the provenance chain — can be demonstrated and
   filmed before cloud credentials are in place.
2. Every test that needs a store gets a real one, not a mock, so the code paths
   exercised offline are the same paths that run in production.

Swapping back is a single line at the composition root:

    store = B2Store(settings.b2)      # production
    store = LocalStore("./artifacts") # offline

Keys are byte-identical between the two. An artifact written here and later
uploaded to B2 keeps the same key and the same hash, so a provenance chain built
offline stays valid after migration.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from app.storage.errors import StorageError
from app.storage.keys import content_key, sha256_hex

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class StoredObject:
    """Mirrors app.storage.store.StoredObject exactly."""

    key: str
    sha256: str
    size: int
    deduplicated: bool

    @property
    def uri(self) -> str:
        return f"file://{self.key}"


class LocalStore:
    """Content-addressed storage on the local filesystem."""

    def __init__(self, root: str | Path = "./artifacts") -> None:
        self._root = Path(root).resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        self._meta = self._root / ".metadata"
        self._meta.mkdir(exist_ok=True)

    @property
    def bucket(self) -> str:
        return str(self._root)

    def _path(self, key: str) -> Path:
        # Reject traversal before touching the filesystem. A key is generated
        # internally, but this store also reads keys back from manifests.
        if key.startswith("/") or ".." in key.split("/"):
            raise StorageError(f"unsafe key: {key}")
        return self._root / key

    def _meta_path(self, key: str) -> Path:
        return self._meta / (key.replace("/", "_") + ".json")

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def put(
        self,
        data: bytes,
        *,
        namespace: str,
        extension: str = "",
        content_type: str = "application/octet-stream",
        metadata: dict[str, str] | None = None,
    ) -> StoredObject:
        key = content_key(data, namespace=namespace, extension=extension)
        digest = sha256_hex(data)

        if self.exists(key):
            logger.debug("dedup hit %s", key)
            return StoredObject(key=key, sha256=digest, size=len(data), deduplicated=True)

        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a sibling temp file and rename. A crash mid-write must never
        # leave a truncated object under a hash that promises full content.
        tmp = path.with_suffix(path.suffix + ".tmp")
        try:
            tmp.write_bytes(data)
            tmp.replace(path)
        except OSError as exc:
            tmp.unlink(missing_ok=True)
            raise StorageError(f"write failed for {key}: {exc}") from exc

        self._meta_path(key).write_text(
            json.dumps({"sha256": digest, "content_type": content_type, **(metadata or {})},
                       sort_keys=True),
            encoding="utf-8",
        )
        logger.info("stored %s (%d bytes)", key, len(data))
        return StoredObject(key=key, sha256=digest, size=len(data), deduplicated=False)

    def get(self, key: str) -> bytes:
        path = self._path(key)
        if not path.is_file():
            raise StorageError(f"not found: {key}")
        data = path.read_bytes()

        meta_file = self._meta_path(key)
        if meta_file.is_file():
            expected = json.loads(meta_file.read_text()).get("sha256")
            if expected and sha256_hex(data) != expected:
                raise StorageError(f"integrity check failed for {key}")
        return data

    def put_text(self, text: str, *, namespace: str, extension: str = ".txt") -> StoredObject:
        return self.put(text.encode("utf-8"), namespace=namespace, extension=extension,
                        content_type="text/plain; charset=utf-8")

    def put_at_key(self, payload: str, *, key: str) -> StoredObject:
        data = payload.encode("utf-8")
        digest = sha256_hex(data)
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        self._meta_path(key).write_text(
            json.dumps({"sha256": digest, "content_type": "application/json"}, sort_keys=True),
            encoding="utf-8",
        )
        return StoredObject(key=key, sha256=digest, size=len(data), deduplicated=False)

    def presigned_url(self, key: str, *, expires_in: int = 3600) -> str:
        """No signing offline — returns a path the local server can serve."""
        if expires_in <= 0:
            raise ValueError(f"expires_in must be positive, got {expires_in}")
        return f"/artifacts/{key}"

    def list_keys(self, prefix: str = "") -> list[str]:
        """Every stored key, optionally filtered. Used to rebuild a job index."""
        out: list[str] = []
        for p in self._root.rglob("*"):
            if p.is_file() and ".metadata" not in p.parts and not p.name.endswith(".tmp"):
                rel = str(p.relative_to(self._root))
                if rel.startswith(prefix):
                    out.append(rel)
        return sorted(out)

    def clear(self) -> None:
        """Wipe everything. Test helper only."""
        shutil.rmtree(self._root, ignore_errors=True)
        self._root.mkdir(parents=True, exist_ok=True)
        self._meta.mkdir(exist_ok=True)


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        s = LocalStore(tmp)
        probe = b"interlude offline probe"

        o = s.put(probe, namespace="source", extension=".txt")
        assert o.key.startswith("source/")
        assert not o.deduplicated
        assert s.get(o.key) == probe

        again = s.put(probe, namespace="source", extension=".txt")
        assert again.deduplicated, "identical bytes must not be rewritten"
        assert again.key == o.key

        assert s.put_text("café", namespace="transcript") is not None
        m = s.put_at_key('{"a":1}', key="manifests/run-1.json")
        assert s.get(m.key) == b'{"a":1}'

        try:
            s.get("does/not/exist")
            raise AssertionError("missing key must raise")
        except StorageError:
            pass

        for bad in ("/etc/passwd", "../../escape"):
            try:
                s.put(b"x", namespace="source") and s.get(bad)
                raise AssertionError(f"traversal not blocked: {bad}")
            except StorageError:
                pass

        # Corrupt the bytes without updating the digest; get() must notice.
        (Path(tmp) / o.key).write_bytes(b"tampered")
        try:
            s.get(o.key)
            raise AssertionError("corruption must be detected")
        except StorageError:
            pass

        assert len(s.list_keys("source/")) >= 1
        print("local store self-check OK")
