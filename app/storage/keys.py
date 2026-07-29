"""Content-addressed object keys.

Every artifact Interlude produces is stored under a key derived from the SHA-256
of its bytes. Two consequences:

1. Identical content is never stored twice, however many times it is generated.
2. The key is itself a verifiable claim about the content, which is what makes
   the provenance manifest auditable rather than merely descriptive.
"""

from __future__ import annotations

import hashlib
from typing import Final

# Two-level fan-out on the first four hex characters keeps any single B2 prefix
# from accumulating an unbounded number of objects.
_FANOUT: Final = 2


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def content_key(
    data: bytes,
    *,
    namespace: str,
    extension: str = "",
) -> str:
    """Return the canonical storage key for a blob.

    Args:
        data: The bytes to be stored.
        namespace: Logical grouping, e.g. ``"source"``, ``"description"``,
            ``"mixed"``. Must not be empty or contain slashes.
        extension: Optional suffix including the dot, e.g. ``".mp3"``.

    Returns:
        A key of the form ``namespace/ab/cd/<sha256><extension>``.
    """
    if not namespace or "/" in namespace:
        raise ValueError(f"namespace must be non-empty and slash-free, got {namespace!r}")
    if extension and not extension.startswith("."):
        raise ValueError(f"extension must start with a dot, got {extension!r}")

    digest = sha256_hex(data)
    shard_a = digest[:_FANOUT]
    shard_b = digest[_FANOUT : _FANOUT * 2]
    return f"{namespace}/{shard_a}/{shard_b}/{digest}{extension}"


def manifest_key(run_id: str) -> str:
    """Manifests are addressed by run, not by content.

    A manifest must be locatable before its content hash is known, so it is the
    one artifact keyed by identity rather than by bytes.
    """
    if not run_id:
        raise ValueError("run_id must not be empty")
    return f"manifests/{run_id}.json"
