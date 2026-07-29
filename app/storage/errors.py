"""Duck-typed S3 error inspection.

The storage layer must be testable against an in-memory fake with no
credentials and no botocore import. Rather than isinstance-checking
``botocore.exceptions.ClientError``, we inspect the ``response`` attribute that
botocore-shaped exceptions carry.
"""

from __future__ import annotations

# Codes S3 returns when an object simply is not there. Never an error for us.
NOT_FOUND_CODES = frozenset({"404", "NoSuchKey", "NotFound"})

# Transient conditions worth retrying. B2 returns 503 under sustained load.
RETRYABLE_CODES = frozenset(
    {"500", "502", "503", "504", "SlowDown", "RequestTimeout", "InternalError"}
)


class StorageError(RuntimeError):
    """Raised when an object store operation fails unrecoverably."""


def error_code(exc: BaseException) -> str | None:
    """Extract an S3 error code from a botocore-shaped exception.

    Returns ``None`` when the exception is not botocore-shaped, which signals
    the caller to re-raise rather than swallow an unrelated failure.
    """
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return None
    error = response.get("Error")
    if not isinstance(error, dict):
        return None
    return str(error.get("Code", ""))


def is_not_found(exc: BaseException) -> bool:
    return error_code(exc) in NOT_FOUND_CODES


def is_retryable(exc: BaseException) -> bool:
    return error_code(exc) in RETRYABLE_CODES
