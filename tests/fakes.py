"""In-memory test doubles.

The B2 fake implements enough of the boto3 S3 surface for the storage layer's
tests to exercise real code paths — including the ClientError shapes botocore
raises — without a network or credentials.
"""

from __future__ import annotations

from typing import Any


class FakeClientError(Exception):
    """Mirrors botocore.exceptions.ClientError."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


class _Body:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data


class FakeS3Client:
    """Minimal in-memory S3."""

    def __init__(self) -> None:
        self.objects: dict[str, dict[str, Any]] = {}
        self.put_calls = 0
        self.head_calls = 0

    def head_object(self, Bucket: str, Key: str) -> dict[str, Any]:  # noqa: N803
        self.head_calls += 1
        if Key not in self.objects:
            raise FakeClientError("404")
        return {"ContentLength": len(self.objects[Key]["Body"])}

    def put_object(  # noqa: N803
        self,
        Bucket: str,
        Key: str,
        Body: bytes,
        ContentType: str = "",
        Metadata: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        self.put_calls += 1
        self.objects[Key] = {
            "Body": Body,
            "ContentType": ContentType,
            "Metadata": Metadata or {},
        }
        return {}

    def get_object(self, Bucket: str, Key: str) -> dict[str, Any]:  # noqa: N803
        if Key not in self.objects:
            raise FakeClientError("NoSuchKey")
        stored = self.objects[Key]
        return {"Body": _Body(stored["Body"]), "Metadata": stored["Metadata"]}

    def generate_presigned_url(
        self, operation: str, Params: dict[str, str], ExpiresIn: int  # noqa: N803
    ) -> str:
        return f"https://fake.b2/{Params['Key']}?exp={ExpiresIn}"

    def corrupt(self, key: str, new_body: bytes) -> None:
        """Simulate bit-rot: change bytes without updating the stored digest."""
        self.objects[key]["Body"] = new_body
