"""Pydantic v2 models for the Interlude HTTP API.

These models are the written-down form of the frozen API contract. Every key
here appears in the contract; nothing here is invented. The models exist so the
shape is enforced at the boundary rather than trusted, and so an accidental key
rename in the conversion layer fails loudly instead of silently shipping a
response the frontend cannot read.

All times are SECONDS. ``created_at`` is ISO-8601 UTC.
"""

from __future__ import annotations

from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator

# The five job states the pipeline can be in. Kept as a literal tuple rather
# than importing JobStatus so the schema module stays importable with no
# pipeline dependencies (and so /health works on a broken install).
JOB_STATUSES = ("pending", "running", "partial", "complete", "failed")


class StrictModel(BaseModel):
    """Base: reject unknown fields so a typo is a 422, not a silent drop."""

    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------
# GET /health
# --------------------------------------------------------------------------


class HealthResponse(StrictModel):
    status: str
    version: str
    providers_missing: list[str] = Field(default_factory=list)
    b2_ok: bool


# --------------------------------------------------------------------------
# POST /jobs
# --------------------------------------------------------------------------


class CreateJobRequest(StrictModel):
    source_url: str
    title: str | None = None

    @field_validator("source_url")
    @classmethod
    def _validate_source_url(cls, value: str) -> str:
        candidate = value.strip()
        if not candidate:
            raise ValueError("source_url must not be empty")
        parsed = urlparse(candidate)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("source_url must be an http:// or https:// URL")
        if not parsed.netloc:
            raise ValueError("source_url must include a host")
        return candidate

    @field_validator("title")
    @classmethod
    def _trim_title(cls, value: str | None) -> str | None:
        if value is None:
            return None
        trimmed = value.strip()
        return trimmed or None


class CreateJobResponse(StrictModel):
    job_id: str
    status: str


# --------------------------------------------------------------------------
# POST /jobs/{job_id}/resume
# --------------------------------------------------------------------------


class ResumeJobResponse(StrictModel):
    job_id: str
    status: str
    resumed_from: str


# --------------------------------------------------------------------------
# GET /jobs, GET /jobs/{job_id}
# --------------------------------------------------------------------------


class JobSummary(StrictModel):
    job_id: str
    title: str
    status: str
    created_at: str
    gaps_found: int
    segments_described: int
    segments_rendered: int
    duration_ms: int


class SourceRef(StrictModel):
    key: str
    sha256: str


class StageEntry(StrictModel):
    stage: str
    ok: bool
    duration_ms: int
    detail: str


class SegmentEntry(StrictModel):
    start: float
    end: float
    duration: float
    word_budget: int
    text: str
    accepted: bool
    attempts: int
    audio_key: str
    audio_url: str | None = None


class TokenSummary(StrictModel):
    calls: int
    prompt_tokens_uncompressed: int
    prompt_tokens_compressed: int
    completion_tokens: int
    tokens_saved: int
    reduction_ratio: float


class JobDetail(JobSummary):
    source: SourceRef
    stages: list[StageEntry] = Field(default_factory=list)
    segments: list[SegmentEntry] = Field(default_factory=list)
    tokens: TokenSummary
    manifest_hash: str


class JobListResponse(StrictModel):
    jobs: list[JobSummary] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


class ErrorResponse(StrictModel):
    error: str
    detail: str = ""


if __name__ == "__main__":
    import sys

    from pydantic import ValidationError

    # --- one valid instance of every model -------------------------------
    health = HealthResponse(
        status="ok", version="1.0.0", providers_missing=["GOOGLE_API_KEY"], b2_ok=False
    )
    assert health.b2_ok is False
    assert health.model_dump() == {
        "status": "ok",
        "version": "1.0.0",
        "providers_missing": ["GOOGLE_API_KEY"],
        "b2_ok": False,
    }

    req = CreateJobRequest(source_url="  https://media.example.edu/lec-101.mp4  ",
                           title="  Lecture 101  ")
    assert req.source_url == "https://media.example.edu/lec-101.mp4"
    assert req.title == "Lecture 101"
    assert CreateJobRequest(source_url="http://x.test/a.mp4").title is None

    created = CreateJobResponse(job_id="a1b2c3d4e5f6", status="pending")
    assert created.model_dump() == {"job_id": "a1b2c3d4e5f6", "status": "pending"}

    resumed = ResumeJobResponse(
        job_id="a1b2c3d4e5f6", status="running", resumed_from="describe"
    )
    assert resumed.resumed_from == "describe"

    summary = JobSummary(
        job_id="a1b2c3d4e5f6",
        title="Lecture 101",
        status="partial",
        created_at="2026-07-29T03:12:24+00:00",
        gaps_found=20,
        segments_described=20,
        segments_rendered=18,
        duration_ms=41230,
    )
    assert set(summary.model_dump()) == {
        "job_id", "title", "status", "created_at",
        "gaps_found", "segments_described", "segments_rendered", "duration_ms",
    }

    detail = JobDetail(
        **summary.model_dump(),
        source=SourceRef(key="source/ab/cd/abcd.mp4", sha256="ab" * 32),
        stages=[StageEntry(stage="transcribe", ok=True, duration_ms=900, detail="812 words")],
        segments=[
            SegmentEntry(
                start=12.0,
                end=15.5,
                duration=3.5,
                word_budget=9,
                text="The lecturer points at a labelled diagram.",
                accepted=True,
                attempts=1,
                audio_key="description-audio/ab/cd/abcd.mp3",
                audio_url="https://s3.us-west-004.backblazeb2.com/signed",
            ),
            SegmentEntry(
                start=40.0,
                end=42.0,
                duration=2.0,
                word_budget=5,
                text="A new slide appears.",
                accepted=True,
                attempts=2,
                audio_key="",
                audio_url=None,
            ),
        ],
        tokens=TokenSummary(
            calls=20,
            prompt_tokens_uncompressed=40000,
            prompt_tokens_compressed=12000,
            completion_tokens=1800,
            tokens_saved=28000,
            reduction_ratio=0.7,
        ),
        manifest_hash="f" * 64,
    )
    detail_keys = set(detail.model_dump())
    assert detail_keys == {
        "job_id", "title", "status", "created_at",
        "gaps_found", "segments_described", "segments_rendered", "duration_ms",
        "source", "stages", "segments", "tokens", "manifest_hash",
    }, detail_keys
    assert detail.segments[1].audio_url is None

    listing = JobListResponse(jobs=[summary])
    assert listing.model_dump()["jobs"][0]["job_id"] == "a1b2c3d4e5f6"

    err = ErrorResponse(error="job_not_found", detail="no job with id zzz")
    assert err.error == "job_not_found"

    # --- invalid inputs are rejected -------------------------------------
    for bad in ("", "   ", "ftp://example.com/a.mp4", "not-a-url", "https://"):
        try:
            CreateJobRequest(source_url=bad)
        except ValidationError:
            pass
        else:  # pragma: no cover - assertion path
            print(f"FAIL: source_url {bad!r} should have been rejected", file=sys.stderr)
            raise SystemExit(1)

    try:
        CreateJobRequest(source_url="https://x.test/a.mp4", nonsense=1)  # type: ignore[call-arg]
    except ValidationError:
        pass
    else:  # pragma: no cover
        print("FAIL: extra field should have been rejected", file=sys.stderr)
        raise SystemExit(1)

    try:
        HealthResponse(status="ok", version="1", b2_ok="maybe")  # type: ignore[arg-type]
    except ValidationError:
        pass
    else:  # pragma: no cover
        print("FAIL: non-boolean b2_ok should have been rejected", file=sys.stderr)
        raise SystemExit(1)

    print("app.api.schemas self-check OK")
