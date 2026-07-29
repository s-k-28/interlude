"""In-memory job registry and JobState -> API dict conversion.

Why in-memory. The durable record of a run is the manifest in B2, not this
process's RAM. The registry exists so the API can answer "what is happening
right now" cheaply; anything that must survive a restart is written to storage
by the pipeline, and a restarted process resumes from there. Keeping the
registry deliberately dumb avoids pretending it is a database.

Thread safety. FastAPI runs jobs in ``BackgroundTasks``, which execute on a
worker thread while request handlers read the same objects. Every read and
write of the registry dict, and every serialization of a JobState, happens
under one lock. Serializing under the lock costs microseconds and removes an
entire class of torn-read bugs.

Conversion. :func:`to_summary` and :func:`to_detail` emit exactly the keys in
the frozen API contract. The self-check at the bottom asserts that key set
literally, so a rename here fails at build time rather than in the browser.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from app.pipeline.runner import JobState, JobStatus, Stage, build_manifest

# Exact key sets from the frozen contract. Used by the self-check and by
# tests/test_api.py; not used to build the dicts, so a drift is detectable.
SUMMARY_KEYS = frozenset(
    {
        "job_id",
        "title",
        "status",
        "created_at",
        "gaps_found",
        "segments_described",
        "segments_rendered",
        "duration_ms",
    }
)
DETAIL_KEYS = SUMMARY_KEYS | frozenset(
    {"source", "stages", "segments", "tokens", "manifest_hash"}
)
TOKEN_KEYS = frozenset(
    {
        "calls",
        "prompt_tokens_uncompressed",
        "prompt_tokens_compressed",
        "completion_tokens",
        "tokens_saved",
        "reduction_ratio",
    }
)


_MANIFEST_CACHE: dict[str, object] = {}


def cached_manifest(state):
    """One manifest per job id.

    Manifest.new() timestamps from the clock, so rebuilding yields a different
    canonical_hash every call. The hash is the product's central claim; it must
    be stable for the life of the job.
    """
    key = state.job_id
    if key not in _MANIFEST_CACHE:
        _MANIFEST_CACHE[key] = build_manifest(state)
    return _MANIFEST_CACHE[key]


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(slots=True)
class JobRecord:
    """A JobState plus the API-only metadata the pipeline does not carry."""

    state: JobState
    title: str
    source_url: str
    created_at: str = field(default_factory=_utc_now_iso)

    @property
    def job_id(self) -> str:
        return self.state.job_id


class JobRegistry:
    """Thread-safe map of job_id -> JobRecord, newest-first on listing."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: dict[str, JobRecord] = {}
        self._order: list[str] = []

    # -- mutation ---------------------------------------------------------

    def add(self, state: JobState, *, title: str, source_url: str) -> JobRecord:
        """Register a new job. Raises KeyError on a duplicate job_id."""
        record = JobRecord(state=state, title=title, source_url=source_url)
        with self._lock:
            if record.job_id in self._records:
                raise KeyError(f"job {record.job_id} is already registered")
            self._records[record.job_id] = record
            self._order.append(record.job_id)
        return record

    def set_status(self, job_id: str, status: JobStatus, *, error: str = "") -> None:
        """Update status (and optionally error) under the lock."""
        with self._lock:
            record = self._records.get(job_id)
            if record is None:
                return
            record.state.status = status
            if error:
                record.state.error = error

    # -- reads ------------------------------------------------------------

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._records.get(job_id)

    def __contains__(self, job_id: object) -> bool:
        with self._lock:
            return job_id in self._records

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)

    def summaries(self) -> list[dict[str, Any]]:
        """All jobs as summary dicts, newest first, serialized under the lock."""
        with self._lock:
            records = [self._records[jid] for jid in reversed(self._order)]
            return [to_summary(r) for r in records]

    def detail(
        self, job_id: str, presign: Callable[[str], str] | None = None
    ) -> dict[str, Any] | None:
        with self._lock:
            record = self._records.get(job_id)
            if record is None:
                return None
            return to_detail(record, presign=presign)

    def manifest_payload(self, job_id: str) -> dict[str, Any] | None:
        """The provenance manifest for a job, or None if the job is unknown."""
        with self._lock:
            record = self._records.get(job_id)
            if record is None:
                return None
            manifest = cached_manifest(record.state)
            payload = manifest.to_payload()
            payload["canonical_hash"] = manifest.canonical_hash()
            return payload

    def clear(self) -> None:
        """Drop every job. Test-only convenience."""
        with self._lock:
            self._records.clear()
            self._order.clear()


# --------------------------------------------------------------------------
# Serialization
# --------------------------------------------------------------------------


def total_duration_ms(state: JobState) -> int:
    """Wall-clock work recorded across all stages, in milliseconds."""
    return sum(r.duration_ms for r in state.completed_stages)


def to_summary(record: JobRecord) -> dict[str, Any]:
    """JobState -> JobSummary dict. Keys are frozen; do not add to them."""
    state = record.state
    return {
        "job_id": state.job_id,
        "title": record.title,
        "status": state.status.value,
        "created_at": record.created_at,
        "gaps_found": len(state.gaps),
        "segments_described": len(state.segments),
        "segments_rendered": state.spoken_count,
        "duration_ms": total_duration_ms(state),
    }


def _segment_dicts(
    state: JobState, presign: Callable[[str], str] | None
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for segment in state.segments:
        audio_url: str | None = None
        if segment.audio_key and presign is not None:
            # A presign failure degrades one row to a null URL. It must never
            # take down the whole detail response: the text description is
            # still useful to a screen-reader user without the audio link.
            try:
                audio_url = presign(segment.audio_key)
            except Exception:  # noqa: BLE001 - deliberately swallowed
                audio_url = None
        out.append(
            {
                "start": segment.gap.start,
                "end": segment.gap.end,
                "duration": segment.gap.duration,
                "word_budget": segment.gap.word_budget,
                "text": segment.text,
                "accepted": segment.accepted,
                "attempts": segment.attempts,
                "audio_key": segment.audio_key,
                "audio_url": audio_url,
            }
        )
    return out


def _token_dict(state: JobState) -> dict[str, Any]:
    summary = state.ledger.summary()
    return {
        "calls": int(summary["calls"]),
        "prompt_tokens_uncompressed": int(summary["prompt_tokens_uncompressed"]),
        "prompt_tokens_compressed": int(summary["prompt_tokens_compressed"]),
        "completion_tokens": int(summary["completion_tokens"]),
        "tokens_saved": int(summary["tokens_saved"]),
        "reduction_ratio": float(summary["reduction_ratio"]),
    }


def _manifest_hash(state: JobState) -> str:
    """Hash of this run's manifest, or "" if it cannot be built yet.

    A job that has not reached the manifest stage still has a valid partial
    manifest, so this normally succeeds. It is wrapped anyway because a 500 on
    the detail endpoint would hide the very state an operator is inspecting.
    """
    try:
        return cached_manifest(state).canonical_hash()
    except Exception:  # noqa: BLE001
        return ""


def to_detail(
    record: JobRecord, *, presign: Callable[[str], str] | None = None
) -> dict[str, Any]:
    """JobState -> JobDetail dict. Superset of the summary, keys frozen."""
    state = record.state
    detail = to_summary(record)
    detail["source"] = {"key": state.source_key, "sha256": state.source_sha256}
    detail["stages"] = [
        {
            "stage": r.stage.value,
            "ok": r.ok,
            "duration_ms": r.duration_ms,
            "detail": r.detail,
        }
        for r in sorted(state.completed_stages, key=lambda r: r.stage.index())
    ]
    detail["segments"] = _segment_dicts(state, presign)
    detail["tokens"] = _token_dict(state)
    detail["manifest_hash"] = _manifest_hash(state)
    return detail


def statuses_of(records: Iterable[JobRecord]) -> list[str]:
    """Small helper used by the resume endpoint's logging."""
    return [r.state.status.value for r in records]


if __name__ == "__main__":
    import sys

    from app.pipeline.gaps import Gap
    from app.pipeline.runner import DescribedSegment, TranscriptInput

    registry = JobRegistry()

    state = JobState.new("source/ab/cd/" + "ab" * 32 + ".mp4", "ab" * 32)
    assert len(state.job_id) == 12, state.job_id
    assert state.status is JobStatus.PENDING

    record = registry.add(state, title="Lecture 101", source_url="https://x.test/a.mp4")
    assert registry.get(state.job_id) is record
    assert registry.get("nope") is None
    assert state.job_id in registry
    assert len(registry) == 1

    # duplicate registration is refused
    try:
        registry.add(state, title="dup", source_url="https://x.test/a.mp4")
    except KeyError:
        pass
    else:  # pragma: no cover
        print("FAIL: duplicate job_id should raise", file=sys.stderr)
        raise SystemExit(1)

    # -- empty job serializes cleanly -------------------------------------
    empty_summary = registry.summaries()[0]
    assert set(empty_summary) == set(SUMMARY_KEYS), set(empty_summary)
    assert empty_summary["status"] == "pending"
    assert empty_summary["gaps_found"] == 0
    assert empty_summary["duration_ms"] == 0

    # -- populate a realistic partial run ---------------------------------
    state.transcript = TranscriptInput(text="hello world", word_spans=[(0.0, 0.5)], duration=60.0)
    state.gaps = [Gap(0.5, 6.0), Gap(20.0, 23.0)]
    state.segments = [
        DescribedSegment(
            gap=Gap(0.5, 6.0),
            text="A lecturer stands beside a projected diagram of a cell.",
            accepted=True,
            attempts=1,
            audio_key="description-audio/aa/bb/aabb.mp3",
            audio_sha256="cd" * 32,
        ),
        DescribedSegment(
            gap=Gap(20.0, 23.0),
            text="The slide changes to a bar chart.",
            accepted=True,
            attempts=2,
        ),
    ]
    state.ledger.record(uncompressed_tokens=1000, compressed_tokens=400, completion_tokens=60)
    state.mark(Stage.TRANSCRIBE, ok=True, duration_ms=900, detail="1 words")
    state.mark(Stage.DETECT_GAPS, ok=True, duration_ms=3, detail="2 gaps")
    state.mark(Stage.DESCRIBE, ok=True, duration_ms=4100, detail="2/2 within budget")
    state.status = JobStatus.PARTIAL

    summary = registry.summaries()[0]
    assert set(summary) == set(SUMMARY_KEYS)
    assert summary["job_id"] == state.job_id
    assert summary["title"] == "Lecture 101"
    assert summary["status"] == "partial"
    assert summary["gaps_found"] == 2
    assert summary["segments_described"] == 2
    assert summary["segments_rendered"] == 1
    assert summary["duration_ms"] == 900 + 3 + 4100, summary["duration_ms"]
    assert summary["created_at"].endswith("+00:00"), summary["created_at"]

    # -- detail, with a presigner that works ------------------------------
    detail = registry.detail(state.job_id, presign=lambda key: f"https://signed.test/{key}")
    assert detail is not None
    assert set(detail) == set(DETAIL_KEYS), set(detail) ^ set(DETAIL_KEYS)
    assert set(detail["source"]) == {"key", "sha256"}
    assert len(detail["stages"]) == 3
    assert [s["stage"] for s in detail["stages"]] == ["transcribe", "detect_gaps", "describe"]
    assert set(detail["stages"][0]) == {"stage", "ok", "duration_ms", "detail"}
    assert len(detail["segments"]) == 2
    assert set(detail["segments"][0]) == {
        "start", "end", "duration", "word_budget", "text",
        "accepted", "attempts", "audio_key", "audio_url",
    }
    assert detail["segments"][0]["audio_url"] == (
        "https://signed.test/description-audio/aa/bb/aabb.mp3"
    )
    assert detail["segments"][1]["audio_url"] is None
    assert detail["segments"][1]["audio_key"] == ""
    assert abs(detail["segments"][0]["duration"] - 5.5) < 1e-9
    assert detail["segments"][0]["word_budget"] == Gap(0.5, 6.0).word_budget
    assert set(detail["tokens"]) == set(TOKEN_KEYS), set(detail["tokens"])
    assert detail["tokens"]["calls"] == 1
    assert detail["tokens"]["tokens_saved"] == 600
    assert len(detail["manifest_hash"]) == 64, detail["manifest_hash"]

    # -- a broken presigner must not break the response -------------------
    def _boom(_key: str) -> str:
        raise RuntimeError("no credentials")

    degraded = registry.detail(state.job_id, presign=_boom)
    assert degraded is not None
    assert degraded["segments"][0]["audio_url"] is None
    assert degraded["segments"][0]["audio_key"] == "description-audio/aa/bb/aabb.mp3"

    # -- no presigner at all ----------------------------------------------
    bare = registry.detail(state.job_id)
    assert bare is not None
    assert bare["segments"][0]["audio_url"] is None

    assert registry.detail("nope") is None

    # -- manifest ----------------------------------------------------------
    payload = registry.manifest_payload(state.job_id)
    assert payload is not None
    assert payload["run_id"] == state.job_id
    assert payload["canonical_hash"] == detail["manifest_hash"]
    assert payload["metadata"]["segments_rendered"] == 1
    assert registry.manifest_payload("nope") is None

    # -- status mutation ---------------------------------------------------
    registry.set_status(state.job_id, JobStatus.FAILED, error="provider timeout")
    assert registry.get(state.job_id).state.status is JobStatus.FAILED  # type: ignore[union-attr]
    assert state.error == "provider timeout"
    registry.set_status("nope", JobStatus.FAILED)  # must be a no-op, not a raise

    # -- ordering: newest first --------------------------------------------
    second = JobState.new("source/ff/ee/x.mp4", "ff" * 32)
    registry.add(second, title="Lecture 102", source_url="https://x.test/b.mp4")
    ordered = registry.summaries()
    assert [s["job_id"] for s in ordered] == [second.job_id, state.job_id]

    assert statuses_of([registry.get(second.job_id)]) == ["pending"]  # type: ignore[list-item]

    registry.clear()
    assert len(registry) == 0
    assert registry.summaries() == []

    print("app.api.jobs self-check OK")
