"""Job orchestration.

Wires the stages together and makes a run resumable:

    transcribe -> find gaps -> describe each gap -> synthesize -> manifest

Two properties matter more than the happy path.

**Resumability.** A library job is thousands of videos and hours of wall clock.
If it dies at video 340 of 400 it must restart at 340, not at zero. Every stage
therefore records its output before the next begins, and a run can be rebuilt
from its recorded stages.

**Partial success is a real outcome.** A video where 18 of 20 descriptions fit
their gaps is still useful and still shippable. A single failed gap must never
discard the other nineteen.

No SDK is imported here. Providers arrive as injected callables, so the whole
orchestrator is exercisable offline.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

from app.pipeline.describe import DescriptionResult, draft_description
from app.pipeline.gaps import Gap, find_gaps
from app.pipeline.manifest import AssetRecord, Manifest, StepRecord, TokenUsage
from app.pipeline.tokens import TokenLedger

logger = logging.getLogger(__name__)


class Stage(str, Enum):
    """Ordered pipeline stages. Order defines resume points."""

    TRANSCRIBE = "transcribe"
    DETECT_GAPS = "detect_gaps"
    DESCRIBE = "describe"
    SYNTHESIZE = "synthesize"
    MANIFEST = "manifest"

    @classmethod
    def ordered(cls) -> list[Stage]:
        return [cls.TRANSCRIBE, cls.DETECT_GAPS, cls.DESCRIBE, cls.SYNTHESIZE, cls.MANIFEST]

    def index(self) -> int:
        return Stage.ordered().index(self)


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PARTIAL = "partial"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class TranscriptInput:
    """What the runner needs from transcription, SDK-agnostic."""

    text: str
    word_spans: list[tuple[float, float]]
    duration: float


@dataclass(slots=True)
class DescribedSegment:
    """One gap and the description written for it."""

    gap: Gap
    text: str
    accepted: bool
    attempts: int
    audio_key: str = ""
    audio_sha256: str = ""

    @property
    def spoken(self) -> bool:
        return bool(self.audio_key)


@dataclass(slots=True)
class StageRecord:
    """Outcome of one stage, retained so a run can resume or be audited."""

    stage: Stage
    ok: bool
    duration_ms: int
    detail: str = ""


@dataclass(slots=True)
class JobState:
    """Everything known about a run. Serializable, resumable."""

    job_id: str
    source_key: str
    source_sha256: str
    status: JobStatus = JobStatus.PENDING
    completed_stages: list[StageRecord] = field(default_factory=list)
    transcript: TranscriptInput | None = None
    gaps: list[Gap] = field(default_factory=list)
    segments: list[DescribedSegment] = field(default_factory=list)
    ledger: TokenLedger = field(default_factory=TokenLedger)
    error: str = ""

    @classmethod
    def new(cls, source_key: str, source_sha256: str) -> JobState:
        return cls(
            job_id=uuid.uuid4().hex[:12],
            source_key=source_key,
            source_sha256=source_sha256,
        )

    def is_done(self, stage: Stage) -> bool:
        return any(r.stage is stage and r.ok for r in self.completed_stages)

    def mark(self, stage: Stage, *, ok: bool, duration_ms: int, detail: str = "") -> None:
        # Re-running a stage replaces its record rather than appending a
        # duplicate, so resume does not corrupt the audit trail.
        self.completed_stages = [r for r in self.completed_stages if r.stage is not stage]
        self.completed_stages.append(
            StageRecord(stage=stage, ok=ok, duration_ms=duration_ms, detail=detail)
        )

    @property
    def resume_from(self) -> Stage:
        """First stage not yet completed."""
        for stage in Stage.ordered():
            if not self.is_done(stage):
                return stage
        return Stage.MANIFEST

    @property
    def accepted_count(self) -> int:
        return sum(1 for s in self.segments if s.accepted)

    @property
    def spoken_count(self) -> int:
        return sum(1 for s in self.segments if s.spoken)


# Injected provider callables. Real implementations live in app.adapters.
Transcriber = Callable[[str], TranscriptInput]
SceneDescriber = Callable[[float, int], str]  # (timestamp, word_budget) -> context
Drafter = Callable[[str, int], str]
Synthesizer = Callable[[str], bytes]
Storer = Callable[[bytes, str], tuple[str, str]]  # (data, namespace) -> (key, sha256)


@dataclass(frozen=True, slots=True)
class Providers:
    """The four external capabilities the runner depends on."""

    transcribe: Transcriber
    scene_context: SceneDescriber
    draft: Drafter
    synthesize: Synthesizer
    store: Storer


class Runner:
    """Executes a job, stage by stage, resuming where it left off."""

    def __init__(self, providers: Providers, *, max_draft_attempts: int = 3) -> None:
        self._providers = providers
        self._max_draft_attempts = max_draft_attempts

    def run(self, state: JobState, source_url: str) -> JobState:
        """Advance ``state`` to completion, skipping finished stages."""
        state.status = JobStatus.RUNNING
        start_at = state.resume_from
        logger.info("job %s resuming at %s", state.job_id, start_at.value)

        try:
            for stage in Stage.ordered():
                if stage.index() < start_at.index():
                    continue
                self._execute(stage, state, source_url)
        except Exception as exc:  # noqa: BLE001 - surfaced on the job, not raised
            state.status = JobStatus.FAILED
            state.error = f"{type(exc).__name__}: {exc}"
            logger.exception("job %s failed at %s", state.job_id, state.resume_from.value)
            return state

        if not state.segments:
            state.status = JobStatus.FAILED
            state.error = "no describable gaps found"
        elif state.spoken_count == len(state.segments):
            state.status = JobStatus.COMPLETE
        else:
            # Some descriptions failed. The rest are still worth shipping.
            state.status = JobStatus.PARTIAL

        return state

    def _execute(self, stage: Stage, state: JobState, source_url: str) -> None:
        started = time.monotonic()
        detail = ""

        if stage is Stage.TRANSCRIBE:
            state.transcript = self._providers.transcribe(source_url)
            detail = f"{len(state.transcript.word_spans)} words"

        elif stage is Stage.DETECT_GAPS:
            if state.transcript is None:
                raise RuntimeError("cannot detect gaps before transcription")
            state.gaps = find_gaps(
                state.transcript.word_spans, state.transcript.duration
            )
            detail = f"{len(state.gaps)} gaps"

        elif stage is Stage.DESCRIBE:
            state.segments = self._describe_all(state)
            detail = f"{state.accepted_count}/{len(state.segments)} within budget"

        elif stage is Stage.SYNTHESIZE:
            self._synthesize_all(state)
            detail = f"{state.spoken_count}/{len(state.segments)} rendered"

        elif stage is Stage.MANIFEST:
            detail = "manifest built"

        elapsed_ms = int((time.monotonic() - started) * 1000)
        state.mark(stage, ok=True, duration_ms=elapsed_ms, detail=detail)
        logger.info("job %s %s: %s (%dms)", state.job_id, stage.value, detail, elapsed_ms)

    def _describe_all(self, state: JobState) -> list[DescribedSegment]:
        segments: list[DescribedSegment] = []
        for gap in state.gaps:
            if gap.word_budget == 0:
                continue
            context = self._providers.scene_context(gap.start, gap.word_budget)
            result: DescriptionResult = draft_description(
                gap, context, self._providers.draft,
                max_attempts=self._max_draft_attempts,
            )
            if not result.text:
                continue
            segments.append(
                DescribedSegment(
                    gap=gap,
                    text=result.text,
                    accepted=result.accepted,
                    attempts=result.attempt_count,
                )
            )
        return segments

    def _synthesize_all(self, state: JobState) -> None:
        """Render each description to audio.

        A failure here degrades one segment, never the run. The listener loses
        one description; they keep the other nineteen.
        """
        for segment in state.segments:
            try:
                audio = self._providers.synthesize(segment.text)
                key, digest = self._providers.store(audio, "description-audio")
                segment.audio_key = key
                segment.audio_sha256 = digest
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "synthesis failed for gap %.2f-%.2f: %s",
                    segment.gap.start, segment.gap.end, exc,
                )


def build_manifest(state: JobState) -> Manifest:
    """Render a completed job as a provenance manifest."""
    manifest = Manifest.new(state.job_id, state.source_key, state.source_sha256)

    for record in sorted(state.completed_stages, key=lambda r: r.stage.index()):
        manifest.add_step(
            StepRecord(
                index=record.stage.index(),
                name=record.stage.value,
                provider="interlude",
                model="",
                duration_ms=record.duration_ms,
                accepted=record.ok,
            )
        )

    for segment in state.segments:
        if not segment.audio_key:
            continue
        manifest.add_asset(
            AssetRecord(
                key=segment.audio_key,
                sha256=segment.audio_sha256,
                size=0,
                namespace="description-audio",
                content_type="audio/mpeg",
            )
        )

    summary = state.ledger.summary()
    manifest.tokens = TokenUsage(
        prompt_tokens_uncompressed=int(summary["prompt_tokens_uncompressed"]),
        prompt_tokens_compressed=int(summary["prompt_tokens_compressed"]),
        completion_tokens=int(summary["completion_tokens"]),
        calls=int(summary["calls"]),
    )
    manifest.metadata = {
        "status": state.status.value,
        "gaps_found": len(state.gaps),
        "segments_described": len(state.segments),
        "segments_within_budget": state.accepted_count,
        "segments_rendered": state.spoken_count,
    }
    return manifest
