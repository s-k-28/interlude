"""End-to-end integration tests for the Interlude pipeline.

These exercise the real pipeline objects (gaps -> describe -> runner ->
manifest -> tokens) wired to stub providers. They require no network access,
no API keys, and no vendor SDKs: every external boundary is a plain Python
callable supplied by the test.
"""

from __future__ import annotations

import json

import pytest

from app.pipeline.describe import count_words
from app.pipeline.gaps import MIN_GAP_SECONDS, find_gaps
from app.pipeline.manifest import verify_hash
from app.pipeline.runner import (
    JobState,
    JobStatus,
    Providers,
    Runner,
    Stage,
    TranscriptInput,
    build_manifest,
)
from app.pipeline.tokens import TokenLedger

SOURCE_KEY = "lectures/bio101/week3.mp4"
SOURCE_SHA = "a" * 64
SOURCE_URL = "file:///tmp/week3.mp4"

DEFAULT_SPANS: list[tuple[float, float]] = [(0.0, 5.0), (20.0, 25.0), (40.0, 45.0)]
DEFAULT_DURATION = 60.0


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def words(n: int) -> str:
    """Return a string of ``n`` space-separated tokens."""
    return " ".join(f"w{i}" for i in range(n))


def make_providers(**overrides) -> Providers:
    """Build a fully working stub provider bundle; override any one callable."""

    def transcribe(source_url: str) -> TranscriptInput:
        return TranscriptInput(
            text="the lecturer writes on the board",
            word_spans=list(DEFAULT_SPANS),
            duration=DEFAULT_DURATION,
        )

    def scene_context(start: float, budget: int) -> str:
        return f"A diagram is drawn on the whiteboard at t={start:.1f}s."

    def draft(prompt: str, budget: int) -> str:
        return words(max(1, min(budget, 4)))

    def synthesize(text: str) -> bytes:
        return b"RIFF" + text.encode("utf-8")

    def store(payload: bytes, name: str) -> tuple[str, str]:
        return (f"audio/{name}", f"{len(payload):064d}")

    bundle = {
        "transcribe": transcribe,
        "scene_context": scene_context,
        "draft": draft,
        "synthesize": synthesize,
        "store": store,
    }
    bundle.update(overrides)
    return Providers(**bundle)


def make_state() -> JobState:
    return JobState.new(SOURCE_KEY, SOURCE_SHA)


def run_job(providers: Providers | None = None, state: JobState | None = None) -> JobState:
    providers = providers if providers is not None else make_providers()
    state = state if state is not None else make_state()
    return Runner(providers).run(state, SOURCE_URL)


def gap_for(state: JobState, segment, index: int):
    """Gap belonging to a segment, tolerating segments that carry it directly."""
    gap = getattr(segment, "gap", None)
    if gap is not None:
        return gap
    return state.gaps[index]


def segment_text(segment) -> str:
    return getattr(segment, "text", "") or ""


def segment_audio_key(segment):
    return getattr(segment, "audio_key", None)


# --------------------------------------------------------------------------
# GROUP 1 - full happy path
# --------------------------------------------------------------------------


def test_lecture_with_three_speech_bursts_is_fully_described_and_voiced() -> None:
    state = run_job()

    assert state.status is JobStatus.COMPLETE
    assert state.gaps, "expected silence windows between speech bursts"
    assert len(state.segments) == len(state.gaps)
    assert state.accepted_count == len(state.segments)
    assert state.spoken_count == len(state.segments)


def test_every_pipeline_stage_is_recorded_in_execution_order() -> None:
    state = run_job()

    assert [r.stage for r in state.completed_stages] == list(Stage.ordered())
    indexes = [r.stage.index() for r in state.completed_stages]
    assert indexes == sorted(indexes)
    for stage in Stage.ordered():
        assert state.is_done(stage)


def test_every_described_segment_receives_playable_audio() -> None:
    state = run_job()

    for segment in state.segments:
        key = segment_audio_key(segment)
        assert key, "each described segment must carry a non-empty audio key"


# --------------------------------------------------------------------------
# GROUP 2 - the word budget contract holds end to end
# --------------------------------------------------------------------------


def test_overlong_drafts_are_never_accepted() -> None:
    providers = make_providers(draft=lambda prompt, budget: words(200))
    state = run_job(providers)

    assert state.segments
    for segment in state.segments:
        assert segment.accepted is False


def test_description_never_overruns_its_silence_window() -> None:
    providers = make_providers(draft=lambda prompt, budget: words(200))
    state = run_job(providers)

    assert state.segments
    for index, segment in enumerate(state.segments):
        gap = gap_for(state, segment, index)
        assert count_words(segment_text(segment)) <= gap.word_budget


def test_short_drafts_are_accepted_on_the_first_attempt() -> None:
    providers = make_providers(draft=lambda prompt, budget: words(3))
    state = run_job(providers)

    assert state.segments
    for segment in state.segments:
        assert segment.accepted is True
    assert state.accepted_count == len(state.segments)


def test_final_text_respects_each_segments_own_budget() -> None:
    for size in (3, 200):
        providers = make_providers(draft=lambda prompt, budget, n=size: words(n))
        state = run_job(providers)
        for index, segment in enumerate(state.segments):
            gap = gap_for(state, segment, index)
            assert gap.word_budget == int(gap.duration * 165 / 60)
            assert count_words(segment_text(segment)) <= gap.word_budget


# --------------------------------------------------------------------------
# GROUP 3 - resume
# --------------------------------------------------------------------------


def test_already_transcribed_job_does_not_retranscribe() -> None:
    calls = {"transcribe": 0}

    def transcribe(source_url: str) -> TranscriptInput:
        calls["transcribe"] += 1
        return TranscriptInput("x", list(DEFAULT_SPANS), DEFAULT_DURATION)

    state = make_state()
    state.transcript = TranscriptInput(
        text="pre-existing transcript",
        word_spans=list(DEFAULT_SPANS),
        duration=DEFAULT_DURATION,
    )
    state.mark(Stage.TRANSCRIBE, ok=True, duration_ms=0, detail="restored")

    result = run_job(make_providers(transcribe=transcribe), state)

    assert calls["transcribe"] == 0
    assert result.status is JobStatus.COMPLETE


def test_rerunning_a_finished_job_does_not_duplicate_stage_records() -> None:
    providers = make_providers()
    runner = Runner(providers)
    state = runner.run(make_state(), SOURCE_URL)
    assert state.status is JobStatus.COMPLETE

    again = runner.run(state, SOURCE_URL)

    assert len(again.completed_stages) == 5
    assert again.status is JobStatus.COMPLETE


# --------------------------------------------------------------------------
# GROUP 4 - partial failure
# --------------------------------------------------------------------------


def test_one_failed_voice_render_does_not_discard_the_others() -> None:
    seen: list[str] = []

    def synthesize(text: str) -> bytes:
        if not seen:
            seen.append(text)
        if text == seen[0]:
            raise RuntimeError("tts backend refused this utterance")
        return b"RIFF" + text.encode("utf-8")

    counter = {"n": 0}

    def draft(prompt: str, budget: int) -> str:
        counter["n"] += 1
        return f"scene {counter['n']}"

    state = run_job(make_providers(draft=draft, synthesize=synthesize))

    assert state.status is JobStatus.PARTIAL
    assert state.segments
    assert state.spoken_count < len(state.segments)
    rendered = [s for s in state.segments if segment_audio_key(s)]
    assert rendered, "surviving segments must keep their audio keys"
    assert len(rendered) == state.spoken_count


def test_transcription_outage_fails_the_job_without_raising() -> None:
    def transcribe(source_url: str) -> TranscriptInput:
        raise ConnectionError("speech service unreachable")

    state = run_job(make_providers(transcribe=transcribe))

    assert state.status is JobStatus.FAILED
    assert state.error is not None
    assert "ConnectionError" in state.error


# --------------------------------------------------------------------------
# GROUP 5 - degenerate inputs
# --------------------------------------------------------------------------


def test_wall_to_wall_speech_leaves_nowhere_to_describe() -> None:
    def transcribe(source_url: str) -> TranscriptInput:
        return TranscriptInput("nonstop", [(0.0, 60.0)], 60.0)

    state = run_job(make_providers(transcribe=transcribe))

    assert state.status is JobStatus.FAILED
    assert state.error is not None
    assert "no describable gaps" in state.error


def test_silent_video_is_one_long_describable_window() -> None:
    gaps = find_gaps([], 30.0)

    assert len(gaps) == 1
    assert gaps[0].start == pytest.approx(0.0)
    assert gaps[0].end == pytest.approx(30.0)
    assert gaps[0].duration == pytest.approx(30.0)


def test_clip_shorter_than_the_minimum_window_yields_no_gaps() -> None:
    assert find_gaps([], MIN_GAP_SECONDS - 0.1) == []


def test_overlapping_speaker_spans_do_not_create_phantom_gaps() -> None:
    gaps = find_gaps([(0.0, 10.0), (3.0, 4.0), (11.5, 12.0)], 12.0)

    assert len(gaps) == 1
    assert gaps[0].start == pytest.approx(10.0)
    assert gaps[0].end == pytest.approx(11.5)


# --------------------------------------------------------------------------
# GROUP 6 - manifest integrity end to end
# --------------------------------------------------------------------------


def test_manifest_hash_is_a_full_length_hex_digest() -> None:
    state = run_job()
    manifest = build_manifest(state)

    digest = manifest.canonical_hash()
    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)


def test_untouched_manifest_verifies() -> None:
    manifest = build_manifest(run_job())

    assert verify_hash(manifest.to_json()) is True


def test_tampering_with_the_source_digest_breaks_verification() -> None:
    manifest = build_manifest(run_job())
    document = json.loads(manifest.to_json())

    document["source"]["sha256"] = "b" * 64

    assert verify_hash(json.dumps(document)) is False


def test_manifest_metadata_reports_the_accessibility_counters() -> None:
    manifest = build_manifest(run_job())

    for key in (
        "status",
        "gaps_found",
        "segments_described",
        "segments_within_budget",
        "segments_rendered",
    ):
        assert key in manifest.metadata


def test_only_rendered_segments_are_published_as_assets() -> None:
    def synthesize(text: str) -> bytes:
        raise RuntimeError("no voice for you")

    state = run_job(make_providers(synthesize=synthesize))
    manifest = build_manifest(state)

    assert [s for s in state.segments if segment_audio_key(s)] == []
    assert list(manifest.assets) == []


# --------------------------------------------------------------------------
# GROUP 7 - token ledger through a run
# --------------------------------------------------------------------------


def test_compression_savings_survive_into_the_manifest() -> None:
    ledger = TokenLedger()
    ledger.record(uncompressed_tokens=1000, compressed_tokens=250)
    ledger.record(uncompressed_tokens=500, compressed_tokens=125)

    state = run_job()
    state.ledger = ledger
    manifest = build_manifest(state)

    assert manifest.tokens.prompt_tokens_uncompressed == 1500
    assert manifest.tokens.prompt_tokens_compressed == 375
    assert manifest.tokens.reduction_ratio == pytest.approx(0.75)


def test_empty_ledger_reports_no_savings_instead_of_dividing_by_zero() -> None:
    summary = TokenLedger().summary()

    assert summary["calls"] == 0
    assert summary["prompt_tokens_uncompressed"] == 0
    assert summary["reduction_ratio"] == pytest.approx(0.0)
