"""Tests for job orchestration.

Every provider is a stub, so these exercise real control flow with no network,
no keys, and no SDK. The properties under test are resumability and graceful
degradation — the two things that separate a pipeline from a demo script.
"""

from __future__ import annotations

import pytest

from app.pipeline.gaps import Gap
from app.pipeline.runner import (
    DescribedSegment,
    JobState,
    JobStatus,
    Providers,
    Runner,
    Stage,
    StageRecord,
    TranscriptInput,
    build_manifest,
)


def words(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


def make_providers(
    *,
    transcript: TranscriptInput | None = None,
    draft_words: int = 3,
    synth_fails_on: set[str] | None = None,
    calls: dict[str, int] | None = None,
) -> Providers:
    """Build a stub provider set with optional failure injection."""
    counters = calls if calls is not None else {}
    fails = synth_fails_on or set()

    default_transcript = TranscriptInput(
        text="hello world",
        word_spans=[(0.0, 2.0), (8.0, 9.0)],
        duration=20.0,
    )

    def transcribe(_url: str) -> TranscriptInput:
        counters["transcribe"] = counters.get("transcribe", 0) + 1
        return transcript if transcript is not None else default_transcript

    def scene_context(_ts: float, _budget: int) -> str:
        counters["scene"] = counters.get("scene", 0) + 1
        return "a lecture hall"

    def draft(_prompt: str, _budget: int) -> str:
        counters["draft"] = counters.get("draft", 0) + 1
        return words(draft_words)

    def synthesize(text: str) -> bytes:
        counters["synthesize"] = counters.get("synthesize", 0) + 1
        if text in fails:
            raise RuntimeError("tts provider unavailable")
        return b"audio-" + text.encode()

    def store(data: bytes, namespace: str) -> tuple[str, str]:
        counters["store"] = counters.get("store", 0) + 1
        return f"{namespace}/key-{len(data)}", f"sha-{len(data)}"

    return Providers(
        transcribe=transcribe,
        scene_context=scene_context,
        draft=draft,
        synthesize=synthesize,
        store=store,
    )


class TestStage:
    def test_order_is_stable(self) -> None:
        assert [s.value for s in Stage.ordered()] == [
            "transcribe", "detect_gaps", "describe", "synthesize", "manifest",
        ]

    def test_index_reflects_order(self) -> None:
        assert Stage.TRANSCRIBE.index() == 0
        assert Stage.MANIFEST.index() == 4


class TestJobState:
    def test_new_generates_id(self) -> None:
        state = JobState.new("source/k", "abc")
        assert len(state.job_id) == 12
        assert state.status is JobStatus.PENDING

    def test_resume_from_starts_at_first_stage(self) -> None:
        assert JobState.new("k", "h").resume_from is Stage.TRANSCRIBE

    def test_resume_from_skips_completed(self) -> None:
        state = JobState.new("k", "h")
        state.mark(Stage.TRANSCRIBE, ok=True, duration_ms=10)
        state.mark(Stage.DETECT_GAPS, ok=True, duration_ms=5)
        assert state.resume_from is Stage.DESCRIBE

    def test_failed_stage_does_not_count_as_done(self) -> None:
        state = JobState.new("k", "h")
        state.mark(Stage.TRANSCRIBE, ok=False, duration_ms=10)
        assert state.resume_from is Stage.TRANSCRIBE

    def test_marking_twice_replaces_not_duplicates(self) -> None:
        # Resume must not corrupt the audit trail with repeated entries.
        state = JobState.new("k", "h")
        state.mark(Stage.TRANSCRIBE, ok=False, duration_ms=10)
        state.mark(Stage.TRANSCRIBE, ok=True, duration_ms=20)
        records = [r for r in state.completed_stages if r.stage is Stage.TRANSCRIBE]
        assert len(records) == 1
        assert records[0].ok

    def test_counts(self) -> None:
        state = JobState.new("k", "h")
        state.segments = [
            DescribedSegment(Gap(0, 5), "a", accepted=True, attempts=1, audio_key="k1"),
            DescribedSegment(Gap(6, 9), "b", accepted=False, attempts=3),
        ]
        assert state.accepted_count == 1
        assert state.spoken_count == 1


class TestRunnerHappyPath:
    def test_completes(self) -> None:
        state = JobState.new("source/k", "abc")
        result = Runner(make_providers()).run(state, "https://example/v.mp4")
        assert result.status is JobStatus.COMPLETE
        assert result.error == ""

    def test_all_stages_recorded(self) -> None:
        state = JobState.new("source/k", "abc")
        result = Runner(make_providers()).run(state, "u")
        assert {r.stage for r in result.completed_stages} == set(Stage.ordered())

    def test_finds_gaps_and_describes_them(self) -> None:
        state = JobState.new("source/k", "abc")
        result = Runner(make_providers()).run(state, "u")
        # spans (0,2) and (8,9) in a 20s video -> gaps 2-8 and 9-20
        assert len(result.gaps) == 2
        assert len(result.segments) == 2

    def test_segments_carry_audio_keys(self) -> None:
        state = JobState.new("source/k", "abc")
        result = Runner(make_providers()).run(state, "u")
        assert all(s.audio_key for s in result.segments)


class TestRunnerResume:
    def test_skips_already_completed_stages(self) -> None:
        calls: dict[str, int] = {}
        state = JobState.new("source/k", "abc")
        state.transcript = TranscriptInput("t", [(0.0, 2.0), (8.0, 9.0)], 20.0)
        state.mark(Stage.TRANSCRIBE, ok=True, duration_ms=1)

        Runner(make_providers(calls=calls)).run(state, "u")
        # The expensive transcription call must not be repeated.
        assert "transcribe" not in calls

    def test_resume_reaches_completion(self) -> None:
        state = JobState.new("source/k", "abc")
        state.transcript = TranscriptInput("t", [(0.0, 2.0), (8.0, 9.0)], 20.0)
        state.mark(Stage.TRANSCRIBE, ok=True, duration_ms=1)
        result = Runner(make_providers(calls={})).run(state, "u")
        assert result.status is JobStatus.COMPLETE


class TestRunnerDegradation:
    def test_partial_when_one_synthesis_fails(self) -> None:
        # Two gaps, both drafted identically; failing that text kills both.
        # Use distinct drafts so exactly one fails.
        drafts = iter(["alpha beta", "gamma delta"])
        providers = make_providers(synth_fails_on={"alpha beta"})
        providers = Providers(
            transcribe=providers.transcribe,
            scene_context=providers.scene_context,
            draft=lambda _p, _b: next(drafts),
            synthesize=providers.synthesize,
            store=providers.store,
        )
        state = JobState.new("source/k", "abc")
        result = Runner(providers).run(state, "u")

        assert result.status is JobStatus.PARTIAL
        assert result.spoken_count == 1
        assert len(result.segments) == 2

    def test_failed_transcription_fails_the_job(self) -> None:
        def boom(_url: str) -> TranscriptInput:
            raise ConnectionError("provider down")

        base = make_providers()
        providers = Providers(
            transcribe=boom,
            scene_context=base.scene_context,
            draft=base.draft,
            synthesize=base.synthesize,
            store=base.store,
        )
        state = JobState.new("source/k", "abc")
        result = Runner(providers).run(state, "u")

        assert result.status is JobStatus.FAILED
        assert "ConnectionError" in result.error

    def test_runner_never_raises(self) -> None:
        # Failures are surfaced on the job, not thrown at the caller — a batch
        # of 400 videos must not abort because one is malformed.
        def boom(_url: str) -> TranscriptInput:
            raise RuntimeError("kaboom")

        base = make_providers()
        providers = Providers(
            transcribe=boom, scene_context=base.scene_context, draft=base.draft,
            synthesize=base.synthesize, store=base.store,
        )
        Runner(providers).run(JobState.new("k", "h"), "u")  # must not raise

    def test_no_gaps_is_a_failure_not_a_crash(self) -> None:
        # Continuous speech: nowhere to put a description.
        transcript = TranscriptInput("t", [(0.0, 20.0)], 20.0)
        state = JobState.new("source/k", "abc")
        result = Runner(make_providers(transcript=transcript)).run(state, "u")
        assert result.status is JobStatus.FAILED
        assert "no describable gaps" in result.error


class TestBuildManifest:
    def test_records_every_stage(self) -> None:
        state = JobState.new("source/k", "abc")
        result = Runner(make_providers()).run(state, "u")
        manifest = build_manifest(result)
        assert len(manifest.steps) == len(Stage.ordered())

    def test_steps_in_pipeline_order(self) -> None:
        state = JobState.new("source/k", "abc")
        result = Runner(make_providers()).run(state, "u")
        manifest = build_manifest(result)
        assert [s.index for s in manifest.steps] == sorted(s.index for s in manifest.steps)

    def test_only_rendered_segments_become_assets(self) -> None:
        state = JobState.new("source/k", "abc")
        state.segments = [
            DescribedSegment(Gap(0, 5), "a", accepted=True, attempts=1, audio_key="k1", audio_sha256="s1"),
            DescribedSegment(Gap(6, 9), "b", accepted=True, attempts=1),  # never rendered
        ]
        manifest = build_manifest(state)
        assert len(manifest.assets) == 1

    def test_metadata_reports_counts(self) -> None:
        state = JobState.new("source/k", "abc")
        result = Runner(make_providers()).run(state, "u")
        manifest = build_manifest(result)
        assert manifest.metadata["segments_rendered"] == 2
        assert manifest.metadata["status"] == "complete"

    def test_manifest_hash_is_computable(self) -> None:
        state = JobState.new("source/k", "abc")
        result = Runner(make_providers()).run(state, "u")
        assert len(build_manifest(result).canonical_hash()) == 64
