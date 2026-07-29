"""Sole entry point for the Genblaze SDK.

Per the Unverified-API Protocol, no other module imports genblaze. Everything
crossing this boundary is a plain dataclass defined here, so a change in the SDK
touches one file.

VERIFIED against wheel source (genblaze-core 0.3.8, genblaze-assemblyai 0.3.2,
genblaze-elevenlabs 0.3.3, genblaze-google 0.3.4):

  * AssemblyAIProvider(api_key=None, *, poll_interval=3.0, models=None,
        retry_policy=None, probe_cache_ttl=None, probe_cache_max_entries=None)
    - name = "assemblyai"
    - Reads ASSEMBLYAI_API_KEY from the environment when api_key is None.
    - Produces a TEXT asset; the transcript string lands in
      ``asset.metadata["text"]`` and word timings in
      ``asset.audio.word_timings`` as list[WordTiming].
  * WordTiming(word: str, start: float, end: float, confidence: float|None)
    - Times are SECONDS. The provider divides AssemblyAI's milliseconds by 1000.
    - Validator enforces end >= start.
  * AudioMetadata(sample_rate, channels, codec, bitrate, word_timings)
  * ElevenLabsTTSProvider(api_key=None, output_dir=None, *, models=None, ...)
    - name = "elevenlabs-tts"
    - Model ids present in source: eleven_v3, eleven_multilingual_v2,
      eleven_turbo_v2_5, eleven_flash_v2_5
  * Pipeline(name=None, tenant_id=None, *, chain=False, max_concurrency=None,
        tracer=None, preflight=True, ...)
    .step(provider, model, prompt=None, modality=..., input_from=...,
          fallback_models=[...], params={...}) -> Pipeline
    .run(sink=None, fail_fast=True, timeout=None, max_retries=None, ...) -> PipelineResult
    .resume_step(...) for partial-failure recovery

CRITICAL FINDING — Gemini text/vision is NOT in genblaze-google:
  genblaze-google ships GeminiImageProvider ("google-gemini-image"),
  ImagenProvider ("google-imagen"), and VeoProvider ("google-veo"). All three are
  IMAGE/VIDEO generators. There is no text or vision-understanding provider.

  Interlude's scene-description step needs Gemini *vision→text*. That is not
  available through genblaze, so that one step calls google-genai directly
  (already an installed dependency of genblaze-google). Genblaze remains
  load-bearing for transcription, TTS, provenance and storage — but we must not
  claim it orchestrates a step it cannot reach. See README-DECISIONS.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Boundary types. Deliberately plain; nothing here is an SDK object.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Word:
    """One spoken word with second-resolution boundaries."""

    text: str
    start: float
    end: float
    confidence: float | None = None

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(f"word end {self.end} precedes start {self.start}")


@dataclass(frozen=True, slots=True)
class Transcript:
    """A transcription result, reduced to what the pipeline needs."""

    text: str
    words: list[Word] = field(default_factory=list)
    language: str | None = None

    @property
    def word_spans(self) -> list[tuple[float, float]]:
        """Timestamp pairs, the shape :func:`app.pipeline.gaps.find_gaps` wants."""
        return [(w.start, w.end) for w in self.words]


@dataclass(frozen=True, slots=True)
class SynthesizedAudio:
    """A rendered speech clip."""

    audio: bytes
    voice_id: str
    model_id: str

    @property
    def size(self) -> int:
        return len(self.audio)


class GenblazeUnavailable(RuntimeError):
    """Raised when the SDK or a required credential is absent."""


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------


def _coerce_words(raw_timings: Any) -> list[Word]:
    """Convert SDK WordTiming objects into boundary types.

    Defensive by design: a malformed entry is dropped rather than aborting a
    transcript that is otherwise usable.
    """
    words: list[Word] = []
    for timing in raw_timings or []:
        try:
            start = float(getattr(timing, "start", 0.0) or 0.0)
            end = float(getattr(timing, "end", 0.0) or 0.0)
            if end < start:
                continue
            words.append(
                Word(
                    text=str(getattr(timing, "word", "") or ""),
                    start=start,
                    end=end,
                    confidence=getattr(timing, "confidence", None),
                )
            )
        except (TypeError, ValueError):
            continue
    return words


def transcribe(audio_url: str, *, api_key: str | None = None) -> Transcript:
    """Transcribe audio, returning text plus word-level timings.

    Args:
        audio_url: Publicly reachable URL. AssemblyAI fetches it server-side, so
            a presigned B2 URL works.
        api_key: Overrides ASSEMBLYAI_API_KEY when supplied.
    """
    if not audio_url:
        raise ValueError("audio_url must not be empty")

    key = api_key or os.environ.get("ASSEMBLYAI_API_KEY", "").strip()
    if not key:
        raise GenblazeUnavailable("ASSEMBLYAI_API_KEY is not set")

    try:
        from genblaze import Pipeline
        from genblaze_assemblyai import AssemblyAIProvider
    except ImportError as exc:
        raise GenblazeUnavailable(f"genblaze assemblyai extra not installed: {exc}") from exc

    provider = AssemblyAIProvider(api_key=key)

    # UNVERIFIED: exact param name for the source audio on the transcription
    # step. Source shows the provider consumes an audio URL, but the payload key
    # is resolved through the model registry's alias pipeline.
    # Confidence: medium. Blast radius: this function.
    result = (
        Pipeline(name="interlude-transcribe")
        .step(provider, model="best", params={"audio_url": audio_url})
        .run()
    )

    asset = _first_asset(result)
    if asset is None:
        raise GenblazeUnavailable("transcription returned no asset")

    text = str((getattr(asset, "metadata", {}) or {}).get("text", ""))
    audio_meta = getattr(asset, "audio", None)
    words = _coerce_words(getattr(audio_meta, "word_timings", None))
    language = (getattr(asset, "metadata", {}) or {}).get("language")

    logger.info("transcribed %d words", len(words))
    return Transcript(text=text, words=words, language=language)


def _first_asset(result: Any) -> Any:
    """Pull the first asset off a PipelineResult, tolerating shape differences."""
    for attr in ("assets", "outputs"):
        items = getattr(result, attr, None)
        if items:
            return items[0]
    steps = getattr(result, "steps", None) or []
    for step in steps:
        assets = getattr(step, "assets", None)
        if assets:
            return assets[0]
    return None


# ---------------------------------------------------------------------------
# Speech synthesis
# ---------------------------------------------------------------------------

DEFAULT_TTS_MODEL = "eleven_turbo_v2_5"


def synthesize(
    text: str,
    *,
    voice_id: str,
    model_id: str = DEFAULT_TTS_MODEL,
    api_key: str | None = None,
) -> SynthesizedAudio:
    """Render a description to speech."""
    if not text.strip():
        raise ValueError("text must not be empty")

    key = api_key or os.environ.get("ELEVENLABS_API_KEY", "").strip()
    if not key:
        raise GenblazeUnavailable("ELEVENLABS_API_KEY is not set")

    try:
        from genblaze import Pipeline
        from genblaze_elevenlabs import ElevenLabsTTSProvider
    except ImportError as exc:
        raise GenblazeUnavailable(f"genblaze elevenlabs extra not installed: {exc}") from exc

    provider = ElevenLabsTTSProvider(api_key=key)

    # UNVERIFIED: params shape for voice selection. Registry aliases may accept
    # voice_id / voice. Confidence: medium. Blast radius: this function.
    result = (
        Pipeline(name="interlude-tts")
        .step(provider, model=model_id, prompt=text, params={"voice_id": voice_id})
        .run()
    )

    asset = _first_asset(result)
    if asset is None:
        raise GenblazeUnavailable("synthesis returned no asset")

    audio = _read_asset_bytes(asset)
    return SynthesizedAudio(audio=audio, voice_id=voice_id, model_id=model_id)


def _read_asset_bytes(asset: Any) -> bytes:
    """Extract bytes from an asset, whether inline or on disk."""
    inline = getattr(asset, "data", None)
    if isinstance(inline, bytes | bytearray):
        return bytes(inline)

    path = getattr(asset, "path", None) or getattr(asset, "local_path", None)
    if path:
        with open(path, "rb") as handle:
            return handle.read()

    url = str(getattr(asset, "url", "") or "")
    if url.startswith("file://"):
        with open(url.removeprefix("file://"), "rb") as handle:
            return handle.read()

    raise GenblazeUnavailable(f"cannot read bytes from asset {url or asset!r}")


def sdk_version() -> str:
    try:
        from importlib.metadata import version

        return version("genblaze")
    except Exception:  # noqa: BLE001
        return "unknown"


if __name__ == "__main__":
    w = Word("hello", 1.0, 1.5, 0.98)
    assert w.end >= w.start

    try:
        Word("bad", 5.0, 2.0)
        raise AssertionError("should have rejected inverted bounds")
    except ValueError:
        pass

    t = Transcript(text="a b", words=[Word("a", 0.0, 0.5), Word("b", 2.0, 2.4)])
    assert t.word_spans == [(0.0, 0.5), (2.0, 2.4)]

    class _FakeTiming:
        def __init__(self, word, start, end, confidence=None):
            self.word, self.start, self.end, self.confidence = word, start, end, confidence

    coerced = _coerce_words([_FakeTiming("x", 0.0, 1.0, 0.9), _FakeTiming("bad", 5.0, 2.0)])
    assert len(coerced) == 1, "inverted timing must be dropped, not fatal"
    assert coerced[0].text == "x"
    assert _coerce_words(None) == []

    assert SynthesizedAudio(b"abc", "v", "m").size == 3

    print(f"genblaze_adapter self-check OK (sdk={sdk_version()})")
