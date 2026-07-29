"""Composition root for Interlude.

This is the ONLY module where the pure pipeline (``app.pipeline.runner``),
the outside-world adapters (``app.adapters.*``), storage (``app.storage.store``)
and configuration (``app.config``) are allowed to meet.

Everything downstream of here is pure: the orchestrator receives a
``Providers`` bundle of plain callables and never learns which SDK, HTTP
client or credential is behind them. Everything upstream of here is a vendor
adapter that knows nothing about Interlude's pipeline.

Rules enforced in this file:
  * No vendor SDK is imported directly (no genblaze, no paritok, no
    google.genai). Only ``app.adapters.*`` wrappers.
  * Every provider callable is a closure over its injected dependencies, so
    tests can swap store / frame loader / ledger without monkeypatching.
  * ``build_offline_providers`` gives a credential-free, network-free bundle
    for demos, CI and the self-check at the bottom of this file.
"""

from __future__ import annotations

import hashlib
from typing import Protocol, runtime_checkable

from app.adapters import gemini_adapter, genblaze_adapter, paritok_adapter
from app.pipeline.runner import Providers, TranscriptInput
from app.pipeline.tokens import TokenLedger
from app.storage.store import B2Store

# ElevenLabs' stock "Rachel" voice. Sensible default for a demo, but callers
# should override it per-course / per-institution via build_providers(voice_id=...).
DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"

# Namespaces recognised by B2Store. Kept here so the storage closure below and
# any caller choosing a namespace agree on spelling.
NAMESPACES = (
    "source",
    "transcript",
    "description-text",
    "description-audio",
    "mixed",
)


@runtime_checkable
class FrameLoader(Protocol):
    """Extracts a single video frame as encoded image bytes.

    Real frame extraction (ffmpeg seek + JPEG encode) is owned by another
    agent; this module only consumes the callable. Implementations must return
    bytes decodable by the vision model (JPEG by default).
    """

    def __call__(self, timestamp: float) -> bytes:  # pragma: no cover - protocol
        ...


# A JPEG SOI marker + JFIF APP0 header + EOI. Structurally a JPEG envelope,
# deliberately tiny; it carries no scan data and is NOT a decodable picture.
_STUB_JPEG_BYTES = (
    b"\xff\xd8"  # SOI
    b"\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"  # APP0
    b"\xff\xd9"  # EOI
)


class StubFrameLoader:
    """STUB. Returns the same tiny JPEG envelope for every timestamp.

    Exists so the pipeline can be wired and exercised offline before the real
    ffmpeg-backed loader lands. It must never be used in production: every
    frame looks identical, so scene descriptions would be meaningless.
    """

    def __init__(self, frame_bytes: bytes = _STUB_JPEG_BYTES) -> None:
        self._frame_bytes = frame_bytes

    def __call__(self, timestamp: float) -> bytes:
        del timestamp  # deliberately ignored: the stub is timestamp-invariant
        return self._frame_bytes


def _keys(settings):
    """Return (assemblyai, google, elevenlabs) keys, or Nones when unset.

    Passing None lets each adapter fall back to its own environment-variable
    lookup, which is what we want in local dev and CI.
    """
    if settings is None:
        return None, None, None
    providers = getattr(settings, "providers", None)
    if providers is None:
        return None, None, None
    return (
        getattr(providers, "assemblyai_api_key", None),
        getattr(providers, "google_api_key", None),
        getattr(providers, "elevenlabs_api_key", None),
    )


def build_providers(
    store: B2Store,
    frame_loader: FrameLoader,
    ledger: TokenLedger,
    *,
    voice_id: str = DEFAULT_VOICE_ID,
    gemini_model: str = "gemini-2.0-flash",
    duration_padding_seconds: float = 0.0,
    settings=None,
) -> Providers:
    """Assemble the live provider bundle the orchestrator runs against.

    Args:
        store: B2 storage handle; used by the ``store`` callable only.
        frame_loader: supplies image bytes for a timestamp (see FrameLoader).
        ledger: token accounting; ``scene_context`` records every model call.
        voice_id: ElevenLabs voice for synthesis.
        gemini_model: vision model id for frame description.
        duration_padding_seconds: added to the transcript-derived duration
            (see the comment in ``transcribe``).
        settings: optional ``app.config.Settings``. When None, adapters read
            their own environment variables.
    """
    assemblyai_key, google_key, elevenlabs_key = _keys(settings)

    def transcribe(source_url: str) -> TranscriptInput:
        t = genblaze_adapter.transcribe(source_url, api_key=assemblyai_key)
        # LIMITATION: we derive duration from the last spoken word's end time.
        # That UNDERSTATES the true video duration whenever the lecture ends in
        # silence (Q&A pause, slide left on screen, applause fade-out). The
        # scheduler uses duration to decide where description gaps may go, so an
        # understated duration silently drops the tail of the video. A caller
        # who knows the real duration (e.g. from ffprobe on the source) should
        # pass duration_padding_seconds to correct it. Proper fix is a real
        # container-level duration probe, owned elsewhere.
        duration = max((w.end for w in t.words), default=0.0)
        return TranscriptInput(
            text=t.text,
            word_spans=t.word_spans,
            duration=duration + duration_padding_seconds,
        )

    def scene_context(timestamp: float, word_budget: int) -> str:
        frame = frame_loader(timestamp)
        result = gemini_adapter.describe_frame(
            frame,
            word_budget,
            timestamp=timestamp,
            model=gemini_model,
            api_key=google_key,
        )
        compression = result.compression
        ledger.record(
            uncompressed_tokens=(
                compression.original_tokens if compression else result.prompt_tokens
            ),
            compressed_tokens=(
                compression.compressed_tokens if compression else result.prompt_tokens
            ),
            completion_tokens=result.completion_tokens,
        )
        return result.text

    def draft(prompt: str, budget: int) -> str:
        # CONTRACT DELTA. The retry loop hands us a fully-built *text* prompt and
        # expects refined text back. Every adapter function we have takes an
        # image (describe_frame) or audio (transcribe/synthesize); there is no
        # text-in/text-out entry point, and inventing a raw SDK call here would
        # violate the adapter boundary. Compression is ready to go
        # (paritok_adapter.compress_prefix(prompt) + ledger.record of both arms)
        # the moment the adapter exists.
        raise NotImplementedError(
            "draft() needs a text-only adapter entry point that does not exist yet: "
            "gemini_adapter.complete_text(prompt: str, *, model, api_key) "
            "-> tuple[str, int, int]  # (text, prompt_tokens, completion_tokens). "
            f"Called with budget={budget}, prompt of {len(prompt)} chars "
            f"(~{paritok_adapter.count_tokens(prompt)} tokens)."
        )

    def synthesize(text: str) -> bytes:
        return genblaze_adapter.synthesize(
            text, voice_id=voice_id, api_key=elevenlabs_key
        ).audio

    def store_bytes(data: bytes, namespace: str) -> tuple[str, str]:
        extension = ".mp3" if namespace == "description-audio" else ""
        obj = store.put(data, namespace=namespace, extension=extension)
        return obj.key, obj.sha256

    return Providers(
        transcribe=transcribe,
        scene_context=scene_context,
        draft=draft,
        synthesize=synthesize,
        store=store_bytes,
    )


def build_offline_providers(ledger: TokenLedger) -> Providers:
    """Credential-free, network-free provider bundle for demos and tests.

    Deterministic by construction: same inputs, same outputs, no clock, no IO.
    """

    def transcribe(source_url: str) -> TranscriptInput:
        del source_url
        return TranscriptInput(
            text="welcome to lecture",
            word_spans=[(0.0, 1.0), (1.0, 2.0)],
            duration=20.0,
        )

    def scene_context(timestamp: float, word_budget: int) -> str:
        del timestamp, word_budget
        # Fixed accounting so tests have something concrete to assert on.
        ledger.record(uncompressed_tokens=100, compressed_tokens=25, completion_tokens=0)
        return "a lecture hall with a whiteboard"

    def draft(prompt: str, budget: int) -> str:
        del prompt
        sentence = "The lecturer stands beside a whiteboard covered in equations."
        words = sentence.split()
        return " ".join(words[: max(budget, 0)])

    def synthesize(text: str) -> bytes:
        return b"OFFLINE_AUDIO" + text.encode("utf-8")

    def store_bytes(data: bytes, namespace: str) -> tuple[str, str]:
        del namespace
        digest = hashlib.sha256(data).hexdigest()
        return "offline/" + digest, digest

    return Providers(
        transcribe=transcribe,
        scene_context=scene_context,
        draft=draft,
        synthesize=synthesize,
        store=store_bytes,
    )


if __name__ == "__main__":
    # Runs with no API keys and no network. Exercises only pure/offline paths.
    ledger = TokenLedger()
    providers = build_offline_providers(ledger)

    transcript = providers.transcribe("file:///nowhere/lecture.mp4")
    assert isinstance(transcript, TranscriptInput)
    assert transcript.text == "welcome to lecture"
    assert transcript.word_spans == [(0.0, 1.0), (1.0, 2.0)]
    assert transcript.duration == 20.0

    scene = providers.scene_context(3.5, 12)
    assert scene == "a lecture hall with a whiteboard"
    summary = ledger.summary()
    assert isinstance(summary, dict) and summary, "ledger recorded nothing"

    drafted = providers.draft("any prompt", 4)
    assert isinstance(drafted, str)
    assert len(drafted.split()) == 4, drafted
    assert providers.draft("any prompt", 0) == ""

    audio = providers.synthesize("hello")
    assert isinstance(audio, bytes)
    assert audio.startswith(b"OFFLINE_AUDIO")
    assert audio.endswith(b"hello")

    stored = providers.store(audio, "description-audio")
    assert isinstance(stored, tuple) and len(stored) == 2
    key, sha = stored
    assert isinstance(key, str) and isinstance(sha, str)
    assert key == "offline/" + sha
    assert len(sha) == 64
    assert providers.store(audio, "description-audio") == stored, "not deterministic"

    frame = StubFrameLoader()(0.0)
    assert isinstance(frame, bytes) and len(frame) > 0
    assert frame.startswith(b"\xff\xd8") and frame.endswith(b"\xff\xd9")
    assert StubFrameLoader()(99.0) == frame, "stub must be timestamp-invariant"

    assert DEFAULT_VOICE_ID == "21m00Tcm4TlvDq8ikWAM"
    assert "description-audio" in NAMESPACES

    print("wiring self-check OK")
