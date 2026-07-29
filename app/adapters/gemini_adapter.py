"""Sole entry point for Google Gemini vision->text.

WHY THIS EXISTS SEPARATELY FROM genblaze_adapter:
  genblaze-google 0.3.4 ships only GeminiImageProvider ("google-gemini-image"),
  ImagenProvider ("google-imagen") and VeoProvider ("google-veo") — all three are
  IMAGE/VIDEO GENERATORS. There is no vision-understanding or text provider in
  the package. Interlude needs a model to LOOK at a frame and DESCRIBE it, which
  genblaze cannot route. So this one capability calls google-genai directly.

  google-genai is already an installed dependency of genblaze-google, so this
  adds no new requirement. The boundary is documented honestly in the README:
  genblaze orchestrates transcription, TTS, provenance and storage; it does not
  orchestrate scene description, and we do not claim it does.

Per the Unverified-API Protocol, no other module imports google.genai.

VERIFIED: google-genai 1.75.0 is installed (pulled in by genblaze-google).
UNVERIFIED: exact client construction and response shape — see inline markers.
"""

from __future__ import annotations

import base64
import logging
import os
from dataclasses import dataclass

from app.adapters.paritok_adapter import CompressionOutcome, compress_prefix

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-2.0-flash"

# The style guide is prepended to EVERY scene call. Across a 12,000-video
# library that is the same block retransmitted hundreds of thousands of times —
# which is exactly the repeated-prefix workload Paritok compresses.
STYLE_GUIDE = """You write audio description for blind and low-vision listeners.

Rules:
- Describe only what is visually present. Never interpret motive or emotion.
- Never describe sound, dialogue, or music; the listener already hears those.
- Present tense, third person, no addressing the viewer.
- Name people by role or appearance, not by guessed identity.
- Prioritise: who is present, what changed, where it happens, on-screen text.
- Plain language. No metaphor. No "we see" or "the camera shows".
"""


class GeminiUnavailable(RuntimeError):
    """Raised when the SDK or GOOGLE_API_KEY is absent."""


@dataclass(frozen=True, slots=True)
class SceneDescription:
    """A model's account of one video frame."""

    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    compression: CompressionOutcome | None = None

    @property
    def compressed(self) -> bool:
        return self.compression is not None and self.compression.applied


def build_scene_prompt(word_budget: int, *, timestamp: float = 0.0) -> str:
    """Assemble the full prompt: style guide plus the per-scene ask."""
    if word_budget <= 0:
        raise ValueError(f"word_budget must be positive, got {word_budget}")
    return (
        f"{STYLE_GUIDE}\n"
        f"This frame occurs at {timestamp:.1f} seconds.\n"
        f"Describe it in at most {word_budget} words."
    )


# ==== UNVERIFIED API — CONFIRM BEFORE RUNNING ====
# Assumed:    google.genai.Client(api_key=...) with
#             client.models.generate_content(model=..., contents=[...])
#             returning an object exposing .text
# Basis:      google-genai 1.75.0 is the current published SDK shape; the
#             genblaze-google provider imports `from google import genai`.
# Confidence: medium-high
# Blast radius: this file only.
def describe_frame(
    image_bytes: bytes,
    word_budget: int,
    *,
    timestamp: float = 0.0,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    mime_type: str = "image/jpeg",
) -> SceneDescription:
    """Describe a single video frame.

    The style-guide prefix is routed through Paritok before the call, and both
    token counts are returned so the saving is measured rather than asserted.
    """
    if not image_bytes:
        raise ValueError("image_bytes must not be empty")

    key = api_key or os.environ.get("GOOGLE_API_KEY", "").strip()
    if not key:
        raise GeminiUnavailable("GOOGLE_API_KEY is not set")

    prompt = build_scene_prompt(word_budget, timestamp=timestamp)
    outcome = compress_prefix(prompt)

    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise GeminiUnavailable(f"google-genai not installed: {exc}") from exc

    client = genai.Client(api_key=key)
    try:
        response = client.models.generate_content(
            model=model,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                outcome.text,
            ],
        )
    except Exception as exc:  # noqa: BLE001
        raise GeminiUnavailable(f"gemini call failed: {exc}") from exc

    text = str(getattr(response, "text", "") or "").strip()
    if not text:
        raise GeminiUnavailable("gemini returned an empty description")

    usage = getattr(response, "usage_metadata", None)
    return SceneDescription(
        text=text,
        model=model,
        prompt_tokens=int(getattr(usage, "prompt_token_count", 0) or 0),
        completion_tokens=int(getattr(usage, "candidates_token_count", 0) or 0),
        compression=outcome,
    )


def make_scene_describer(
    frame_loader,
    *,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
):
    """Adapt :func:`describe_frame` to the runner's SceneDescriber contract.

    The runner expects ``Callable[[float, int], str]`` — timestamp and word
    budget in, scene context out. ``frame_loader`` supplies the frame bytes for
    a timestamp, keeping frame extraction out of this adapter.
    """

    def describe(timestamp: float, word_budget: int) -> str:
        frame = frame_loader(timestamp)
        return describe_frame(
            frame, word_budget, timestamp=timestamp, model=model, api_key=api_key
        ).text

    return describe


def encode_data_uri(image_bytes: bytes, mime_type: str = "image/jpeg") -> str:
    """Inline a frame for the review UI. Used by the API layer, not by Gemini."""
    return f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode('ascii')}"


def sdk_version() -> str:
    try:
        from importlib.metadata import version

        return version("google-genai")
    except Exception:  # noqa: BLE001
        return "unknown"


if __name__ == "__main__":
    p = build_scene_prompt(12, timestamp=3.5)
    assert "at most 12 words" in p
    assert "3.5 seconds" in p
    assert "Never interpret motive" in p

    try:
        build_scene_prompt(0)
        raise AssertionError("zero budget must be rejected")
    except ValueError:
        pass

    d = SceneDescription(text="A man writes on a whiteboard.", model="m")
    assert not d.compressed

    from app.adapters.paritok_adapter import CompressionOutcome as CO

    d2 = SceneDescription(text="x", model="m", compression=CO("x", 100, 25, applied=True))
    assert d2.compressed
    assert d2.compression is not None and d2.compression.saved_tokens == 75

    uri = encode_data_uri(b"\xff\xd8\xff")
    assert uri.startswith("data:image/jpeg;base64,")

    os.environ.pop("GOOGLE_API_KEY", None)
    try:
        describe_frame(b"\xff\xd8", 10)
        raise AssertionError("missing key must raise")
    except GeminiUnavailable:
        pass
    except ValueError:
        raise AssertionError("wrong error for missing key")

    print(f"gemini_adapter self-check OK (google-genai={sdk_version()})")
