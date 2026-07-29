"""Text-only completion. Closes the CONTRACT DELTA reported by the wiring agent.

The description retry loop needs to send a fully-built prompt string and get
text back, with token counts for both compression arms. `gemini_adapter`
handles vision->text and always requires a frame; this handles text->text.

Kept separate from gemini_adapter deliberately: that module's contract is
"describe this image", and overloading it with a text-only path would blur the
one boundary that makes the vision integration auditable.

Per the Unverified-API Protocol, no module outside app/adapters imports
google.genai.
"""

from __future__ import annotations

import logging
import os

from app.adapters.paritok_adapter import compress_prefix

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-2.0-flash"


class TextCompletionUnavailable(RuntimeError):
    """Raised when the SDK or GOOGLE_API_KEY is absent."""


# ==== UNVERIFIED API — CONFIRM BEFORE RUNNING ====
# Assumed:    google.genai.Client(api_key=...).models.generate_content(
#                 model=..., contents=[str]) -> object with .text and
#                 .usage_metadata.{prompt_token_count, candidates_token_count}
# Basis:      google-genai 1.75.0 is installed as a genblaze-google dependency;
#             same call shape already assumed in gemini_adapter.describe_frame.
# Confidence: medium-high. Identical shape to the vision path, minus the Part.
# Blast radius: this file only.
def complete_text(
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    compress: bool = True,
) -> tuple[str, int, int]:
    """Complete a text prompt.

    Args:
        prompt: The fully-built prompt. Callers own prompt construction.
        model: Gemini model id.
        api_key: Overrides GOOGLE_API_KEY when supplied.
        compress: Route the prompt through Paritok first. The retry loop resends
            the same style-guide prefix on every attempt, so compression applies
            here as much as it does on the vision path.

    Returns:
        ``(text, prompt_tokens_uncompressed, completion_tokens)``.

        The first token count is deliberately the *uncompressed* figure. The
        caller records both arms in the ledger, and reporting the compressed
        number here would silently understate what the workload actually costs
        without Paritok — which is the comparison the measurement exists to make.
    """
    if not prompt.strip():
        raise ValueError("prompt must not be empty")

    key = api_key or os.environ.get("GOOGLE_API_KEY", "").strip()
    if not key:
        raise TextCompletionUnavailable("GOOGLE_API_KEY is not set")

    outcome = compress_prefix(prompt) if compress else None
    payload = outcome.text if outcome is not None else prompt
    uncompressed = outcome.original_tokens if outcome is not None else 0

    try:
        from google import genai
    except ImportError as exc:
        raise TextCompletionUnavailable(f"google-genai not installed: {exc}") from exc

    client = genai.Client(api_key=key)
    try:
        response = client.models.generate_content(model=model, contents=[payload])
    except Exception as exc:  # noqa: BLE001
        raise TextCompletionUnavailable(f"text completion failed: {exc}") from exc

    text = str(getattr(response, "text", "") or "").strip()
    if not text:
        raise TextCompletionUnavailable("model returned an empty completion")

    usage = getattr(response, "usage_metadata", None)
    completion_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)

    if uncompressed == 0:
        uncompressed = int(getattr(usage, "prompt_token_count", 0) or 0)

    return text, uncompressed, completion_tokens


def compressed_token_count(prompt: str) -> int:
    """The compressed arm, for callers recording both sides of the measurement."""
    return compress_prefix(prompt).compressed_tokens


def make_drafter(
    *,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    ledger=None,
):
    """Adapt :func:`complete_text` to the retry loop's ``Drafter`` contract.

    The loop expects ``Callable[[str, int], str]`` — prompt and word budget in,
    text out. The budget is already embedded in the prompt by ``build_prompt``,
    so it is accepted and ignored here rather than re-appended, which would
    duplicate the instruction and confuse the model.
    """

    def draft(prompt: str, _word_budget: int) -> str:
        text, uncompressed, completion = complete_text(
            prompt, model=model, api_key=api_key
        )
        if ledger is not None:
            ledger.record(
                uncompressed_tokens=uncompressed,
                compressed_tokens=compressed_token_count(prompt),
                completion_tokens=completion,
            )
        return text

    return draft


if __name__ == "__main__":
    try:
        complete_text("   ")
        raise AssertionError("blank prompt must be rejected")
    except ValueError:
        pass

    saved = os.environ.pop("GOOGLE_API_KEY", None)
    try:
        complete_text("describe a lecture hall")
        raise AssertionError("missing key must raise")
    except TextCompletionUnavailable:
        pass
    finally:
        if saved is not None:
            os.environ["GOOGLE_API_KEY"] = saved

    assert compressed_token_count("some prompt text") >= 0

    drafter = make_drafter(api_key=None)
    assert callable(drafter)

    print("text_adapter self-check OK")
