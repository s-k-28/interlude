"""Measured token-reduction report for Interlude's audio-description pipeline.

WHAT THIS SCRIPT IS
-------------------
This is a measurement harness, not an estimator. Every "before" number
printed here is produced by tokenizing a real, fully-built prompt with the
local tokenizer exposed through ``count_tokens``. No network call, no API
key, and no model invocation is required: tokenization happens on this
machine, so the "before" count is exact for the prompt text that Interlude
would actually send.

WHAT IS MEASURED
----------------
* Prefix arm (MEASURED): the repeated ``STYLE_GUIDE`` prefix that Interlude
  prepends to every scene call is compressed by ``compress_prefix``. Both
  the uncompressed and compressed token counts come from the tokenizer.
* Client arm (MEASURED WHEN AVAILABLE): ``compress_prompt`` performs
  client-side compression. Offline it may be disabled or may decline to
  compress; in that case it reports ``applied=False`` and the compressed
  count equals the original. Those samples are still recorded, at zero
  saving.

WHAT IS NOT MEASURED
--------------------
* No live model is called, so completion tokens are recorded as zero.
* The library-wide figure at the bottom is a PROJECTION: it is the measured
  per-call saving multiplied out by an assumed library shape. It is
  labelled as such and its arithmetic is printed so a reader can check it.

CORRECTNESS PROPERTY
--------------------
Samples whose compression did not apply are never dropped. Dropping them
would inflate the reported ratio. Every sample built is recorded.
"""

from __future__ import annotations

from app.adapters.paritok_adapter import (
    CompressionOutcome,
    compress_prefix,
    count_tokens,
    sdk_version,
)
from app.adapters.paritok_client import (
    ClientCompression,
    compress_prompt,
    is_disabled,
)
from app.adapters.gemini_adapter import STYLE_GUIDE, build_scene_prompt
from app.pipeline.ledger2 import DualLedger

WIDTH: int = 78
SAMPLE_COUNT: int = 12

SCENE_CONTEXTS: list[str] = [
    "A lecturer stands beside a projected diagram of an enzyme.",
    "A hand writes a balanced chemical equation on a whiteboard.",
    "A slide lists four causes of the 1929 market crash in bullets.",
    "The camera pans across a lecture hall of seated students.",
    "A professor rotates a ball-and-stick model of a benzene ring.",
    "An animation shows blood flowing through a four-chambered heart.",
    "A scatter plot of housing prices appears with a fitted line.",
    "A demonstrator pours blue liquid into a graduated cylinder.",
    "A map of trade routes across the Indian Ocean fades in.",
    "Code on a terminal scrolls as a sorting algorithm runs.",
    "A close-up of calipers measuring a metal specimen in a lab.",
    "A timeline of the Roman Republic stretches across the slide.",
]

WORD_BUDGETS: list[int] = [
    18, 22, 25, 28, 30, 32, 35, 38, 40, 45, 50, 60,
]


def _rule(char: str = "-") -> str:
    """Return a horizontal rule that fits an 80-column terminal."""
    return char * WIDTH


def measure_prefix_arm(word_budgets: list[int]) -> tuple[DualLedger, dict]:
    """Measure the repeated-prefix arm.

    For each word budget, build the real scene prompt, tokenize it locally
    (the exact "before"), compress the prefix, and record BOTH counts.
    Samples where compression did not apply are recorded with the
    compressed count equal to the original count, never skipped.
    """
    ledger: DualLedger = DualLedger()
    applied_count: int = 0
    skipped_count: int = 0
    original_total: int = 0
    compressed_total: int = 0

    for budget in word_budgets:
        prompt: str = build_scene_prompt(budget)
        original_tokens: int = count_tokens(prompt)
        outcome: CompressionOutcome = compress_prefix(prompt)

        if outcome.applied:
            compressed_tokens: int = outcome.compressed_tokens
            applied_count += 1
        else:
            compressed_tokens = original_tokens
            skipped_count += 1

        original_total += original_tokens
        compressed_total += compressed_tokens

        ledger.record(
            "prefix",
            uncompressed_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            completion_tokens=0,
        )

    diagnostics: dict = {
        "arm": "prefix",
        "samples": len(word_budgets),
        "applied": applied_count,
        "not_applied": skipped_count,
        "original_total": original_total,
        "compressed_total": compressed_total,
        "saved_total": max(0, original_total - compressed_total),
    }
    return ledger, diagnostics


def measure_client_arm(prompts: list[str]) -> tuple[DualLedger, dict]:
    """Measure the client-side compression arm.

    Same discipline as the prefix arm: every prompt is tokenized locally,
    and prompts whose compression did not apply are recorded at zero
    saving rather than discarded.
    """
    ledger: DualLedger = DualLedger()
    applied_count: int = 0
    skipped_count: int = 0
    original_total: int = 0
    compressed_total: int = 0
    items_compressed_total: int = 0

    for prompt in prompts:
        original_tokens: int = count_tokens(prompt)
        _compressed_text, result = compress_prompt(prompt)

        if isinstance(result, ClientCompression) and result.applied:
            compressed_tokens: int = result.compressed_tokens
            items_compressed_total += result.items_compressed
            applied_count += 1
        else:
            compressed_tokens = original_tokens
            skipped_count += 1

        original_total += original_tokens
        compressed_total += compressed_tokens

        ledger.record(
            "client",
            uncompressed_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            completion_tokens=0,
        )

    diagnostics: dict = {
        "arm": "client",
        "samples": len(prompts),
        "applied": applied_count,
        "not_applied": skipped_count,
        "items_compressed": items_compressed_total,
        "original_total": original_total,
        "compressed_total": compressed_total,
        "saved_total": max(0, original_total - compressed_total),
    }
    return ledger, diagnostics


def library_projection(
    per_scene_saved: float,
    *,
    videos: int = 12000,
    gaps_per_video: int = 15,
    attempts_per_gap: float = 1.4,
) -> dict:
    """Project the measured per-call saving across a whole library.

    This is arithmetic on top of a measurement. It is a PROJECTION and is
    labelled as such everywhere it is printed.
    """
    total_calls: float = float(videos) * float(gaps_per_video) * attempts_per_gap
    projected_tokens_saved: float = total_calls * float(per_scene_saved)
    return {
        "videos": videos,
        "gaps_per_video": gaps_per_video,
        "attempts_per_gap": attempts_per_gap,
        "per_scene_saved": float(per_scene_saved),
        "total_calls": total_calls,
        "projected_tokens_saved": projected_tokens_saved,
    }


def _build_client_prompts() -> list[str]:
    """Build the client-arm sample prompts: style guide + scene context."""
    prompts: list[str] = []
    for index, scene in enumerate(SCENE_CONTEXTS):
        budget: int = WORD_BUDGETS[index % len(WORD_BUDGETS)]
        base: str = build_scene_prompt(budget)
        prompts.append(f"{STYLE_GUIDE}\n\n{base}\n\nScene: {scene}")
    return prompts


def _print_diagnostics(diagnostics: dict) -> None:
    """Print one arm's per-sample accounting."""
    print(f"  arm                 : {diagnostics['arm']}")
    print(f"  samples recorded    : {diagnostics['samples']}")
    print(f"  compression applied : {diagnostics['applied']}")
    print(f"  not applied (0 save): {diagnostics['not_applied']}")
    print(f"  tokens before (sum) : {diagnostics['original_total']}")
    print(f"  tokens after  (sum) : {diagnostics['compressed_total']}")
    print(f"  tokens saved  (sum) : {diagnostics['saved_total']}")


def main() -> int:
    """Print the measured report. Always returns 0; this is not a test."""
    print(_rule("="))
    print("INTERLUDE - MEASURED TOKEN REDUCTION REPORT")
    print(_rule("="))
    print(f"Paritok SDK version : {sdk_version()}")
    print(f"PARITOK_DISABLE set : {is_disabled()}")
    print(f"Samples per arm     : {SAMPLE_COUNT}")
    print("Tokenization        : local, no network, no API key required")
    print()

    prefix_ledger, prefix_diag = measure_prefix_arm(WORD_BUDGETS)
    client_prompts: list[str] = _build_client_prompts()
    client_ledger, client_diag = measure_client_arm(client_prompts)

    print(_rule())
    print("MEASURED RESULTS - PREFIX ARM")
    print(_rule())
    print(prefix_ledger.render_table())
    _print_diagnostics(prefix_diag)
    print()

    print(_rule())
    print("MEASURED RESULTS - CLIENT ARM")
    print(_rule())
    print(client_ledger.render_table())
    _print_diagnostics(client_diag)
    print()

    print(_rule())
    print("METHODOLOGY")
    print(_rule())
    print(f"* {SAMPLE_COUNT} samples per arm; every sample is recorded.")
    print("* 'before' = tokens in the full prompt Interlude would send,")
    print("  counted by the local tokenizer. It is exact, not estimated.")
    print("* 'after'  = tokens after the SDK compressed that same prompt,")
    print("  counted by the same local tokenizer.")
    print("* Compressions that did NOT apply are included with")
    print("  after == before, i.e. zero saving. They are never dropped;")
    print("  dropping them would inflate the reported ratio.")
    print("* No model was called, so completion tokens are zero.")
    print()

    measured_saved: int = int(prefix_diag["saved_total"])
    samples: int = max(1, int(prefix_diag["samples"]))
    per_scene_saved: float = measured_saved / samples
    projection: dict = library_projection(per_scene_saved)

    print(_rule())
    print("LIBRARY PROJECTION (NOT A MEASUREMENT)")
    print(_rule())
    print("Derived from the measured prefix-arm per-call saving.")
    print(f"  measured saved tokens ({samples} calls) : {measured_saved}")
    print(f"  per-call saving = {measured_saved} / {samples}"
          f" = {per_scene_saved:.2f} tokens")
    print(f"  videos                           : {projection['videos']}")
    print(f"  gaps per video                   : {projection['gaps_per_video']}")
    print(f"  attempts per gap                 : {projection['attempts_per_gap']}")
    print(f"  total calls = {projection['videos']}"
          f" x {projection['gaps_per_video']}"
          f" x {projection['attempts_per_gap']}"
          f" = {projection['total_calls']:,.0f}")
    print(f"  projected saved = {projection['total_calls']:,.0f}"
          f" x {per_scene_saved:.2f}"
          f" = {projection['projected_tokens_saved']:,.0f} tokens")
    print("  The call-volume inputs above are ASSUMPTIONS about library")
    print("  shape. Only the per-call saving is measured.")
    print()

    print(_rule("="))
    print("HONEST LIMITATIONS - WHAT WAS NOT MEASURED")
    print(_rule("="))
    print("1. No live model call was made. Latency, cost, and output")
    print("   quality are unmeasured here.")
    print("2. Completion (output) tokens are not included; only prompt")
    print("   tokens are counted, so total-spend savings will differ.")
    print("3. The client arm may need a backend that is absent offline.")
    print("   If it reported 0 applied above, that arm measured no")
    print("   reduction on this machine - a real result, not hidden.")
    print("4. If the tokenizer is unavailable, count_tokens falls back to")
    print("   a len//4 estimate. Check the SDK version line above before")
    print("   quoting these numbers as exact.")
    print("5. The library projection is arithmetic on assumed volumes,")
    print("   not an observed library-wide total.")
    print("6. Scene contexts are representative samples, not a random")
    print("   draw from a production lecture corpus.")
    print(_rule("="))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
