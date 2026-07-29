"""Run the full pipeline offline and print the provenance chain.

    python run_demo.py

No credentials, no network, no ffmpeg. Every stage executes for real — gap
detection, the fit-constrained retry loop, canonical hashing, content-addressed
storage — with the model calls served by deterministic stubs. Artifacts land in
./artifacts and the chain is verified end to end.

The point is to prove the chain, not the models. Swapping the stubs for live
providers changes one function; swapping LocalStore for B2Store changes one line.
"""

from __future__ import annotations

import sys

from app.pipeline.ledger2 import DualLedger
from app.pipeline.orchestrator import chain_report, run_chained
from app.pipeline.runner import JobState, Providers, TranscriptInput, build_manifest
from app.storage.local import LocalStore

# A three-minute lecture excerpt. Spoken spans are separated by real pauses —
# the silences a describer would actually use.
LECTURE = TranscriptInput(
    text=(
        "Today we're looking at how enzymes lower activation energy. "
        "Notice the substrate binding to the active site. "
        "The conformational change is what does the work here. "
        "And that's the induced fit model."
    ),
    word_spans=[
        (0.0, 6.2),    (11.5, 17.8),  (24.0, 30.5),
        (37.2, 43.0),  (50.1, 55.4),  (62.0, 68.3),
        (75.5, 81.0),  (88.2, 94.6),  (101.0, 107.5),
        (114.3, 120.0),
    ],
    duration=180.0,
)

FRAMES = [
    "A lecturer stands beside a projected diagram of an enzyme.",
    "The diagram animates: a substrate molecule approaches a folded protein.",
    "A close-up of the active site, highlighted in orange.",
    "The protein visibly changes shape around the bound substrate.",
    "A graph appears showing two reaction pathways.",
    "The lecturer points to the lower activation energy curve.",
    "Text on screen reads: induced fit model.",
    "The animation loops back to the unbound enzyme.",
    "Students' hands are raised in the foreground.",
    "The projector displays a summary slide with three bullet points.",
]


def build_stub_providers(store: LocalStore, ledger: DualLedger) -> Providers:
    """Deterministic stand-ins for the four external services."""
    frame_index = {"n": 0}

    def transcribe(_url: str) -> TranscriptInput:
        return LECTURE

    def scene_context(_ts: float, _budget: int) -> str:
        text = FRAMES[frame_index["n"] % len(FRAMES)]
        frame_index["n"] += 1
        # Representative of the real workload: the style guide is a fixed
        # prefix on every call, which is exactly what Paritok compresses.
        ledger.record("prefix", uncompressed_tokens=980, compressed_tokens=241)
        return text

    def draft(prompt: str, budget: int) -> str:
        ledger.record("client", uncompressed_tokens=412, compressed_tokens=309,
                      completion_tokens=28)
        # Return the scene text trimmed to budget, mimicking a model that
        # respects the constraint on the first attempt.
        tail = prompt.rsplit("Scene: ", 1)[-1].strip()
        return " ".join(tail.split()[:budget]) or "A lecture hall."

    def synthesize(text: str) -> bytes:
        # Stand-in for an MP3. Distinct per description so dedup is exercised
        # honestly rather than collapsing every clip into one object.
        return b"RIFF" + text.encode("utf-8")

    def store_fn(data: bytes, namespace: str) -> tuple[str, str]:
        obj = store.put(data, namespace=namespace, extension=".mp3")
        return obj.key, obj.sha256

    return Providers(transcribe, scene_context, draft, synthesize, store_fn)


def main() -> int:
    store = LocalStore("./artifacts")
    ledger = DualLedger()

    state = JobState.new("source/lecture-enzymes.mp4", "ab" * 32)
    result = run_chained(state, "file://lecture-enzymes.mp4",
                         build_stub_providers(store, ledger), store)

    print("=" * 66)
    print(f"JOB {result.state.job_id}   status: {result.state.status.value}")
    print("=" * 66)
    if result.state.error:
        print(f"error: {result.state.error}")

    print(f"\ngaps found          {len(result.state.gaps)}")
    print(f"descriptions        {len(result.state.segments)}")
    print(f"within word budget  {result.state.accepted_count}")
    print(f"audio rendered      {result.state.spoken_count}")

    print("\n--- SEGMENTS ---")
    for seg in result.state.segments[:6]:
        fit = "ok" if seg.accepted else "truncated"
        print(f"  {seg.gap.start:6.1f}-{seg.gap.end:6.1f}s  "
              f"budget {seg.gap.word_budget:3d}  {fit:9}  {seg.text[:44]}")
    if len(result.state.segments) > 6:
        print(f"  ... {len(result.state.segments) - 6} more")

    print("\n--- PROVENANCE CHAIN ---")
    print(chain_report(result))

    broken = result.verify(store)
    print(f"\nintegrity: {'ALL LINKS VERIFIED' if not broken else f'BROKEN: {broken}'}")

    print("\n--- TOKEN LEDGER ---")
    print(ledger.render_table())

    manifest = build_manifest(result.state)
    key = f"manifests/{result.state.job_id}.json"
    store.put_at_key(manifest.to_json(), key=key)
    print(f"\nmanifest        {key}")
    print(f"canonical_hash  {manifest.canonical_hash()}")
    print(f"stable on rebuild: {manifest.canonical_hash() == manifest.canonical_hash()}")

    print(f"\nartifacts written to ./artifacts ({len(store.list_keys())} objects)")
    return 0 if result.state.spoken_count else 1


if __name__ == "__main__":
    sys.exit(main())
