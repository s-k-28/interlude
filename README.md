# Interlude

**Audio description for institutional video libraries.**

A university with 12,000 lecture recordings is legally required to make them
accessible to blind and low-vision students. Human audio description costs
$15–25 per minute, so a single 50-minute lecture runs $750–1,250 and the full
library runs into eight figures. Most institutions therefore describe nothing
and remain out of compliance.

Interlude describes the library overnight, and leaves behind a cryptographic
record of how every second of narration was produced.

---

## The problem, precisely

Blind users cannot follow a video without a *described* audio track: a second
narration that explains visual action, spoken in the silences between dialogue.

That constraint is what makes the problem interesting. A description is not
free text — it must fit a specific silence window, or it collides with the next
line of speech and the listener loses both. A 2.4-second gap at natural
narration pace holds six words. Not seven.

So the system has to do three things at once: find the gaps, understand what is
on screen, and write to a hard length budget that changes for every gap.

---

## Pipeline

```
  source video
       │
       ▼
  ┌─────────────────┐
  │  1. TRANSCRIBE  │  AssemblyAI via Genblaze → word-level timings (seconds)
  └────────┬────────┘
           ▼
  ┌─────────────────┐
  │  2. DETECT GAPS │  silences ≥ 1.4s; word budget = duration × 165wpm ÷ 60
  └────────┬────────┘
           ▼
  ┌─────────────────┐
  │  3. DESCRIBE    │  frame → Gemini vision → draft
  │     ⟲ retry     │  over budget? redraft with the measured overage injected
  └────────┬────────┘
           ▼
  ┌─────────────────┐
  │  4. SYNTHESIZE  │  ElevenLabs via Genblaze → speech per segment
  └────────┬────────┘
           ▼
  ┌─────────────────┐
  │  5. MANIFEST    │  canonical JSON, SHA-256, written to B2
  └─────────────────┘
```

Every stage writes a content-addressed artifact to Backblaze B2 **before** the
next begins, and each artifact records the key of its parent. The result is a
lineage that can be walked and re-verified, not a manifest sitting beside a
finished file with nothing in between.

---

## Architecture

| Layer | Module | Responsibility |
|---|---|---|
| Gap detection | `app/pipeline/gaps.py` | Silence windows, word budgets, overlapping-speaker handling |
| Description | `app/pipeline/describe.py` | Fit-constrained drafting loop |
| Orchestration | `app/pipeline/orchestrator.py` | Genblaze-driven execution, provenance chain |
| Legacy runner | `app/pipeline/runner.py` | Stage bodies, resumable job state |
| Provenance | `app/pipeline/manifest.py` | Canonical serialization, hash verification |
| Token accounting | `app/pipeline/ledger2.py` | Dual-arm compression measurement |
| Frame extraction | `app/pipeline/frames.py` | ffmpeg, input-seek, JPEG validation |
| Composition root | `app/pipeline/wiring.py` | Assembles providers from adapters |
| Object storage | `app/storage/store.py` | B2 client, dedup, integrity |
| Offline storage | `app/storage/local.py` | Same contract, filesystem-backed |
| Content addressing | `app/storage/keys.py` | SHA-256 keys, sharded prefixes |
| HTTP API | `app/main.py`, `app/api/` | Six endpoints, background jobs |
| Frontend | `web/` | Vanilla JS dashboard, no build step |

**Adapter boundary.** No module outside `app/adapters/` imports a vendor SDK.
Five adapters — `genblaze_adapter`, `genblaze_pipeline`, `gemini_adapter`,
`text_adapter`, `paritok_adapter`, `paritok_client` — are the only files that
know a third-party API exists. Everything crossing that boundary is a plain
dataclass defined locally. When an SDK's real surface turned out to differ from
our assumption (twice, see below), the blast radius was one file.

---

## The fit-constrained retry loop

This is the technical centre of the product.

```python
draft = model(prompt)
if word_count(draft) > gap.word_budget:
    # Tell the model exactly how far over it went. "Make it shorter" produces
    # materially worse second attempts than "cut 4 words".
    draft = model(prompt_with_overage(draft, overage=word_count(draft) - budget))
```

Three attempts maximum. If all three overrun, the text is truncated on a word
boundary rather than discarded — a clean partial description is useful; a
description clipped mid-syllable by dialogue is not.

Every attempt is retained in the manifest. A description accepted on the third
try is a different provenance claim than one accepted on the first, and an
auditor needs to see the rejected drafts.

---

## Provenance

Every artifact is stored under a key derived from the SHA-256 of its bytes:

```
{namespace}/{hash[0:2]}/{hash[2:4]}/{hash}{ext}
```

Consequences:

- **Identical bytes are stored once.** Regenerating the same description costs
  nothing and occupies no additional space.
- **The key is a verifiable claim about the content.** The digest travels in
  object metadata, so integrity is re-checkable on read without re-hashing the
  payload out of band.
- **The chain is walkable.** `ChainedResult.verify(store)` re-reads every link
  from source video to spoken audio and returns the keys that failed.

Manifests are the single exception — addressed by run id at
`manifests/{run_id}.json`, because a manifest must be locatable before its own
hash is known.

**Canonical serialization matters.** Two runs producing identical work must
produce byte-identical manifests, or the hash proves nothing. Keys are sorted,
separators pinned, and NaN/Infinity rejected — they are not valid JSON and
would produce a document no other parser could verify.

---

## Sponsor technology

### Backblaze B2

B2 is not the output folder. It is the substrate.

Sources, transcripts, gap sets, individual descriptions, rendered audio, and
provenance manifests are all content-addressed objects. Intermediate artifacts
are kept permanently, not discarded after use — which is what makes the audit
trail possible, and which is only economically sensible on storage priced like
B2's.

Accessed through `app/storage/store.py` using the S3-compatible API with
path-style addressing pinned, adaptive retry, and digest verification on read.

### Genblaze

Genblaze owns pipeline orchestration. Each stage is a `SyncProvider` subclass
whose `generate()` calls the underlying stage function, so the SDK controls
sequencing, retry and failover while our code retains the logic.

Used for transcription (AssemblyAI provider, word-level timings in seconds) and
speech synthesis (ElevenLabs provider).

> **Honest boundary.** `genblaze-google` 0.3.4 ships `GeminiImageProvider`,
> `ImagenProvider` and `VeoProvider` — all image or video *generators*. It does
> not expose a vision-to-text model. Interlude's scene-description step
> therefore calls `google-genai` directly through `app/adapters/gemini_adapter.py`
> rather than through Genblaze. Genblaze orchestrates transcription, synthesis
> and storage; it does not orchestrate scene description, and we do not claim
> that it does.

### Paritok

The audio-description style guide is prepended to every scene call. Across a
12,000-video library that same block is retransmitted hundreds of thousands of
times — a genuine repeated-prefix workload.

Two independent compression arms, measured and reported **separately**:

| Arm | What it compresses | Where |
|---|---|---|
| `prefix` | The style-guide block on every prompt | `paritok_adapter.compress_prefix()` |
| `client` | Message history through the SDK's own pipeline | `paritok_client.compress_messages()` |

Blending them into one headline number would make it impossible to tell which
optimization worked. Raw before/after counts are reported alongside the ratio so
a reader can recompute it by hand.

Run `python measure_tokens.py` for the current measurement. Tokenization is
local and exact. **Unsuccessful compressions are recorded at zero saving, not
dropped** — discarding them would silently inflate the reported reduction.

---

## Configuration

All credentials are environment variables. Nothing is committed; `.env` is
gitignored and `.env.example` documents every key.

| Variable | Service | Required | Purpose |
|---|---|---|---|
| `B2_KEY_ID` | Backblaze B2 | yes | Application key id |
| `B2_APP_KEY` | Backblaze B2 | yes | Application key secret |
| `B2_BUCKET` | Backblaze B2 | yes | Target bucket |
| `B2_ENDPOINT` | Backblaze B2 | yes | e.g. `https://s3.us-east-005.backblazeb2.com` |
| `ASSEMBLYAI_API_KEY` | AssemblyAI | for ingest | Transcription and word timings |
| `GOOGLE_API_KEY` | Google AI Studio | for ingest | Gemini vision-to-text |
| `ELEVENLABS_API_KEY` | ElevenLabs | for ingest | Speech synthesis |
| `PARITOK_API_KEY` | Paritok | optional | Hosted compression |
| `PARITOK_DISABLE` | — | optional | Kill switch; bypasses compression entirely |
| `MAX_STEP_RETRIES` | — | optional | Default 3 |

`GET /health` reports which provider keys are absent and never raises, so the
service starts and stays diagnosable with partial configuration.

---

## Running it

```bash
bash bootstrap.sh          # install, repair, run self-checks, run tests
cp .env.example .env       # then fill in credentials
uvicorn app.main:app --reload --port 8000
open web/index.html
```

**Offline demonstration**, no credentials required:

```bash
python run_demo.py         # full pipeline, artifacts to ./artifacts, chain verified
python measure_tokens.py   # measured token reduction
```

`LocalStore` implements the same contract as `B2Store` with byte-identical keys,
so a provenance chain built offline remains valid after the artifacts are
uploaded to B2.

---

## Production considerations

**Resumability.** A library job is thousands of videos and hours of wall clock.
`JobState.resume_from` returns the first incomplete stage; a run that dies at
video 340 of 400 restarts at 340. Re-running a stage replaces its record rather
than appending, so resume does not corrupt the audit trail.

**Partial success is a real outcome.** If synthesis fails on 2 of 20 segments,
the job returns `PARTIAL` with 18 usable descriptions. A single failure never
discards the rest, and `Runner.run()` does not raise — failures land on the job
object, because a 400-video batch must not abort on one malformed file.

**Graceful degradation.** Paritok absent or disabled → the pipeline runs
uncompressed. Compression is an optimization, never a single point of failure.

**Integrity on read.** Every `get()` verifies the payload against its stored
digest and raises on mismatch rather than returning corrupt bytes.

**Permissions are not absence.** A 403 from the object store is never treated
as "object missing" — that mistake would cause silent regeneration of assets
the system simply cannot read.

---

## Testing

```bash
pytest -q     # 255 tests
```

Every external dependency is injected, so the full suite runs offline with no
credentials and no network. Each module additionally carries a runnable
self-check:

```bash
python -m app.adapters.paritok_adapter
python -m app.adapters.genblaze_adapter
python -m app.adapters.gemini_adapter
python -m app.storage.local
python -m app.pipeline.ledger2
```

---

## What we found by reading the SDK source

Two integration defects that a green test suite did not catch. Both are written
up in `PARITOK_FEEDBACK.md`.

**Paritok's entry point is not `compress()`.** Our adapter assumed
`ParitokEngine.compress(text)`. No such method exists — the real entry point is
`process_request(messages, tools, upstream_model)`, which takes a message list.
Because the adapter degraded gracefully on failure, every call landed silently
in its `except` branch and the integration did nothing at all, while every test
stayed green. Passing tests proved the plumbing, not the product.

**Genblaze ships no Gemini vision-to-text provider.** Documented above. Found by
reading the wheel, not by trusting the package name.

Both were found the same way: extracting the published wheels and reading the
source rather than relying on recall or inference.

---

## License

Apache 2.0. See `LICENSE`.
