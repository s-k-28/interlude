# Interlude

Interlude generates described audio tracks for university lecture video libraries so blind and low-vision students can actually use them.

Blind students cannot access a video without a described audio track that narrates what is on screen. Human description runs $15–25 per minute, so a single 50-minute lecture costs $750–1,250 to describe. A university sitting on 12,000 lecture videos therefore describes none of them and stays out of compliance.

## What it does

1. Transcribe the lecture audio and get word-level timings.
2. Find every silence gap of at least 1.4 seconds between spoken words.
3. Compute how many words fit in each gap at 165 words per minute.
4. Have a vision model describe the frame at that timestamp.
5. Check the draft against the word budget and redraft with the measured overage fed back if it overruns.
6. Synthesize speech for the accepted description.
7. Write every artifact to Backblaze B2 under a content-addressed key with a provenance manifest.

## Architecture

| Layer | Module | Responsibility |
| --- | --- | --- |
| Gap detection | `app/pipeline/gaps.py` | Locate silence gaps ≥ 1.4s and size the word budget at 165 wpm |
| Description loop | `app/pipeline/describe.py` | Draft, measure against the budget, redraft with overage fed back |
| Orchestration | `app/pipeline/runner.py` | Sequence the steps, retries, resume, partial success |
| Provenance | `app/pipeline/manifest.py` | Record what was produced, from what, and by which provider |
| Token accounting | `app/pipeline/tokens.py` | Record uncompressed and compressed prompt tokens per call |
| Storage | `app/storage/store.py` | Content-addressed writes and integrity-verified reads on B2 |
| HTTP API | `app/main.py` | FastAPI app exposing job creation, status and manifests |
| Frontend | `web/` | Vanilla HTML/CSS/JS UI, no build step |

## Sponsor technology

### Backblaze B2

Every artifact — source, transcript, each description, each audio clip, and the manifest — is stored under a SHA-256 content-addressed key of the form `{namespace}/{hash[0:2]}/{hash[2:4]}/{hash}{ext}`. Identical bytes are never stored twice. Each object carries its digest in metadata so integrity is re-verifiable without re-hashing the payload. Manifests are the one exception, addressed by run id at `manifests/{run_id}.json`, because they must be locatable before their own hash is known.

### Genblaze

Used for transcription (AssemblyAI provider, word-level timings in seconds) and speech synthesis (ElevenLabs provider). Accessed only through `app/adapters/genblaze_adapter.py`.

### Honesty note

"genblaze-google 0.3.4 ships GeminiImageProvider, ImagenProvider and VeoProvider — all image or video generators. It does not expose a vision-to-text model. Interlude's scene-description step therefore calls google-genai directly through app/adapters/gemini_adapter.py rather than through Genblaze. Genblaze orchestrates transcription, synthesis and storage; it does not orchestrate scene description, and we do not claim that it does."

### Paritok

The audio-description style guide is prepended to every scene call. Across a 12,000-video library that same block is retransmitted hundreds of thousands of times, which is the repeated-prefix workload Paritok compresses. Both arms are measured — uncompressed and compressed prompt tokens are recorded per call — so the reduction is a measurement, not a claim. Raw counts are reported alongside the ratio so a reader can recompute it. Accessed only through `app/adapters/paritok_adapter.py`.

## Setup

Works in a GitHub Codespace or locally.

```bash
bash bootstrap.sh
# fill in .env
uvicorn app.main:app --reload --port 8000
```

Then open `web/index.html`.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Liveness check |
| POST | `/jobs` | Create a description job |
| GET | `/jobs` | List jobs |
| GET | `/jobs/{id}` | Fetch a single job |
| POST | `/jobs/{id}/resume` | Resume a job from where it stopped |
| GET | `/jobs/{id}/manifest` | Fetch the provenance manifest for a run |

## Production considerations

- **Resumable jobs.** A run that dies at video 340 of 400 restarts at 340, not zero.
- **Partial success is a real outcome.** 18 of 20 descriptions rendered is shippable; one failure never discards the rest.
- **Content-addressed deduplication.** Identical bytes are stored once.
- **Integrity verification on read.** The stored digest is checked without re-hashing the payload.
- **Graceful degradation.** The pipeline runs when Paritok is absent or disabled.
- **Injected dependencies.** Every external dependency is injected, so the whole suite runs offline with no API keys.

## Testing

```bash
pytest -q
```

Every provider is injected, so the suite needs no network access and no credentials.

## License

Apache 2.0.
