# Paritok SDK Feedback

We built an audio-description pipeline for university lecture libraries: each lecture video is segmented into scenes, and every scene issues its own vision→text model call carrying the same long style-guide prefix. That repeated prefix across thousands of per-scene calls is exactly the shape Paritok targets, so we integrated `paritok` 1.2.7 (pip, PyPI) as a real dependency rather than a demo. The findings below are all verified against the installed 1.2.7 wheel source.

## Summary

| # | Finding | Severity | Fix effort |
|---|---------|----------|-----------|
| 1 | No `compress(text)`; real entry point is `process_request(messages, ...)` | High | Small (wrapper + README) |
| 2 | `ParitokClient` supports Anthropic-shaped clients only | Medium | Docs only |
| 3 | `ratio` has opposite orientations in SDK vs proxy `/stats` | Medium | Small (rename or document) |
| 4 | `__version__` reports 1.2.3 while distribution is 1.2.7 | Low | Trivial |
| 5 | `[proxy]` / `[toolselect]` extras pull ~3 GB of PyTorch | Low | Docs only |

## 1. The obvious entry point does not exist (High)

**What happened.** We wrote our adapter assuming `ParitokEngine.compress(text: str)` — the natural first guess for a compression library and the shape most comparable SDKs use. It does not exist. The actual entry point is:

```python
ParitokEngine.process_request(
    messages: list[dict],
    tools: list[dict] | None = None,
    upstream_model: str = "",
) -> tuple[list[dict], list[dict] | None, CompressionStats, list[dict]]
```

It takes a **message list**, not a bare string.

**Why it matters.** Our adapter wrapped the call in a `try/except` that degraded gracefully. Every call landed silently in the except branch, returned `applied=False`, and the integration did nothing — while the full test suite stayed green. The suite proved the plumbing, not the product. We only found it by reading the wheel source; that cost hours.

**Reproduction.**
```python
from paritok import ParitokEngine
ParitokEngine().compress("hello")   # AttributeError: 'ParitokEngine' object has no attribute 'compress'
```

**Suggested fix**, in order of value:
1. Ship `compress_text(text: str) -> CompressionResult` for the single-prompt case. It is the first thing a new user reaches for.
2. Make the README's *first* code example `process_request` with its real signature and return tuple.
3. Add a `__getattr__` on `ParitokEngine` (or a targeted `AttributeError` message) that says "no `compress`; did you mean `process_request`?".

## 2. Anthropic-shaped client only (Medium)

**What happened.** `ParitokClient` wraps a client whose interface is `client.messages.create(**kwargs)` — Anthropic-shaped. There is no generic httpx transport adapter and no OpenAI-SDK wrapper inside the library. OpenAI-shaped traffic is served only by the standalone proxy (`paritok up`) via `ANTHROPIC_BASE_URL` / `OPENAI_BASE_URL` redirection.

**Why it matters.** Our heaviest inference path is Google Gemini vision→text, which is neither shape. We could not client-wrap it and instead called `ParitokEngine` directly on the prompt payload. This is a legitimate design boundary, not a bug — but it materially changes how you integrate, and it is not stated near the top of the README.

**Reproduction.** `ParitokClient(genai_client)` — the wrapper's call path assumes `.messages.create`, which the Gemini client does not expose.

**Suggested fix.** State the supported client shapes in the README's first section: "SDK wrapping supports Anthropic-shaped clients (`client.messages.create`). OpenAI-shaped and other providers: use the proxy, or call `ParitokEngine` directly."

## 3. Two opposite orientations for "ratio" (Medium)

**What happened.** `CompressionStats.ratio` is `round(1 - compressed/original, 3)` — the fraction **saved**, higher is better. The proxy server's `/stats` endpoint reports the **inverse** orientation.

**Why it matters.** Anyone reading both surfaces and not checking the source will publish an inverted number. We nearly did, in a results table.

**Reproduction.** Compare `CompressionStats.ratio` from an in-process `process_request` call against `curl localhost:<port>/stats` for the same payload; the two values are complements.

**Suggested fix.** Rename one of them — e.g. `saved_fraction` (SDK) vs `size_ratio` (proxy) — or document both side by side with a worked example showing the same payload producing both numbers.

## 4. `__version__` disagrees with the installed distribution (Low)

**What happened.** `paritok/__init__.py` line 3 hard-codes `__version__ = "1.2.3"` while the distribution installed by `pip install paritok` is 1.2.7.

**Why it matters.** We log the SDK version into provenance manifests for reproducibility. The version we recorded was not the version we ran.

**Reproduction.**
```python
import paritok, importlib.metadata as md
print(paritok.__version__)                  # 1.2.3
print(md.version("paritok"))                # 1.2.7
```

**Suggested fix.** `__version__ = importlib.metadata.version("paritok")`, or a release-time assertion that the literal matches the packaged version.

## 5. Install-size trap in the extras (Low)

**What happened.** Base install is small and pleasant: `click`, `httpx`, `pyyaml`, `tiktoken`. `paritok[proxy]` and `paritok[toolselect]` pull `sentence-transformers`, which pulls PyTorch — roughly 3 GB.

**Why it matters.** We built in a disk-constrained environment; installing an extra speculatively would have been fatal.

**Reproduction.** `pip install 'paritok[proxy]'` in a clean venv and compare `du -sh site-packages` against the base install.

**Suggested fix.** One line in the install docs listing which extras are heavy and roughly how heavy.

## What works well

- **`CompressionStats` exposes raw `original_tokens` and `compressed_tokens`**, not only a percentage. That let us publish a measurement a reader can recompute by hand, which is what an auditable claim requires.
- **The `PARITOK_DISABLE` kill switch is well judged.** Compression is an optimization and should never be a single point of failure; a documented bypass made our degradation path trivial to write and test.
- **Dependency-light base install.** Genuinely appreciated in a constrained build.

## Environment

paritok 1.2.7 · Python 3.12 · Linux (GitHub Codespaces) · installed via `pip` from PyPI.
