# Genblaze API — verified from wheel source (genblaze-core 0.3.8)

Read directly from the installed wheel. Everything below is copied from real source, not inferred.

## Pipeline

```python
Pipeline(
    name: str | None = None,
    tenant_id: str | None = None,
    *,
    project_id: str | None = None,
    chain: bool = False,
    structured_log: bool = False,
    max_concurrency: int | None = None,
    moderation: ModerationHook | None = None,
    tracer: Tracer | None = None,
    preflight: bool = True,
)
```

Fluent config methods return `Pipeline`: `.config(cfg)`, `.tracer(t)`, `.cache(StepCache)`,
`.preflight(bool)`, `.metadata(**kw)`, `.from_result(PipelineResult)`.

`preflight=True` (default) validates every step's model against the provider registry BEFORE any
network call. `NOT_FOUND` raises immediately. This is free production-readiness evidence.

## Pipeline.step()

```python
.step(
    provider: BaseProvider,
    model: str,
    prompt: str | PromptTemplate | None = None,
    modality: Modality = Modality.IMAGE,
    step_type: StepType = StepType.GENERATE,
    fallback_models: list[str] | None = None,
    input_from: list[int] | int | None = None,
    external_inputs: list[Asset] | None = None,
    expected_duration_sec: float | None = None,
    metadata: dict[str, Any] | None = None,
    prompt_visibility: PromptVisibility = PromptVisibility.PUBLIC,
    params: dict[str, Any] | None = None,
    **extra_params,
) -> Pipeline
```

KEY FINDINGS:
- `input_from` wires step outputs to later step inputs by index -> this is the DAG mechanism.
- `fallback_models` is built in. Provider outage handling for free — cite in Production Readiness.
- `prompt_visibility` controls whether the prompt is recorded in the manifest.

## Pipeline.run()

```python
.run(
    sink: BaseSink | None = None,
    fail_fast: bool = True,
    raise_on_failure: bool | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
    on_progress = None,
    progress: bool | None = None,
    pipeline_timeout: float | None = None,
    on_step_complete = None,
    on_retry = None,
) -> PipelineResult
```

Also available: `.arun()` (async), `.batch_run(prompts=... | items=..., max_concurrency=...)`,
`.stream()` / `.astream()` for events, `.invoke()`, and critically **`.resume_step()`**.

`resume_step()` = partial-failure resume without reprocessing completed work. This is exactly the
"beyond a simple demo" evidence Backblaze's Production Readiness criterion asks for.

## StepBuilder (fluent Step construction)

```python
StepBuilder(provider: str, model: str)
    .prompt(text) .negative_prompt(text) .modality(m) .visibility(v) .step_type(t)
    .seed(s) .model_version(v) .model_hash(h)
    .input_asset(url, media_type, **kw) .params(**kw) .status(s)
    .asset(url, media_type, **kw) .meta(**kw)
    .build() -> Step
```

`.seed()`, `.model_version()`, `.model_hash()` exist as first-class fields — the SDK is designed for
reproducibility, which is the provenance story.

## Manifest (models/manifest.py)

"A hash-verified, canonical JSON document capturing full provenance."

Fields: `schema_version`, `run: Run`, `canonical_hash` (SHA-256 over canonical JSON),
`manifest_uri`, `signature` (reserved), `transfer_failures`.

`ManifestVerification`: `hash_ok`, `unverified_sha256_ids`, `invalid_metadata_ids`, `.ok` property.

There is a `canonical/` package (`canonical/json.py`, `canonical/_normalize.py`) implementing
deterministic serialization so the hash is stable across runs.

## StorageConfig

Fields include multipart thresholds, retries, timeouts, `signing_addressing_style: "virtual"|"path"`,
`user_agent_extra`. Source comments reference `b2ai-genblaze/<version>` and the
"Backblaze sample-app convention".

**B2 note:** use `signing_addressing_style="path"` for the S3-compatible endpoint.

## Other confirmed modules

- `pipeline/cache.py` -> `StepCache` (content-addressed step caching)
- `pipeline/moderation.py` -> `ModerationHook`, pre-step text moderation
- `pipeline/ingest.py` -> `Pipeline.ingest()`
- `pipeline/template.py` -> `to_template()` / `PipelineTemplate`
- `providers/compositor.py` -> `FFmpegCompositor`, `FFmpegTransform`
- `providers/retry.py`, `providers/probe.py`, `providers/pricing.py` (`estimated_cost` -> Decimal)
- `providers/model_registry.py`, `providers/discovery.py` -> `discover_providers()`
- `sinks/` -> `ParquetSink`, `BaseSink`; `ObjectStorageSink` in genblaze-s3

## Strategy implications for Interlude

1. `input_from` gives the real DAG. Steps are wired, not chained by hand.
2. `fallback_models` = provider resilience, free.
3. `resume_step()` = partial-failure recovery, free.
4. `estimated_cost` = per-run cost reporting, which pairs with the Paritok savings table.
5. `Manifest` + `canonical_hash` = provenance is native, not bolted on. Criterion #3 and #4 are
   satisfied by the same system.
