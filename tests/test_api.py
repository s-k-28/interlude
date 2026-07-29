"""API-level tests.

Exercises the six frozen endpoints through FastAPI's TestClient with fakes
substituted via ``dependency_overrides``. No network, no B2, no API keys: the
whole surface is testable on a laptop with no credentials, which is the point
of injecting the registry, the store, and the provider factory.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api.jobs import DETAIL_KEYS, SUMMARY_KEYS, TOKEN_KEYS, JobRegistry
from app.main import create_app, get_provider_factory, get_registry, get_store
from app.pipeline.gaps import Gap
from app.pipeline.runner import (
    DescribedSegment,
    JobState,
    JobStatus,
    Providers,
    Stage,
    TranscriptInput,
)


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------


class FakeStore:
    """Stands in for B2Store. Only `presigned_url` is exercised by the API."""

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[str] = []

    def presigned_url(self, key: str, *, expires_in: int = 3600) -> str:
        self.calls.append(key)
        if self.fail:
            raise RuntimeError("presign unavailable")
        return f"https://signed.test/{key}?e={expires_in}"


def fake_providers() -> Providers:
    """A pipeline that describes gaps and renders them, entirely offline."""

    def transcribe(_url: str) -> TranscriptInput:
        return TranscriptInput(
            text="one two", word_spans=[(0.0, 0.5), (10.0, 10.5)], duration=30.0
        )

    def scene_context(_ts: float, _budget: int) -> str:
        return "A lecturer beside a projected diagram."

    def draft(_prompt: str, budget: int) -> str:
        return " ".join(["word"] * max(1, min(budget, 6)))

    def synthesize(_text: str) -> bytes:
        return b"ID3fake-audio"

    def store(_data: bytes, namespace: str) -> tuple[str, str]:
        return (f"{namespace}/aa/bb/" + "aa" * 32 + ".mp3", "aa" * 32)

    return Providers(
        transcribe=transcribe,
        scene_context=scene_context,
        draft=draft,
        synthesize=synthesize,
        store=store,
    )


def exploding_providers() -> Providers:
    def transcribe(_url: str) -> TranscriptInput:
        raise RuntimeError("transcription provider is down")

    return Providers(
        transcribe=transcribe,
        scene_context=lambda _t, _b: "",
        draft=lambda _p, _b: "",
        synthesize=lambda _t: b"",
        store=lambda _d, _n: ("", ""),
    )


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def registry() -> JobRegistry:
    return JobRegistry()


@pytest.fixture
def store() -> FakeStore:
    return FakeStore()


@pytest.fixture
def client(registry: JobRegistry, store: FakeStore) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_registry] = lambda: registry
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_provider_factory] = lambda: fake_providers
    return TestClient(app, raise_server_exceptions=False)


def seed_job(registry: JobRegistry, *, title: str = "Lecture 101") -> JobState:
    """A realistic partially-complete job, registered without running it."""
    state = JobState.new(source_key="source/ab/cd/x.mp4", source_sha256="ab" * 32)
    state.transcript = TranscriptInput(text="hi", word_spans=[(0.0, 0.5)], duration=60.0)
    state.gaps = [Gap(0.5, 6.0), Gap(20.0, 23.0)]
    state.segments = [
        DescribedSegment(
            gap=Gap(0.5, 6.0),
            text="A lecturer stands beside a projected diagram.",
            accepted=True,
            attempts=1,
            audio_key="description-audio/aa/bb/aabb.mp3",
            audio_sha256="cd" * 32,
        ),
        DescribedSegment(
            gap=Gap(20.0, 23.0), text="The slide changes.", accepted=True, attempts=2
        ),
    ]
    state.ledger.record(uncompressed_tokens=1000, compressed_tokens=400, completion_tokens=60)
    state.mark(Stage.TRANSCRIBE, ok=True, duration_ms=900, detail="1 words")
    state.mark(Stage.DETECT_GAPS, ok=True, duration_ms=3, detail="2 gaps")
    state.mark(Stage.DESCRIBE, ok=True, duration_ms=4100, detail="2/2 within budget")
    state.status = JobStatus.PARTIAL
    registry.add(state, title=title, source_url="https://media.test/lec.mp4")
    return state


# --------------------------------------------------------------------------
# /health
# --------------------------------------------------------------------------


def test_health_shape_and_never_raises(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"status", "version", "providers_missing", "b2_ok"}
    assert isinstance(body["b2_ok"], bool)
    assert isinstance(body["providers_missing"], list)
    assert body["status"] in {"ok", "degraded"}


def test_health_reports_b2_false_when_config_missing(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    import app.main as main_module
    from app.config import ConfigError

    def boom() -> object:
        raise ConfigError("B2_KEY_ID is not set")

    monkeypatch.setattr(main_module, "get_settings", boom)
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["b2_ok"] is False
    assert "GOOGLE_API_KEY" in body["providers_missing"]


# --------------------------------------------------------------------------
# POST /jobs
# --------------------------------------------------------------------------


def test_create_job_returns_202_pending(client: TestClient) -> None:
    response = client.post("/jobs", json={"source_url": "https://media.test/a.mp4"})
    assert response.status_code == 202
    body = response.json()
    assert set(body) == {"job_id", "status"}
    assert len(body["job_id"]) == 12
    assert body["status"] == "pending"


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"source_url": ""},
        {"source_url": "   "},
        {"source_url": "ftp://media.test/a.mp4"},
        {"source_url": "not-a-url"},
        {"source_url": "https://"},
        {"source_url": "https://x.test/a.mp4", "unexpected": True},
    ],
)
def test_create_job_rejects_bad_input(client: TestClient, payload: dict) -> None:
    response = client.post("/jobs", json=payload)
    assert response.status_code == 422
    assert response.json()["error"] == "invalid_request"


def test_create_job_runs_pipeline_to_completion(client: TestClient) -> None:
    # TestClient runs BackgroundTasks synchronously before returning, so by the
    # time the POST resolves the job has already executed.
    job_id = client.post(
        "/jobs", json={"source_url": "https://media.test/a.mp4"}
    ).json()["job_id"]
    detail = client.get(f"/jobs/{job_id}").json()
    assert detail["status"] == "complete", detail
    assert detail["gaps_found"] > 0
    assert detail["segments_described"] > 0
    assert detail["segments_rendered"] == detail["segments_described"]
    assert [s["stage"] for s in detail["stages"]] == [
        "transcribe",
        "detect_gaps",
        "describe",
        "synthesize",
        "manifest",
    ]


def test_provider_failure_lands_on_the_job_not_the_request(
    registry: JobRegistry, store: FakeStore
) -> None:
    app = create_app()
    app.dependency_overrides[get_registry] = lambda: registry
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_provider_factory] = lambda: exploding_providers
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post("/jobs", json={"source_url": "https://media.test/a.mp4"})
    assert response.status_code == 202
    detail = client.get(f"/jobs/{response.json()['job_id']}").json()
    assert detail["status"] == "failed"


def test_unwired_provider_factory_fails_the_job_cleanly(
    registry: JobRegistry, store: FakeStore
) -> None:
    app = create_app()
    app.dependency_overrides[get_registry] = lambda: registry
    app.dependency_overrides[get_store] = lambda: store
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post("/jobs", json={"source_url": "https://media.test/a.mp4"})
    assert response.status_code == 202
    detail = client.get(f"/jobs/{response.json()['job_id']}").json()
    assert detail["status"] == "failed"


def test_title_defaults_to_filename(client: TestClient) -> None:
    job_id = client.post(
        "/jobs", json={"source_url": "https://media.test/bio-204.mp4"}
    ).json()["job_id"]
    detail = client.get(f"/jobs/{job_id}").json()
    assert detail["title"] == "bio-204.mp4"


# --------------------------------------------------------------------------
# GET /jobs
# --------------------------------------------------------------------------


def test_list_jobs_empty(client: TestClient) -> None:
    assert client.get("/jobs").json() == {"jobs": []}


def test_list_jobs_summary_keys_are_frozen(
    client: TestClient, registry: JobRegistry
) -> None:
    seed_job(registry)
    jobs = client.get("/jobs").json()["jobs"]
    assert len(jobs) == 1
    assert set(jobs[0]) == set(SUMMARY_KEYS)
    assert jobs[0]["gaps_found"] == 2
    assert jobs[0]["segments_described"] == 2
    assert jobs[0]["segments_rendered"] == 1
    assert jobs[0]["duration_ms"] == 5003


def test_list_jobs_is_newest_first(client: TestClient, registry: JobRegistry) -> None:
    first = seed_job(registry, title="A")
    second = seed_job(registry, title="B")
    ids = [j["job_id"] for j in client.get("/jobs").json()["jobs"]]
    assert ids == [second.job_id, first.job_id]


# --------------------------------------------------------------------------
# GET /jobs/{job_id}
# --------------------------------------------------------------------------


def test_detail_keys_are_frozen(client: TestClient, registry: JobRegistry) -> None:
    state = seed_job(registry)
    detail = client.get(f"/jobs/{state.job_id}").json()
    assert set(detail) == set(DETAIL_KEYS)
    assert set(detail["source"]) == {"key", "sha256"}
    assert set(detail["stages"][0]) == {"stage", "ok", "duration_ms", "detail"}
    assert set(detail["segments"][0]) == {
        "start",
        "end",
        "duration",
        "word_budget",
        "text",
        "accepted",
        "attempts",
        "audio_key",
        "audio_url",
    }
    assert set(detail["tokens"]) == set(TOKEN_KEYS)
    assert len(detail["manifest_hash"]) == 64


def test_detail_presigns_only_rendered_segments(
    client: TestClient, registry: JobRegistry, store: FakeStore
) -> None:
    state = seed_job(registry)
    segments = client.get(f"/jobs/{state.job_id}").json()["segments"]
    assert segments[0]["audio_url"].startswith("https://signed.test/description-audio/")
    assert segments[1]["audio_url"] is None
    assert store.calls == ["description-audio/aa/bb/aabb.mp3"]


def test_presign_failure_degrades_to_null_not_500(registry: JobRegistry) -> None:
    app = create_app()
    app.dependency_overrides[get_registry] = lambda: registry
    app.dependency_overrides[get_store] = lambda: FakeStore(fail=True)
    app.dependency_overrides[get_provider_factory] = lambda: fake_providers
    client = TestClient(app, raise_server_exceptions=False)

    state = seed_job(registry)
    response = client.get(f"/jobs/{state.job_id}")
    assert response.status_code == 200
    assert response.json()["segments"][0]["audio_url"] is None
    assert response.json()["segments"][0]["audio_key"] != ""


def test_missing_store_omits_urls(registry: JobRegistry) -> None:
    app = create_app()
    app.dependency_overrides[get_registry] = lambda: registry
    app.dependency_overrides[get_store] = lambda: None
    app.dependency_overrides[get_provider_factory] = lambda: fake_providers
    client = TestClient(app, raise_server_exceptions=False)

    state = seed_job(registry)
    segments = client.get(f"/jobs/{state.job_id}").json()["segments"]
    assert all(s["audio_url"] is None for s in segments)


def test_detail_404(client: TestClient) -> None:
    response = client.get("/jobs/deadbeefcafe")
    assert response.status_code == 404
    assert response.json()["error"] == "job_not_found"
    assert "deadbeefcafe" in response.json()["detail"]


def test_token_values_round_trip(client: TestClient, registry: JobRegistry) -> None:
    state = seed_job(registry)
    tokens = client.get(f"/jobs/{state.job_id}").json()["tokens"]
    assert tokens["calls"] == 1
    assert tokens["prompt_tokens_uncompressed"] == 1000
    assert tokens["prompt_tokens_compressed"] == 400
    assert tokens["completion_tokens"] == 60
    assert tokens["tokens_saved"] == 600
    assert 0.0 <= tokens["reduction_ratio"] <= 1.0


# --------------------------------------------------------------------------
# POST /jobs/{job_id}/resume
# --------------------------------------------------------------------------


def test_resume_reports_the_stage_it_restarts_at(
    client: TestClient, registry: JobRegistry
) -> None:
    state = seed_job(registry)
    # seed_job completed transcribe/detect_gaps/describe, so synthesize is next
    response = client.post(f"/jobs/{state.job_id}/resume")
    assert response.status_code == 202
    body = response.json()
    assert set(body) == {"job_id", "status", "resumed_from"}
    assert body["job_id"] == state.job_id
    assert body["resumed_from"] == "synthesize"


def test_resume_actually_finishes_the_job(
    client: TestClient, registry: JobRegistry
) -> None:
    state = seed_job(registry)
    client.post(f"/jobs/{state.job_id}/resume")
    detail = client.get(f"/jobs/{state.job_id}").json()
    assert detail["status"] in {"complete", "partial"}
    assert detail["segments_rendered"] == 2
    assert [s["stage"] for s in detail["stages"]] == [
        "transcribe",
        "detect_gaps",
        "describe",
        "synthesize",
        "manifest",
    ]


def test_resume_404(client: TestClient) -> None:
    response = client.post("/jobs/nope/resume")
    assert response.status_code == 404
    assert response.json()["error"] == "job_not_found"


def test_resume_is_idempotent_on_a_finished_job(client: TestClient) -> None:
    job_id = client.post(
        "/jobs", json={"source_url": "https://media.test/a.mp4"}
    ).json()["job_id"]
    first = client.get(f"/jobs/{job_id}").json()
    response = client.post(f"/jobs/{job_id}/resume")
    assert response.status_code == 202
    assert response.json()["resumed_from"] == "manifest"
    second = client.get(f"/jobs/{job_id}").json()
    assert second["segments_described"] == first["segments_described"]
    assert len(second["stages"]) == len(first["stages"]) == 5


# --------------------------------------------------------------------------
# GET /jobs/{job_id}/manifest
# --------------------------------------------------------------------------


def test_manifest_is_hash_verifiable(client: TestClient, registry: JobRegistry) -> None:
    state = seed_job(registry)
    response = client.get(f"/jobs/{state.job_id}/manifest")
    assert response.status_code == 200
    payload = response.json()
    assert payload["run_id"] == state.job_id
    assert payload["source"]["sha256"] == "ab" * 32
    assert len(payload["canonical_hash"]) == 64
    detail = client.get(f"/jobs/{state.job_id}").json()
    assert detail["manifest_hash"] == payload["canonical_hash"]


def test_manifest_404(client: TestClient) -> None:
    response = client.get("/jobs/nope/manifest")
    assert response.status_code == 404
    assert response.json()["error"] == "job_not_found"


# --------------------------------------------------------------------------
# Cross-cutting
# --------------------------------------------------------------------------


def test_cors_headers_present(client: TestClient) -> None:
    response = client.get("/health", headers={"Origin": "http://localhost:5173"})
    assert response.headers.get("access-control-allow-origin") == "*"


def test_registry_is_thread_safe_under_concurrent_writes() -> None:
    import threading

    registry = JobRegistry()

    def worker() -> None:
        for _ in range(50):
            registry.add(
                JobState.new("source/x", "00" * 32),
                title="t",
                source_url="https://x.test/a.mp4",
            )

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(registry) == 400
    assert len(registry.summaries()) == 400
