"""Interlude HTTP API.

Six endpoints, exactly as frozen:

    GET  /health
    POST /jobs
    GET  /jobs
    GET  /jobs/{job_id}
    POST /jobs/{job_id}/resume
    GET  /jobs/{job_id}/manifest

Three design constraints shape this file.

**Importing must never fail.** A missing B2 credential is an operational
problem, not an import problem. Nothing external is constructed at import
time; the store and the settings are built lazily inside dependencies and
every construction is wrapped. ``/health`` in particular is written so that it
cannot raise -- an operator's first diagnostic must not itself 500.

**POST /jobs must return immediately.** Describing a lecture takes minutes.
The request registers the job, hands the work to ``BackgroundTasks``, and
returns 202 with status ``pending``. The client polls ``GET /jobs/{id}``.

**Everything is injectable.** The registry, the store, and the provider
factory are FastAPI dependencies, so tests substitute fakes through
``app.dependency_overrides`` without monkeypatching module globals. The
default provider factory raises ``NotImplementedError`` with a pointed
message: real provider wiring is another module's job.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from fastapi import BackgroundTasks, Depends, FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.jobs import JobRecord, JobRegistry
from app.api.schemas import (
    CreateJobRequest,
    CreateJobResponse,
    ErrorResponse,
    HealthResponse,
    JobDetail,
    JobListResponse,
    ResumeJobResponse,
)
from app.config import ConfigError, get_settings
from app.pipeline.runner import JobState, JobStatus, Providers, Runner

logger = logging.getLogger(__name__)

VERSION = "1.0.0"

ProviderFactory = Callable[[], Providers]


# --------------------------------------------------------------------------
# Dependencies
# --------------------------------------------------------------------------

_registry = JobRegistry()


def get_registry() -> JobRegistry:
    """The process-wide job registry. Overridden in tests."""
    return _registry


def get_store() -> Any | None:
    """A B2Store, or None when B2 is not configured.

    Constructed per-request rather than at import time so that a missing
    credential degrades presigned URLs to null instead of preventing the
    application from starting.
    """
    try:
        from app.storage.store import B2Store  # local import: keeps import graph lazy

        settings = get_settings()
        return B2Store(settings.b2)
    except Exception:  # noqa: BLE001 - a missing store is a degraded mode, not an error
        logger.warning("B2 store unavailable; presigned URLs will be omitted", exc_info=True)
        return None


def _unwired_providers() -> Providers:
    raise NotImplementedError(
        "No provider factory is wired. Override the `get_provider_factory` "
        "dependency (or pass one to `create_app`) with a callable returning "
        "app.pipeline.runner.Providers. See app/adapters/."
    )


def get_provider_factory() -> ProviderFactory:
    """Returns the callable that builds real Providers. Overridden by wiring."""
    return _unwired_providers


def _presigner(store: Any | None) -> Callable[[str], str] | None:
    if store is None:
        return None

    def presign(key: str) -> str:
        return store.presigned_url(key)

    return presign


# --------------------------------------------------------------------------
# Background execution
# --------------------------------------------------------------------------


def _execute_job(
    registry: JobRegistry,
    job_id: str,
    source_url: str,
    factory: ProviderFactory,
) -> None:
    """Run one job to completion on a background thread.

    Never raises. A failure is recorded on the job so the client can see it;
    an exception escaping here would be swallowed by the task runner and the
    job would sit at ``running`` forever.
    """
    record = registry.get(job_id)
    if record is None:
        logger.error("background task for unknown job %s", job_id)
        return

    try:
        providers = factory()
    except Exception as exc:  # noqa: BLE001
        logger.exception("provider wiring failed for job %s", job_id)
        registry.set_status(job_id, JobStatus.FAILED, error=f"{type(exc).__name__}: {exc}")
        return

    registry.set_status(job_id, JobStatus.RUNNING)
    try:
        max_attempts = 3
        try:
            max_attempts = get_settings().max_step_retries
        except ConfigError:
            pass
        Runner(providers, max_draft_attempts=max_attempts).run(record.state, source_url)
    except Exception as exc:  # noqa: BLE001 - Runner already traps, this is belt-and-braces
        logger.exception("job %s crashed outside the runner", job_id)
        registry.set_status(job_id, JobStatus.FAILED, error=f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------
# Application
# --------------------------------------------------------------------------


def _not_found(job_id: str) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content=ErrorResponse(
            error="job_not_found", detail=f"no job with id {job_id!r}"
        ).model_dump(),
    )


def create_app() -> FastAPI:
    app = FastAPI(
        title="Interlude",
        version=VERSION,
        description="Automated audio description for institutional video libraries.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(RequestValidationError)
    async def _validation_handler(  # pyright: ignore[reportUnusedFunction]
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ErrorResponse(error="invalid_request", detail=str(exc.errors())).model_dump(),
        )

    # -- GET /health -------------------------------------------------------

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:  # pyright: ignore[reportUnusedFunction]
        """Must never raise. A broken config is reported, not thrown."""
        providers_missing: list[str] = []
        b2_ok = False
        try:
            settings = get_settings()
            providers_missing = settings.providers.missing()
            b2_ok = bool(
                settings.b2.key_id
                and settings.b2.app_key
                and settings.b2.bucket
                and settings.b2.endpoint
            )
        except ConfigError as exc:
            logger.warning("health: configuration incomplete: %s", exc)
            providers_missing = [
                "GOOGLE_API_KEY",
                "ASSEMBLYAI_API_KEY",
                "ELEVENLABS_API_KEY",
            ]
        except Exception:  # noqa: BLE001
            logger.exception("health: unexpected configuration failure")
        return HealthResponse(
            status="ok" if b2_ok else "degraded",
            version=VERSION,
            providers_missing=providers_missing,
            b2_ok=b2_ok,
        )

    # -- POST /jobs --------------------------------------------------------

    @app.post("/jobs", response_model=CreateJobResponse, status_code=status.HTTP_202_ACCEPTED)
    def create_job(  # pyright: ignore[reportUnusedFunction]
        payload: CreateJobRequest,
        background: BackgroundTasks,
        registry: JobRegistry = Depends(get_registry),
        factory: ProviderFactory = Depends(get_provider_factory),
    ) -> CreateJobResponse:
        # source_key/source_sha256 are filled in by the transcribe stage once
        # the source has actually been fetched and stored. Until then the job
        # is honest about not knowing them.
        state = JobState.new(source_key="", source_sha256="")
        title = payload.title or payload.source_url.rsplit("/", 1)[-1] or state.job_id
        registry.add(state, title=title, source_url=payload.source_url)
        background.add_task(
            _execute_job, registry, state.job_id, payload.source_url, factory
        )
        logger.info("job %s accepted (%s)", state.job_id, payload.source_url)
        return CreateJobResponse(job_id=state.job_id, status=state.status.value)

    # -- GET /jobs ---------------------------------------------------------

    @app.get("/jobs", response_model=JobListResponse)
    def list_jobs(  # pyright: ignore[reportUnusedFunction]
        registry: JobRegistry = Depends(get_registry),
    ) -> JobListResponse:
        return JobListResponse(jobs=registry.summaries())  # type: ignore[arg-type]

    # -- GET /jobs/{job_id} ------------------------------------------------

    @app.get("/jobs/{job_id}", response_model=JobDetail, responses={404: {"model": ErrorResponse}})
    def get_job(  # pyright: ignore[reportUnusedFunction]
        job_id: str,
        registry: JobRegistry = Depends(get_registry),
        store: Any | None = Depends(get_store),
    ) -> Any:
        detail = registry.detail(job_id, presign=_presigner(store))
        if detail is None:
            return _not_found(job_id)
        return JSONResponse(status_code=status.HTTP_200_OK, content=detail)

    # -- POST /jobs/{job_id}/resume ---------------------------------------

    @app.post(
        "/jobs/{job_id}/resume",
        response_model=ResumeJobResponse,
        status_code=status.HTTP_202_ACCEPTED,
        responses={404: {"model": ErrorResponse}},
    )
    def resume_job(  # pyright: ignore[reportUnusedFunction]
        job_id: str,
        background: BackgroundTasks,
        registry: JobRegistry = Depends(get_registry),
        factory: ProviderFactory = Depends(get_provider_factory),
    ) -> Any:
        record: JobRecord | None = registry.get(job_id)
        if record is None:
            return _not_found(job_id)
        resumed_from = record.state.resume_from.value
        record.state.error = ""
        registry.set_status(job_id, JobStatus.PENDING)
        background.add_task(
            _execute_job, registry, job_id, record.source_url, factory
        )
        logger.info("job %s resume requested at stage %s", job_id, resumed_from)
        return ResumeJobResponse(
            job_id=job_id, status=JobStatus.PENDING.value, resumed_from=resumed_from
        )

    # -- GET /jobs/{job_id}/manifest --------------------------------------

    @app.get("/jobs/{job_id}/manifest", responses={404: {"model": ErrorResponse}})
    def get_manifest(  # pyright: ignore[reportUnusedFunction]
        job_id: str,
        registry: JobRegistry = Depends(get_registry),
    ) -> Any:
        payload = registry.manifest_payload(job_id)
        if payload is None:
            return _not_found(job_id)
        return JSONResponse(status_code=status.HTTP_200_OK, content=payload)

    return app


app = create_app()


if __name__ == "__main__":
    import sys

    from fastapi.testclient import TestClient

    from app.pipeline.gaps import Gap
    from app.pipeline.runner import DescribedSegment, Stage

    test_app = create_app()
    registry = JobRegistry()

    def _fake_providers() -> Providers:
        raise AssertionError("self-check should not invoke real providers")

    test_app.dependency_overrides[get_registry] = lambda: registry
    test_app.dependency_overrides[get_store] = lambda: None
    test_app.dependency_overrides[get_provider_factory] = lambda: _fake_providers

    client = TestClient(test_app, raise_server_exceptions=False)

    # /health never raises, whatever the environment
    r = client.get("/health")
    assert r.status_code == 200, r.status_code
    body = r.json()
    assert set(body) == {"status", "version", "providers_missing", "b2_ok"}, set(body)
    assert isinstance(body["b2_ok"], bool)
    assert body["version"] == VERSION

    # empty listing
    r = client.get("/jobs")
    assert r.status_code == 200 and r.json() == {"jobs": []}

    # bad URLs are 422
    for bad in ({"source_url": ""}, {"source_url": "ftp://x/y"}, {}):
        r = client.post("/jobs", json=bad)
        assert r.status_code == 422, (bad, r.status_code)
        assert r.json()["error"] == "invalid_request"

    # unknown ids are 404 on all three lookup routes
    for path in ("/jobs/deadbeef", "/jobs/deadbeef/manifest"):
        r = client.get(path)
        assert r.status_code == 404, (path, r.status_code)
        assert r.json()["error"] == "job_not_found", r.json()
    r = client.post("/jobs/deadbeef/resume")
    assert r.status_code == 404 and r.json()["error"] == "job_not_found"

    # a job registered directly (bypassing the background task) serializes fully
    state = JobState.new(source_key="source/ab/cd/x.mp4", source_sha256="ab" * 32)
    state.gaps = [Gap(0.5, 6.0)]
    state.segments = [
        DescribedSegment(gap=Gap(0.5, 6.0), text="A diagram appears.", accepted=True, attempts=1)
    ]
    state.mark(Stage.TRANSCRIBE, ok=True, duration_ms=120, detail="1 words")
    state.status = JobStatus.PARTIAL
    registry.add(state, title="Lecture 101", source_url="https://x.test/a.mp4")

    r = client.get("/jobs")
    assert r.status_code == 200
    jobs = r.json()["jobs"]
    assert len(jobs) == 1
    assert set(jobs[0]) == {
        "job_id", "title", "status", "created_at",
        "gaps_found", "segments_described", "segments_rendered", "duration_ms",
    }, set(jobs[0])

    r = client.get(f"/jobs/{state.job_id}")
    assert r.status_code == 200, r.status_code
    detail = r.json()
    assert set(detail) == {
        "job_id", "title", "status", "created_at",
        "gaps_found", "segments_described", "segments_rendered", "duration_ms",
        "source", "stages", "segments", "tokens", "manifest_hash",
    }, set(detail)
    assert detail["segments"][0]["audio_url"] is None
    assert detail["status"] == "partial"

    r = client.get(f"/jobs/{state.job_id}/manifest")
    assert r.status_code == 200
    assert r.json()["run_id"] == state.job_id
    assert len(r.json()["canonical_hash"]) == 64

    r = client.post(f"/jobs/{state.job_id}/resume")
    assert r.status_code == 202, r.status_code
    resumed = r.json()
    assert set(resumed) == {"job_id", "status", "resumed_from"}
    assert resumed["resumed_from"] == "detect_gaps", resumed

    # POST /jobs returns 202 pending; the background task then fails loudly
    # because the fake factory raises -- proving failures land on the job.
    r = client.post("/jobs", json={"source_url": "https://x.test/b.mp4", "title": "L102"})
    assert r.status_code == 202, r.status_code
    created = r.json()
    assert set(created) == {"job_id", "status"}
    assert created["status"] == "pending"
    new_id = created["job_id"]
    after = client.get(f"/jobs/{new_id}").json()
    assert after["status"] == "failed", after["status"]
    assert after["title"] == "L102"

    if not sys.flags.optimize:
        print("app.main self-check OK")
