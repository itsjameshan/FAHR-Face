"""FastAPI application: ``GET /``, ``/static/*``, ``GET /api/health``, ``POST /api/analyze``.

Nothing about uploaded frames or detected faces is stored or logged.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from app import __version__
from app.config import Settings

log = logging.getLogger("fahr.server")

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_STATIC_DIR = PACKAGE_DIR / "static"

API_PREFIX = "/api"


def _static_dir(settings: Settings) -> Path:
    return Path(settings.static_dir).expanduser() if settings.static_dir else DEFAULT_STATIC_DIR


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    """Build the application.  Models and the detector load in the lifespan."""
    settings = settings if settings is not None else Settings.from_env()
    static_dir = _static_dir(settings)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        state = application.state
        state.settings = settings
        state.detector = None
        state.detector_error = None
        state.models = None
        state.models_error = None
        state.analyzer = None
        state.device = None

        def _load() -> None:
            from app.detectors.factory import create_detector

            try:
                state.detector = create_detector(
                    settings.detector,
                    threshold=settings.detection_threshold,
                    model_path=settings.yunet_model,
                    cache_dir=settings.model_cache_path,
                )
                log.info("Detector ready: %s (threshold %.2f)", state.detector.name, state.detector.threshold)
            except Exception as exc:  # noqa: BLE001 - the server must still start
                state.detector_error = f"{type(exc).__name__}: {exc}"
                log.error("Face detector unavailable: %s", state.detector_error)

            try:
                from app.models import FaceModels

                state.models = FaceModels(settings)
                state.device = state.models.device
                log.info(
                    "Models ready on %s: age=%s survival=%s",
                    state.models.device,
                    state.models.age.weights,
                    state.models.survival.weights,
                )
            except Exception as exc:  # noqa: BLE001
                state.models_error = f"{type(exc).__name__}: {exc}"
                log.error("Models unavailable: %s", state.models_error)

            if state.detector is not None and state.models is not None:
                from app.pipeline import Analyzer

                state.analyzer = Analyzer(state.detector, state.models, settings)

        await run_in_threadpool(_load)
        try:
            yield
        finally:
            state.analyzer = None
            state.detector = None
            state.models = None

    app = FastAPI(
        title="FAHR-Face Live",
        version=__version__,
        lifespan=lifespan,
        docs_url=None,  # the Swagger UI would load assets from a CDN
        redoc_url=None,
    )

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["Permissions-Policy"] = "camera=(self)"
        if request.url.path.startswith(API_PREFIX):
            response.headers["Cache-Control"] = "no-store"
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["Referrer-Policy"] = "no-referrer"
        return response

    # The static directory may be created after the server starts; check_dir=False
    # lets the mount exist either way (missing files simply 404).
    app.mount("/static", StaticFiles(directory=str(static_dir), check_dir=False), name="static")

    @app.get("/", include_in_schema=False)
    async def index():
        index_html = static_dir / "index.html"
        if index_html.is_file():
            return FileResponse(str(index_html), media_type="text/html", headers={"Cache-Control": "no-cache"})
        return JSONResponse({"detail": "UI not built"}, status_code=404)

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        for name in ("favicon.ico", "favicon.svg"):
            candidate = static_dir / name
            if candidate.is_file():
                return FileResponse(str(candidate))
        return JSONResponse({"detail": "Not found"}, status_code=404)

    @app.get(f"{API_PREFIX}/health")
    async def health(request: Request):
        state = request.app.state
        detector = state.detector
        models = state.models
        if models is not None:
            models_block = models.health()
            device = models.device
            age_ok = models.age.weights == "checkpoint"
        else:
            error = state.models_error or "models not loaded"
            models_block = {
                "age": {"available": False, "weights": "none", "checkpoint": settings.age_checkpoint, "error": error},
                "survival": {
                    "available": False,
                    "weights": "none",
                    "checkpoint": settings.survival_checkpoint,
                    "error": error,
                },
            }
            device = None
            age_ok = False

        if detector is not None:
            detector_block = {
                "name": detector.name,
                "threshold": float(detector.threshold),
                "available": True,
                "error": None,
            }
        else:
            detector_block = {
                "name": settings.detector,
                "threshold": settings.detection_threshold,
                "available": False,
                "error": state.detector_error or "detector not loaded",
            }

        status = "ok" if (detector is not None and age_ok) else "degraded"
        return {
            "status": status,
            "version": __version__,
            "device": device,
            "detector": detector_block,
            "models": models_block,
            "limits": {"max_faces": settings.max_faces, "max_upload_bytes": settings.max_upload_bytes},
        }

    @app.post(f"{API_PREFIX}/analyze")
    async def analyze(
        request: Request,
        image: UploadFile = File(..., description="JPEG/PNG frame"),
        include_crops: bool = Query(False, description="Return 112x112 aligned crops as JPEG data URLs"),
        max_faces: Optional[int] = Query(None, description="Analyse at most this many faces (1..max_faces)"),
    ):
        state = request.app.state
        limit = settings.max_upload_bytes

        content_length = request.headers.get("content-length")
        if content_length and content_length.isdigit() and int(content_length) > limit + 4096:
            raise HTTPException(status_code=413, detail=f"Upload exceeds {limit} bytes")

        data = await image.read()
        await image.close()
        if len(data) > limit:
            raise HTTPException(status_code=413, detail=f"Upload exceeds {limit} bytes")
        if not data:
            raise HTTPException(status_code=400, detail="Empty image upload")

        analyzer = state.analyzer
        if analyzer is None or state.detector is None:
            raise HTTPException(
                status_code=503,
                detail=f"Face detector unavailable: {state.detector_error or state.models_error or 'not loaded'}",
            )

        from app.pipeline import ImageDecodeError, decode_image

        t0 = time.perf_counter()
        try:
            img = await run_in_threadpool(decode_image, data)
        except ImageDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"Could not decode image: {exc}") from exc
        decode_ms = (time.perf_counter() - t0) * 1000.0

        if max_faces is not None:
            max_faces = max(1, min(int(max_faces), settings.max_faces))

        result = await run_in_threadpool(
            analyzer.analyze, img, max_faces=max_faces, include_crops=include_crops, decode_ms=decode_ms
        )
        return JSONResponse(result)

    return app


app = create_app()

__all__ = ["app", "create_app"]
