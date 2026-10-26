"""``python -m app`` -> run the demo server with uvicorn."""

from __future__ import annotations

import logging
import sys

from app.config import Settings


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if any(a in ("-h", "--help") for a in argv):
        print(
            "usage: python -m app\n\n"
            "Runs the FAHR-Face Live demo server. Configure it with environment variables:\n"
            "  FAHR_HOST / FAHR_PORT            bind address (default 127.0.0.1:8000)\n"
            "  FAHR_AGE_CHECKPOINT              age weights (default checkpoints/fahr_faceage.pth)\n"
            "  FAHR_SURVIVAL_CHECKPOINT         survival weights (optional)\n"
            "  FAHR_DETECTOR                    auto | yunet | retinaface\n"
            "  FAHR_DETECTION_THRESHOLD         detector score threshold\n"
            "  FAHR_DEVICE                      auto | cpu | cuda | mps\n"
            "  FAHR_ALLOW_RANDOM_WEIGHTS=1      dev only: random weights when a checkpoint is missing\n"
            "  FAHR_MAX_FACES, FAHR_MAX_UPLOAD_BYTES, FAHR_MODEL_CACHE_DIR, FAHR_YUNET_MODEL"
        )
        return 0

    try:
        settings = Settings.from_env()
    except ValueError as exc:
        print(f"Invalid configuration: {exc}", file=sys.stderr)
        return 2

    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    display_host = "localhost" if settings.host in ("0.0.0.0", "::", "") else settings.host
    print(f"FAHR-Face Live: open http://{display_host}:{settings.port} in your browser", flush=True)
    if settings.allow_random_weights:
        print("WARNING: FAHR_ALLOW_RANDOM_WEIGHTS=1 -> missing checkpoints use random weights (meaningless output)",
              flush=True)

    uvicorn.run("app.server:app", host=settings.host, port=settings.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
