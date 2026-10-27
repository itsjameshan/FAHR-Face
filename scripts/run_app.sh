#!/usr/bin/env bash
# One-command start of the FAHR-Face Live webcam demo.
#
#   bash scripts/run_app.sh                 # create .venv, install dependencies, start the server
#   bash scripts/run_app.sh --skip-install  # start the server with the existing .venv
#   FAHR_PORT=8080 bash scripts/run_app.sh  # any FAHR_* variable is passed through to the server
#
# On Linux, torch/torchvision are installed from the CPU-only wheel index first (much smaller
# than the default CUDA wheels). Set FAHR_GPU=1 to skip that and get the default wheels.
# The server itself has no command-line flags; it is configured via FAHR_* environment
# variables (see `python -m app --help` or the README section "Live webcam demo").
#
# Portable to bash 3.2 (macOS): no mapfile, no associative arrays.
set -euo pipefail

usage() {
  cat <<'EOF'
usage: bash scripts/run_app.sh [--skip-install] [-h|--help]

Creates the virtual environment .venv (python3 -m venv) if it does not exist, installs
requirements-app.txt into it and starts the demo server with `python -m app`.
Then open http://localhost:8000 (or FAHR_PORT) in your browser and allow the camera.

  --skip-install   do not touch pip; just start the server from the existing .venv
  -h, --help       show this help and exit

Environment variables read by this script:
  FAHR_GPU=1       (Linux) install the default torch wheels instead of the CPU-only ones
Environment variables passed through to the server (all optional):
  FAHR_HOST, FAHR_PORT, FAHR_AGE_CHECKPOINT, FAHR_SURVIVAL_CHECKPOINT, FAHR_DETECTOR,
  FAHR_DETECTION_THRESHOLD, FAHR_DEVICE, FAHR_ALLOW_RANDOM_WEIGHTS, FAHR_MAX_FACES,
  FAHR_MAX_UPLOAD_BYTES, FAHR_MODEL_CACHE_DIR, FAHR_YUNET_MODEL
EOF
}

SKIP_INSTALL=0
args=()
for arg in "$@"; do
  case "$arg" in
    --skip-install) SKIP_INSTALL=1 ;;
    -h|--help) usage; exit 0 ;;
    *) args+=("$arg") ;;
  esac
done
# Remaining arguments are handed to `python -m app` unchanged.
set -- ${args[@]+"${args[@]}"}

# Work from the repository root: requirements-app.txt, .venv and the default
# checkpoints/ path are all relative to it.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"
cd "$REPO_ROOT"

# Interpreter inside .venv (POSIX layout, or Scripts/ for Git Bash on Windows).
venv_python() {
  if [ -x .venv/bin/python ]; then
    echo .venv/bin/python
  elif [ -x .venv/Scripts/python.exe ]; then
    echo .venv/Scripts/python.exe
  else
    return 1
  fi
}

if [ ! -d .venv ]; then
  if ! command -v python3 > /dev/null 2>&1; then
    echo "error: python3 was not found on PATH. Install Python 3.10 or newer and try again." >&2
    exit 1
  fi
  echo "==> Creating virtual environment .venv ..."
  if ! python3 -m venv .venv; then
    echo "error: could not create .venv (on Debian/Ubuntu: sudo apt install python3-venv)." >&2
    exit 1
  fi
fi

if ! PY="$(venv_python)"; then
  echo "error: .venv exists but contains no Python interpreter. Remove .venv and run this script again." >&2
  exit 1
fi

if [ "$SKIP_INSTALL" -eq 0 ]; then
  echo "==> Upgrading pip ..."
  "$PY" -m pip install --quiet --upgrade pip
  if [ "$(uname -s)" = "Linux" ] && [ "${FAHR_GPU:-0}" != "1" ]; then
    echo "==> Installing CPU-only torch and torchvision (set FAHR_GPU=1 for the default wheels) ..."
    "$PY" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  fi
  echo "==> Installing requirements-app.txt ..."
  "$PY" -m pip install -r requirements-app.txt
else
  echo "==> Skipping dependency installation (--skip-install)."
fi

AGE_CHECKPOINT="${FAHR_AGE_CHECKPOINT:-checkpoints/fahr_faceage.pth}"
if [ ! -s "$AGE_CHECKPOINT" ]; then
  echo "note: no age checkpoint at $AGE_CHECKPOINT."
  echo "      Run 'bash scripts/download_checkpoints.sh' to fetch it; until then the page shows"
  echo "      the age model as missing (or random weights if FAHR_ALLOW_RANDOM_WEIGHTS=1)."
fi

echo "==> Starting FAHR-Face Live at http://localhost:${FAHR_PORT:-8000}  (press Ctrl-C to stop)"
exec "$PY" -m app "$@"
