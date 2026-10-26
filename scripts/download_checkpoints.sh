#!/usr/bin/env bash
# Download the publicly released FAHR checkpoints from Hugging Face into checkpoints/.
#
#   bash scripts/download_checkpoints.sh                    # fahr_faceage.pth (FAHR-FaceAge)
#   bash scripts/download_checkpoints.sh --with-foundation  # + fahr_face.pth (FAHR-Face foundation)
#   bash scripts/download_checkpoints.sh --force            # re-download existing files
#
# Downloads are resumable: each file is fetched with `curl -C -` into <name>.part and
# renamed when complete, so an interrupted transfer continues where it stopped the next
# time the script runs. Complete files are skipped unless --force is given.
#
# Alternative with the Hugging Face CLI (pip install -U "huggingface_hub[cli]"):
#   hf download fridolin22/FAHR-Face fahr_faceage.pth --local-dir checkpoints
#   hf download fridolin22/FAHR-Face fahr_face.pth fahr_faceage.pth --local-dir checkpoints
#
# Portable to bash 3.2 (macOS): no mapfile, no associative arrays, no ${var,,}.
set -euo pipefail

# Mirror support (advanced): point FAHR_HF_BASE_URL at another host that serves the
# same file names, e.g. https://hf-mirror.com/fridolin22/FAHR-Face/resolve/main
BASE_URL="${FAHR_HF_BASE_URL:-https://huggingface.co/fridolin22/FAHR-Face/resolve/main}"
CHECKPOINT_DIR="checkpoints"
AGE_FILE="fahr_faceage.pth"
FOUNDATION_FILE="fahr_face.pth"
SURVIVAL_FILE="fahr_facesurvival.pth"

usage() {
  cat <<'EOF'
usage: bash scripts/download_checkpoints.sh [--with-foundation] [--force] [-h|--help]

Downloads the publicly released FAHR checkpoints from Hugging Face into checkpoints/
(created next to this repository's run.py). No Hugging Face login is required.

  (default)           fahr_faceage.pth   FAHR-FaceAge; needed by the webcam demo and run.py
  --with-foundation   also fahr_face.pth FAHR-Face foundation model (not needed for the demo)
  --force             re-download a file even if it already exists
  -h, --help          show this help and exit

Interrupted downloads resume automatically: just run the script again.
Environment: FAHR_HF_BASE_URL overrides the download host (for Hugging Face mirrors).

FAHR-FaceSurvival weights are not publicly released. If you have them, place the file at
checkpoints/fahr_facesurvival.pth yourself; the demo then shows the relative risk score.
EOF
}

# Print a file size in a human-friendly unit (works on macOS and Linux).
file_size() {
  local bytes
  bytes=$(( $(wc -c < "$1") ))
  if [ "$bytes" -ge 1048576 ]; then
    echo "$(( bytes / 1048576 )) MB"
  elif [ "$bytes" -ge 1024 ]; then
    echo "$(( bytes / 1024 )) KB"
  else
    echo "$bytes bytes"
  fi
}

# download <file name>  -> 0 on success / skip, 1 on failure (partial file is kept for resume)
download() {
  local name="$1"
  local dest="$CHECKPOINT_DIR/$name"
  local part="$dest.part"
  local url="$BASE_URL/$name?download=true"

  if [ "$FORCE" -eq 1 ]; then
    rm -f "$dest" "$part"
  elif [ -s "$dest" ]; then
    echo "==> $dest already exists ($(file_size "$dest")); skipping."
    echo "    Use --force to download it again (e.g. if the demo reports it cannot be loaded)."
    return 0
  fi

  if [ -s "$part" ]; then
    echo "==> Resuming $name ($(file_size "$part") already downloaded) ..."
  else
    echo "==> Downloading $name (this file is large; it may take a while) ..."
  fi
  echo "    from $url"

  if ! curl -L -C - --fail --retry 3 -o "$part" "$url"; then
    echo "error: download of $name failed." >&2
    echo "       The partial file is kept at $part; run this script again to resume." >&2
    echo "       Manual alternative: hf download fridolin22/FAHR-Face $name --local-dir $CHECKPOINT_DIR" >&2
    return 1
  fi
  if [ ! -s "$part" ]; then
    rm -f "$part"
    echo "error: $name downloaded as an empty file; please try again." >&2
    return 1
  fi
  mv -f "$part" "$dest"
  echo "    saved $dest ($(file_size "$dest"))"
  return 0
}

FORCE=0
WITH_FOUNDATION=0
while [ $# -gt 0 ]; do
  case "$1" in
    --force) FORCE=1 ;;
    --with-foundation) WITH_FOUNDATION=1 ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "error: unknown option: $1" >&2
      echo >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

if ! command -v curl > /dev/null 2>&1; then
  echo "error: curl is required but was not found on PATH." >&2
  echo "       Install curl, or use the Hugging Face CLI instead:" >&2
  echo "       hf download fridolin22/FAHR-Face $AGE_FILE --local-dir $CHECKPOINT_DIR" >&2
  exit 1
fi

# Always work from the repository root so checkpoints/ lands where `python -m app`
# and run.py expect it (their default path is checkpoints/fahr_faceage.pth).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"
cd "$REPO_ROOT"
mkdir -p "$CHECKPOINT_DIR"
echo "FAHR-Face checkpoints -> $REPO_ROOT/$CHECKPOINT_DIR"

trap 'echo; echo "Interrupted. Run the script again to resume the download."; exit 130' INT

status=0
download "$AGE_FILE" || status=1
if [ "$WITH_FOUNDATION" -eq 1 ]; then
  download "$FOUNDATION_FILE" || status=1
fi

echo
echo "Reminder: FAHR-FaceSurvival weights are not publicly released."
if [ -s "$CHECKPOINT_DIR/$SURVIVAL_FILE" ]; then
  echo "          Found $CHECKPOINT_DIR/$SURVIVAL_FILE; the demo will show the relative risk score."
else
  echo "          If you have them, place the file at $CHECKPOINT_DIR/$SURVIVAL_FILE; until then the"
  echo "          demo shows the survival model as \"not available\"."
fi

if [ "$status" -eq 0 ]; then
  echo
  echo "Done. Start the demo with:  bash scripts/run_app.sh"
  echo "            or manually:    pip install -r requirements-app.txt && python -m app"
else
  echo
  echo "Some downloads failed (see above). Run the script again to resume." >&2
fi
exit "$status"
