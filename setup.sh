#!/usr/bin/env bash
# deh-duplicate :: one-shot setup.
#
# Creates the .venv that bin/*.sh runs (that interpreter path is hard-coded
# there), installs the project into it, seeds .env, and downloads the
# MobileNetV2 weights into var/models — the TORCH_HOME deh/__init__.py pins,
# so deep/hybrid mode needs no network access afterwards.
#
# Idempotent: re-running repairs what is missing and touches nothing else.
#   ./setup.sh          install (torch from PyPI: GPU-ready, large download)
#   ./setup.sh --cpu    same, with the much smaller CPU-only torch wheels
#   ./setup.sh --dev    also install pytest + ruff (the test / static gate)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

VENV=".venv"
PY="${PYTHON:-python3}"
CPU=0
DEV=0
for arg in "$@"; do
    case "$arg" in
        --cpu)  CPU=1 ;;
        --dev)  DEV=1 ;;
        -h|--help) echo "usage: ./setup.sh [--cpu] [--dev]"; exit 0 ;;
        *) echo "setup: unknown option '$arg' (options: --cpu --dev)" >&2; exit 2 ;;
    esac
done

say()  { printf '\033]0;DEH-SETUP\007%s\n' "$*"; }      # same title idiom as bin/*.sh
fail() { printf 'setup: %s\n' "$*" >&2; exit 1; }

# --- 1. interpreter: the venv and bin/*.sh must agree on a 3.10+ python -----
command -v "${PY}" >/dev/null 2>&1 \
    || fail "'${PY}' not found; install Python 3.10+ or run PYTHON=/path/to/python ./setup.sh"
"${PY}" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' \
    || fail "Python 3.10+ required, found '$("${PY}" -V 2>&1)'"

# --- 2. venv -----------------------------------------------------------------
if [ -x "${VENV}/bin/python" ]; then
    say "venv: ${VENV} already there, reusing it"
else
    say "venv: creating ${VENV}"
    "${PY}" -m venv "${VENV}" \
        || fail "'${PY} -m venv' failed (Debian/Ubuntu: apt install python3-venv)"
fi
PIP="${VENV}/bin/pip"

# --- 3. dependencies ---------------------------------------------------------
say "deps: pip install -e . (the PyPI torch wheel is a large download)"
"${PIP}" install --upgrade pip setuptools wheel >/dev/null
if [ "${CPU}" -eq 1 ]; then
    say "deps: CPU-only torch wheels from download.pytorch.org"
    "${PIP}" install --index-url https://download.pytorch.org/whl/cpu torch torchvision
fi
"${PIP}" install -e .
if [ "${DEV}" -eq 1 ]; then
    "${PIP}" install -e ".[dev]" -r requirements-dev.txt
fi

# --- 4. .env -----------------------------------------------------------------
if [ -f .env ]; then
    say "config: .env already there, leaving it alone"
else
    cp .env.example .env
    say "config: copied annotated .env.example -> .env"
fi

# --- 5. model weights --------------------------------------------------------
# Let torchvision do the download: it owns the URL, the sha256 and the
# TORCH_HOME layout, so nothing is hard-coded here and the checkpoint lands
# exactly where deh/__init__.py points TORCH_HOME. It is a no-op when the
# file is already cached.
say "model: MobileNetV2 weights -> var/models/hub/checkpoints"
TORCH_HOME="${SCRIPT_DIR}/var/models" "${VENV}/bin/python" - <<'PY'
import os
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

weights = MobileNet_V2_Weights.DEFAULT
mobilenet_v2(weights=weights)          # fetch + sha256-verify into TORCH_HOME
print(f"model: {weights.url.rsplit('/', 1)[-1]} cached under {os.environ['TORCH_HOME']}")
PY

# --- 6. smoke test -----------------------------------------------------------
TORCH_HOME="${SCRIPT_DIR}/var/models" "${VENV}/bin/python" -c \
    'import deh, torch; print(f"smoke: deh {deh.__version__}, torch {torch.__version__} — imports OK")'

say "done."
say "  run:   ./bin/deh-duplicate.sh ./photos --dry-run"
say "  test:  .venv/bin/python -m pytest -q"
