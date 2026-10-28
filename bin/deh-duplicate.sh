#!/usr/bin/env bash
# deh-duplicate :: find & quarantine visually similar images.
# Thin launcher: checks .venv (points at ./setup.sh when it is missing), sets the
# terminal title, loads project .env, and runs the CLI from the caller's directory.
set -euo pipefail

printf "\033]0;DEH-DUPLICATE\007"                     # set terminal window title


SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [ ! -x "${PROJECT_DIR}/.venv/bin/python" ]; then
    echo "deh-duplicate: no .venv in ${PROJECT_DIR} — run ./setup.sh first." >&2
    exit 1
fi

if [ -f "${PROJECT_DIR}/.env" ]; then
    while IFS= read -r line || [ -n "$line" ]; do
        if [[ "$line" =~ ^[[:space:]]*(#|$) ]]; then
            continue                                  # skip blank lines and comments
        fi
        line="${line#export }"                        # tolerate "export KEY=..."
        if [[ "$line" != *=* || -z "${line%%=*}" ]]; then
            continue                                  # skip malformed lines
        fi
        key="${line%%=*}"
        value="${line#*=}"
        case "$value" in                             # strip one layer of matching quotes
            \"*\") value="${value#\"}"; value="${value%\"}" ;;
            \'*\') value="${value#\'}"; value="${value%\'}" ;;
        esac
        export "${key}=${value}"
    done < "${PROJECT_DIR}/.env"
fi

exec env PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PROJECT_DIR}/.venv/bin/python" -m deh.cli.duplicate "$@"
