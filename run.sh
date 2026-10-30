#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ -n "${CHOICE_PYTHON:-}" ]]; then
  python_bin="$CHOICE_PYTHON"
elif [[ -x .venv/bin/python ]]; then
  python_bin=.venv/bin/python
else
  python_bin=python3
fi
exec "$python_bin" -B -u experiments.py "$@"
