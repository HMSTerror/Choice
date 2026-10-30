#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
python_bin="${CHOICE_SETUP_PYTHON:-python3}"
"$python_bin" -c 'import sys; assert (3, 10) <= sys.version_info[:2] <= (3, 12), "Use Python 3.10, 3.11 or 3.12"'
if [[ ! -x .venv/bin/python ]]; then
  "$python_bin" -m venv .venv
fi
.venv/bin/python -m pip install --upgrade pip
if [[ "${1:-}" == --cpu ]]; then
  .venv/bin/python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
else
  .venv/bin/python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
fi
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -c 'import torch, infomap; print("PyTorch:", torch.__version__, "CUDA:", torch.cuda.is_available())'
.venv/bin/python -c 'import faiss; print("Faiss:", faiss.__version__)'
