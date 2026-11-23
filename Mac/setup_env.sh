#!/bin/bash
# One-off setup: clone the working fire-emulator environment into a separate fno-mac
# environment, then check MPS. Later changes to either environment do not affect the other.
# Usage: bash Mac/setup_env.sh
set -euo pipefail

ENV=fno-mac
SOURCE=fire-emulator
HERE="$(cd "$(dirname "$0")" && pwd)"
CONDA_BASE="$(conda info --base)"

if [ -x "$CONDA_BASE/envs/$ENV/bin/python" ]; then
  echo "Environment $ENV already exists."
else
  conda create -y -n "$ENV" --clone "$SOURCE"
fi

"$CONDA_BASE/envs/$ENV/bin/python" -c "import torch, numpy, scipy, h5py; print('torch', torch.__version__, 'mps', torch.backends.mps.is_available())"
"$CONDA_BASE/envs/$ENV/bin/python" "$HERE/check_mps.py"
