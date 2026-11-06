#!/bin/bash
# One-off environment setup. Run on a Gadi LOGIN node (compute nodes have no internet).
# Usage: bash gadi/setup_env.sh            (replace ab12 with your NCI project code first)
set -euo pipefail
PROJ=ab12
PYMOD=python3/3.11.7          # check with: module avail python3

module load $PYMOD
ENV=/g/data/$PROJ/$USER/envs/fno
python3 -m venv $ENV
source $ENV/bin/activate
pip install --upgrade pip
# CUDA wheel bundles its own runtime; the A100 driver on the DGX node is enough
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install numpy scipy h5py matplotlib gdown
mkdir -p /g/data/$PROJ/$USER/fno/data /scratch/$PROJ/$USER/fno/runs
echo "Environment ready at $ENV"
