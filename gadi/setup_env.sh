#!/bin/bash
# One-off setup. Run on a Gadi LOGIN node (compute nodes have no internet).
# Every job uses the NCI "pet" environment (torch, numpy, scipy, h5py, matplotlib) as-is.
# The only extra is gdown, for fetch_data.pbs, in a small venv layered on pet's python.
# Usage: bash gadi/setup_env.sh
set -euo pipefail
PROJ=dv70

module use /g/data/dk92/apps/Modules/modulefiles
module load pet/2025.08

GDOWN=/g/data/$PROJ/pag548/envs/gdown
python3 -m venv --system-site-packages $GDOWN
$GDOWN/bin/python -m pip install --no-cache-dir gdown
$GDOWN/bin/python -c "import torch, numpy, scipy, h5py, gdown; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

mkdir -p /g/data/$PROJ/pag548/fno/data /scratch/$PROJ/pag548/fno/runs
echo "Done. Job scripts need gdata/dk92 in '#PBS -l storage=' and:"
echo "  module use /g/data/dk92/apps/Modules/modulefiles && module load pet/2025.08"
