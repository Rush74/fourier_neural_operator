#!/bin/bash
# Submit both halves of the FNO-3D sweep to gpuvolta (Gadi has no PBS job arrays).
# Run it directly (not with qsub), from anywhere: bash gadi/submit_sweep_gpuvolta.sh
# It changes to the repository root first, because the jobs use `#PBS -l wd`.
cd "$(dirname "$(readlink -f "$0")")/.." || exit 1
for half in 0 1; do
  qsub -v HALF=$half gadi/sweep_fno3d_gpuvolta.pbs
done
