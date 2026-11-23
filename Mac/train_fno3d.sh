#!/bin/bash
# Train FNO-3D on this Mac. Rerun the same command to resume from the latest checkpoint.
# Extra arguments go to Mac/train_fno3d.py, for example:
#   bash Mac/train_fno3d.sh --epochs 100
#   FNO_RUN=v1e-3_m12_w32 bash Mac/train_fno3d.sh --modes 12 --width 32
#
# Optional environment variables: FNO_ROOT, FNO_DATA, FNO_RUN, FNO_PYTHON
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="${FNO_ROOT:-/Volumes/PG Backups T9/fno}"
DATA="${FNO_DATA:-$ROOT/data/PDE_datasets/NavierStokes_V1e-3_N5000_T50.mat}"
RUN="${FNO_RUN:-v1e-3_m8_w20}"
PY="${FNO_PYTHON:-$HOME/miniconda3/envs/fno-mac/bin/python}"
OUT="$ROOT/runs/$RUN"

[ -x "$PY" ] || { echo "Python not found: $PY. Run: bash Mac/setup_env.sh" >&2; exit 1; }
[ -f "$DATA" ] || { echo "Dataset not found: $DATA. Is the T9 mounted?" >&2; exit 1; }
mkdir -p "$OUT"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

# Skip the MPS check only when the run is explicitly on CPU.
case " $* " in
  *" --device cpu "*) ;;
  *) "$PY" "$REPO/Mac/check_mps.py" ;;
esac

# caffeinate keeps the Mac awake for the length of the run.
caffeinate -ims "$PY" -u "$REPO/Mac/train_fno3d.py" --data "$DATA" --out "$OUT" "$@" 2>&1 | tee -a "$OUT/train.log"
