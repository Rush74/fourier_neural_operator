#!/bin/bash
# Short end-to-end run on the real dataset: 100 training samples, 20 test samples, 2 epochs.
# Each run gets a new directory, so it never resumes or overwrites a real training run.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
FNO_RUN="smoke_$(date +%Y%m%d_%H%M%S)" exec bash "$HERE/train_fno3d.sh" \
  --ntrain 100 --ntest 20 --epochs 2 --ckpt-every 1 "$@"
