# Running FNO-3D Navier-Stokes on a Mac Studio (M1 Max)

Run every command from the repository root.

| Step | Command |
|---|---|
| 1. Create the `fno-mac` environment and check MPS | `bash Mac/setup_env.sh` |
| 2. Short test on the real data | `bash Mac/smoke_test.sh` |
| 3. Train, or resume after a stop | `bash Mac/train_fno3d.sh` |
| 4. Inference | see below |

Data and outputs live on the external drive:

    /Volumes/PG Backups T9/fno/data/PDE_datasets/NavierStokes_V1e-3_N5000_T50.mat
    /Volumes/PG Backups T9/fno/runs/<run>/{checkpoint.pt,train.log,pred.mat}

Keep run outputs out of `~/Documents`, which is synced by iCloud.

## Why this differs from the Gadi version

PyTorch 2.10 on MPS has two problems that affect this model:

- Backpropagation through complex `torch.einsum` returns wrong gradients, while forward
  predictions stay correct. `Mac/fno3d_mac.py` uses real-valued einsums instead.
- The complex second-moment update in `Adam.py` can produce NaN weights for some very small
  gradients. `MacAdam` stores the second moment as a real tensor.

`Mac/check_mps.py` compares CPU and MPS. Rerun it after any PyTorch or macOS upgrade.
`Mac/train_fno3d.sh` runs it automatically before MPS training.

## Common options

    bash Mac/train_fno3d.sh --device cpu
    bash Mac/train_fno3d.sh --epochs 100
    FNO_RUN=v1e-3_m12_w32 bash Mac/train_fno3d.sh --modes 12 --width 32

A run directory can only resume with the same model and data settings. Use a new `FNO_RUN`
when changing them.

## Inference

    PY=~/miniconda3/envs/fno-mac/bin/python
    ROOT="/Volumes/PG Backups T9/fno"
    $PY Mac/infer_fno3d.py --ckpt "$ROOT/runs/v1e-3_m8_w20/checkpoint.pt" \
        --data "$ROOT/data/PDE_datasets/NavierStokes_V1e-3_N5000_T50.mat" \
        --out "$ROOT/runs/v1e-3_m8_w20/infer.mat" --first 4800 --n 200
