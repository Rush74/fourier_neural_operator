# Running FNO-3D Navier-Stokes on NCI Gadi (dgxa100)

Full write-up, with the mathematics and the reasoning behind each job: `docs/fno3d_gadi.html`.

Submit every job from the repository root (the scripts use `#PBS -l wd`).

| Step | File | Queue | Resources |
|---|---|---|---|
| 1. Environment | `bash gadi/setup_env.sh` | login node | - |
| 2. Download data | `qsub gadi/fetch_data.pbs` | copyq (needs internet) | 1 CPU |
| 3. Smoke test | `qsub -I` (below) | dgxa100 | 1 GPU, 16 CPU |
| 4. Train one config | `qsub gadi/train_fno3d.pbs` | dgxa100 | 1 GPU, 16 CPU, 200 GB |
| 5. Sweep 8 configs | `qsub gadi/sweep_fno3d.pbs` | dgxa100 | 8 GPU, 128 CPU |
| 6. Generate data | `qsub gadi/gen_ns.pbs` | dgxa100 | 8 GPU, 128 CPU |
| 7. Multi-GPU training | `qsub gadi/train_ddp.pbs` | dgxa100 | 8 GPU, 128 CPU |
| 8. Inference | `qsub gadi/infer_fno3d.pbs` | dgxa100 | 1 GPU, 16 CPU |

dgxa100 rules: ncpus = 16 x ngpus, 4.5 SU per core-hour (72 SU per GPU-hour),
48 h walltime on one node. Keep memory at or below 250 GB per GPU.

Interactive smoke test (5 epochs, to time an epoch before a long run):

    qsub -I -P dv70 -q dgxa100 -l ngpus=1,ncpus=16,mem=200GB,jobfs=50GB,walltime=00:30:00 \
         -l storage=gdata/dv70+scratch/dv70+gdata/dk92 -l wd
    module use /g/data/dk92/apps/Modules/modulefiles && module load pet/2025.08
    python fourier_3d_gadi.py --data /g/data/dv70/pag548/fno/data/PDE_datasets/NavierStokes_V1e-3_N5000_T50.mat \
           --out /scratch/dv70/pag548/fno/runs/smoke --epochs 5

Python files used by these jobs:

- `fourier_3d_gadi.py`: training with CLI settings, checkpoint/resume, saved normalisers, optional torchrun DDP
- `infer_fno3d.py`: predictions from a checkpoint
- `data_generation/navier_stokes/ns_2d_gadi.py`: seeded data shards at unit time spacing on a 64x64 grid
- `gadi/merge_shards.py`: joins shards into one HDF5 file that `MatReader` reads
