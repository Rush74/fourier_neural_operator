# Running FNO-3D Navier-Stokes on NCI Gadi (dgxa100)

Full write-up, with the mathematics and the reasoning behind each job: `docs/fno3d_gadi.html`.

Before you start, replace the placeholder project code `ab12` with yours:

    sed -i 's/ab12/<your-project>/g' gadi/*.pbs gadi/setup_env.sh

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

    qsub -I -P ab12 -q dgxa100 -l ngpus=1,ncpus=16,mem=200GB,jobfs=50GB,walltime=00:30:00 \
         -l storage=gdata/ab12+scratch/ab12 -l wd
    module load python3/3.11.7 && source /g/data/ab12/$USER/envs/fno/bin/activate
    python fourier_3d_gadi.py --data /g/data/ab12/$USER/fno/data/NavierStokes_V1e-3_N5000_T50.mat \
           --out /scratch/ab12/$USER/fno/runs/smoke --epochs 5

Python files used by these jobs:

- `fourier_3d_gadi.py`: training with CLI settings, checkpoint/resume, saved normalisers, optional torchrun DDP
- `infer_fno3d.py`: predictions from a checkpoint
- `data_generation/navier_stokes/ns_2d_gadi.py`: seeded data shards at unit time spacing on a 64x64 grid
- `gadi/merge_shards.py`: joins shards into one HDF5 file that `MatReader` reads
