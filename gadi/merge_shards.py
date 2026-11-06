"""
Merge dataset shards from ns_2d_gadi.py into one file that MatReader can load.

Writes HDF5 (MATLAB v7.3 style, axes reversed) because MATLAB v5 .mat files written by
scipy cannot hold a single variable over 2 GB, e.g. 5000 x 64 x 64 x 50 float32 = 4.1 GB.
MatReader falls back to h5py and reverses the axes back, so read_field('u') is [N, S, S, T].

python gadi/merge_shards.py --out NavierStokes_V1e-3_N5120_T50.mat ns_0.mat ns_1.mat ...
"""

import argparse

import h5py
import numpy as np
import scipy.io


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('shards', nargs='+')
    args = p.parse_args()

    a, u, t = [], [], None
    for path in args.shards:
        d = scipy.io.loadmat(path)
        a.append(d['a'].astype(np.float32))
        u.append(d['u'].astype(np.float32))
        t = d['t']
        print(path, d['u'].shape)
    a, u = np.concatenate(a), np.concatenate(u)

    with h5py.File(args.out, 'w') as f:
        f.create_dataset('a', data=a.transpose(range(a.ndim - 1, -1, -1)))
        f.create_dataset('u', data=u.transpose(range(u.ndim - 1, -1, -1)))
        f.create_dataset('t', data=np.asarray(t, dtype=np.float32))
    print('wrote', args.out, u.shape)


if __name__ == '__main__':
    main()
