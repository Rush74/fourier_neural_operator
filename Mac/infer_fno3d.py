"""
Run a trained FNO-3D checkpoint (from the Mac or Gadi) on Navier-Stokes data.

python Mac/infer_fno3d.py --ckpt runs/v1e-3/checkpoint.pt --data NavierStokes_V1e-3_N5000_T50.mat \
                          --out runs/v1e-3/infer.mat --first 4800 --n 200
"""

import argparse
import os
import warnings

import numpy as np
import scipy.io
import torch

from fno3d_mac import build_model, get_device
from utilities3 import MatReader


def main():
    warnings.filterwarnings('ignore', message='An output with one or more elements was resized')
    p = argparse.ArgumentParser(description="FNO-3D inference (Mac)")
    p.add_argument('--ckpt', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--device', choices=['auto', 'mps', 'cpu'], default='auto')
    p.add_argument('--first', type=int, default=0, help="index of the first sample to use")
    p.add_argument('--n', type=int, default=None, help="number of samples (default: all from --first)")
    p.add_argument('--batch', type=int, default=10)
    args = p.parse_args()

    device = get_device(args.device)
    s = torch.load(args.ckpt, map_location='cpu')
    a = s['args']
    T_in, T, sub = a['t_in'], a['t'], a['sub']
    model = build_model(a['modes'], a['width'], T_in)
    model.load_state_dict(s['model'])
    model = model.to(device).eval()
    a_mean, a_std = [t.to(device) for t in s['a_norm']]
    y_mean, y_std = [t.to(device) for t in s['y_norm']]
    eps = 1e-5  # same as UnitGaussianNormalizer

    u = MatReader(args.data).read_field('u')
    end = u.shape[0] if args.n is None else args.first + args.n
    u = u[args.first:end, ::sub, ::sub].clone()
    S = u.shape[1]
    assert S == a_mean.shape[0], \
        "grid %d differs from training grid %d: normaliser is per grid point" % (S, a_mean.shape[0])
    has_truth = u.shape[-1] >= T_in + T

    preds, errs = [], []
    with torch.no_grad():
        for i in range(0, u.shape[0], args.batch):
            w = u[i:i + args.batch].to(device)
            x = (w[..., :T_in] - a_mean) / (a_std + eps)
            x = x.unsqueeze(3).expand(-1, -1, -1, T, -1)
            pred = model(x).squeeze(-1) * (y_std + eps) + y_mean
            preds.append(pred.cpu())
            if has_truth:
                y = w[..., T_in:T_in + T]
                b = y.shape[0]
                err = torch.norm((pred - y).reshape(b, -1), dim=1) / torch.norm(y.reshape(b, -1), dim=1)
                errs.append(err.cpu())

    pred = torch.cat(preds).numpy()
    out = {'pred': pred}
    if has_truth:
        rel = torch.cat(errs).numpy()
        out['rel_l2'] = rel
        print('samples', len(rel), 'mean rel. L2', float(np.mean(rel)))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    scipy.io.savemat(args.out, mdict=out)
    print('wrote', args.out, pred.shape)


if __name__ == '__main__':
    main()
