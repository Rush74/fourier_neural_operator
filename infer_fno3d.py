"""
Run a trained FNO-3D (from fourier_3d_gadi.py) on new Navier-Stokes data.

Reads the first t_in snapshots of field 'u' in --data, predicts the next t snapshots,
and writes them to --out as 'pred' [N, S, S, t]. If the file also holds the true
future snapshots, the relative L2 error per sample is printed and saved as 'rel_l2'.

python infer_fno3d.py --ckpt runs/v1e-3/checkpoint.pt --data NavierStokes_V1e-3_N5000_T50.mat \
                      --out pred/v1e-3.mat --first 4800 --n 200
"""

import argparse
import os

import numpy as np
import scipy.io
import torch

from utilities3 import MatReader
from fourier_3d_gadi import FNO3d


def main():
    p = argparse.ArgumentParser(description="FNO-3D inference")
    p.add_argument('--ckpt', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--first', type=int, default=0, help="index of the first sample to use")
    p.add_argument('--n', type=int, default=None, help="number of samples (default: all from --first)")
    p.add_argument('--batch', type=int, default=50)
    args = p.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.backends.cuda.matmul.allow_tf32 = True

    s = torch.load(args.ckpt, map_location=device)
    a = s['args']
    T_in, T, sub = a['t_in'], a['t'], a['sub']
    model = FNO3d(a['modes'], a['modes'], a['modes'], a['width'], t_in=T_in).to(device)
    model.load_state_dict(s['model'])
    model.eval()
    a_mean, a_std = [t.to(device) for t in s['a_norm']]
    y_mean, y_std = [t.to(device) for t in s['y_norm']]
    eps = 1e-5  # same as UnitGaussianNormalizer

    u = MatReader(args.data).read_field('u')
    end = u.shape[0] if args.n is None else args.first + args.n
    u = u[args.first:end, ::sub, ::sub]
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
