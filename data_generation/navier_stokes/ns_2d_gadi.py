"""
Seeded, configurable version of ns_2d.py for generating dataset shards on a cluster.

Solves 2D Navier-Stokes (vorticity form) on the unit torus with the same pseudo-spectral
Crank-Nicolson scheme as ns_2d.py, then stores snapshots at unit time spacing on a
subsampled grid, matching the published NavierStokes_*.mat layout:
    a: [N, s/sub, s/sub]       initial vorticity
    u: [N, s/sub, s/sub, T]    vorticity at t = 1, 2, ..., T   (record_steps = T)
    t: [T]

python ns_2d_gadi.py --seed 0 --n 640 --visc 1e-3 --T 50 --out ns_0.mat
Run from the repository root or from this folder.
"""

import argparse
import math
import os
import sys
from timeit import default_timer

import scipy.io
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from random_fields import GaussianRF


def navier_stokes_2d(w0, f, visc, T, delta_t=1e-4, record_steps=1):
    """Identical to navier_stokes_2d in ns_2d.py."""
    N = w0.size()[-1]
    k_max = math.floor(N / 2.0)
    steps = math.ceil(T / delta_t)

    w_h = torch.fft.rfft2(w0)
    f_h = torch.fft.rfft2(f)
    if len(f_h.size()) < len(w_h.size()):
        f_h = torch.unsqueeze(f_h, 0)

    record_time = math.floor(steps / record_steps)

    k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=w0.device),
                     torch.arange(start=-k_max, end=0, step=1, device=w0.device)), 0).repeat(N, 1)
    k_x = k_y.transpose(0, 1)
    k_x = k_x[..., :k_max + 1]
    k_y = k_y[..., :k_max + 1]

    lap = 4 * (math.pi ** 2) * (k_x ** 2 + k_y ** 2)
    lap[0, 0] = 1.0
    dealias = torch.unsqueeze(torch.logical_and(torch.abs(k_y) <= (2.0 / 3.0) * k_max,
                                                torch.abs(k_x) <= (2.0 / 3.0) * k_max).float(), 0)

    sol = torch.zeros(*w0.size(), record_steps, device=w0.device)
    sol_t = torch.zeros(record_steps, device=w0.device)

    c = 0
    t = 0.0
    for j in range(steps):
        psi_h = w_h / lap

        q = torch.fft.irfft2(2. * math.pi * k_y * 1j * psi_h, s=(N, N))
        v = torch.fft.irfft2(-2. * math.pi * k_x * 1j * psi_h, s=(N, N))
        w_x = torch.fft.irfft2(2. * math.pi * k_x * 1j * w_h, s=(N, N))
        w_y = torch.fft.irfft2(2. * math.pi * k_y * 1j * w_h, s=(N, N))

        F_h = dealias * torch.fft.rfft2(q * w_x + v * w_y)

        w_h = (-delta_t * F_h + delta_t * f_h + (1.0 - 0.5 * delta_t * visc * lap) * w_h) \
            / (1.0 + 0.5 * delta_t * visc * lap)

        t += delta_t
        if (j + 1) % record_time == 0:
            sol[..., c] = torch.fft.irfft2(w_h, s=(N, N))
            sol_t[c] = t
            c += 1

    return sol, sol_t


def main():
    p = argparse.ArgumentParser(description="Generate a Navier-Stokes dataset shard")
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--n', type=int, default=640, help="samples in this shard")
    p.add_argument('--batch', type=int, default=64, help="samples solved at once on the GPU")
    p.add_argument('--visc', type=float, default=1e-3)
    p.add_argument('--T', type=float, default=50.0, help="final time; one snapshot per unit time")
    p.add_argument('--dt', type=float, default=1e-4)
    p.add_argument('--s', type=int, default=256, help="solver grid size (power of 2)")
    p.add_argument('--sub', type=int, default=4, help="subsampling of the saved grid (256/4 = 64)")
    p.add_argument('--out', required=True)
    args = p.parse_args()
    assert args.n % args.batch == 0, "--n must be a multiple of --batch"

    device = torch.device('cuda')
    torch.manual_seed(args.seed)

    s = args.s
    record_steps = int(round(args.T))
    GRF = GaussianRF(2, s, alpha=2.5, tau=7, device=device)

    t = torch.linspace(0, 1, s + 1, device=device)[:-1]
    X, Y = torch.meshgrid(t, t, indexing='ij')
    f = 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y)))

    so = s // args.sub
    a = torch.zeros(args.n, so, so)
    u = torch.zeros(args.n, so, so, record_steps)

    t0 = default_timer()
    for j in range(args.n // args.batch):
        w0 = GRF.sample(args.batch)
        sol, sol_t = navier_stokes_2d(w0, f, args.visc, args.T, args.dt, record_steps)
        i = j * args.batch
        a[i:i + args.batch] = w0[:, ::args.sub, ::args.sub].cpu()
        u[i:i + args.batch] = sol[:, ::args.sub, ::args.sub, :].cpu()
        print('seed', args.seed, 'batch', j, 'done', i + args.batch, '%.1fs' % (default_timer() - t0), flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    scipy.io.savemat(args.out, mdict={'a': a.numpy(), 'u': u.numpy(), 't': sol_t.cpu().numpy()})
    print('wrote', args.out, tuple(u.shape))


if __name__ == '__main__':
    main()
