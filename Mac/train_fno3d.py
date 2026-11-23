"""
Train FNO-3D on Navier-Stokes data on an Apple Silicon Mac.

Same model, data split, loss and schedule as fourier_3d_gadi.py. Checkpoints are written
to <out>/checkpoint.pt and a rerun with the same --out resumes from the latest one.

python Mac/train_fno3d.py --data NavierStokes_V1e-3_N5000_T50.mat --out runs/v1e-3
"""

import argparse
import math
import os
import warnings
from timeit import default_timer

import numpy as np
import scipy.io
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from fno3d_mac import MacAdam, build_model, get_device, sync
from utilities3 import LpLoss, MatReader, UnitGaussianNormalizer, count_params

# Settings that must match for a checkpoint to be resumed.
RESUME_KEYS = ('ntrain', 'ntest', 'sub', 't_in', 't', 'modes', 'width', 'seed')


def parse_args():
    p = argparse.ArgumentParser(description="Train FNO-3D on Navier-Stokes data (Mac)")
    p.add_argument('--data', required=True, help=".mat file with field 'u' of shape [N, S, S, T_total]")
    p.add_argument('--out', required=True, help="run directory (checkpoint, predictions)")
    p.add_argument('--device', choices=['auto', 'mps', 'cpu'], default='auto')
    p.add_argument('--ntrain', type=int, default=1000)
    p.add_argument('--ntest', type=int, default=200)
    p.add_argument('--sub', type=int, default=1, help="spatial subsampling factor")
    p.add_argument('--t-in', type=int, default=10)
    p.add_argument('--t', type=int, default=40, help="number of output time steps")
    p.add_argument('--modes', type=int, default=8)
    p.add_argument('--width', type=int, default=20)
    p.add_argument('--epochs', type=int, default=500)
    p.add_argument('--batch', type=int, default=10)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--step', type=int, default=100, help="StepLR step size (epochs)")
    p.add_argument('--gamma', type=float, default=0.5)
    p.add_argument('--ckpt-every', type=int, default=5)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--no-pred', action='store_true', help="skip writing test predictions at the end")
    return p.parse_args()


def load_data(args):
    u = MatReader(args.data).read_field('u')
    assert args.ntrain + args.ntest <= u.shape[0], "ntrain + ntest exceeds samples in file"
    assert args.t_in + args.t <= u.shape[-1], "file has fewer time steps than t_in + t"

    # clone() so these slices do not keep the whole dataset in memory
    sub, T_in, T = args.sub, args.t_in, args.t
    train_a = u[:args.ntrain, ::sub, ::sub, :T_in].clone()
    train_u = u[:args.ntrain, ::sub, ::sub, T_in:T_in + T].clone()
    test_a = u[-args.ntest:, ::sub, ::sub, :T_in].clone()
    test_u = u[-args.ntest:, ::sub, ::sub, T_in:T_in + T].clone()
    return train_a, train_u, test_a, test_u


def evaluate(model, loader, y_norm, myloss, device, S, T):
    model.eval()
    total, preds = 0.0, []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = y_norm.decode(model(x).view(x.shape[0], S, S, T))
            total += myloss(out.view(x.shape[0], -1), y.view(x.shape[0], -1)).item()
            preds.append(out.cpu())
    return total, torch.cat(preds)


def main():
    warnings.filterwarnings('ignore', message='An output with one or more elements was resized')
    args = parse_args()
    device = get_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    print('device', device, vars(args), flush=True)

    t0 = default_timer()
    T_in, T = args.t_in, args.t
    train_a, train_u, test_a, test_u = load_data(args)
    S = train_u.shape[1]

    a_norm = UnitGaussianNormalizer(train_a)
    y_norm = UnitGaussianNormalizer(train_u)
    train_a, test_a = a_norm.encode(train_a), a_norm.encode(test_a)
    train_u = y_norm.encode(train_u)
    y_norm.mean, y_norm.std = y_norm.mean.to(device), y_norm.std.to(device)

    train_a = train_a.reshape(-1, S, S, 1, T_in).expand(-1, -1, -1, T, -1)
    test_a = test_a.reshape(-1, S, S, 1, T_in).expand(-1, -1, -1, T, -1)
    train_loader = DataLoader(TensorDataset(train_a, train_u), batch_size=args.batch, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_a, test_u), batch_size=args.batch, shuffle=False)
    print('train', tuple(train_u.shape), 'test', tuple(test_u.shape),
          'preprocessing %.1fs' % (default_timer() - t0), flush=True)

    model = build_model(args.modes, args.width, T_in).to(device)
    optimizer = MacAdam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=args.gamma)

    ckpt_path = os.path.join(args.out, 'checkpoint.pt')
    start = 0
    if os.path.exists(ckpt_path):
        s = torch.load(ckpt_path, map_location=device)
        changed = [k for k in RESUME_KEYS if s['args'].get(k) != getattr(args, k)]
        if changed:
            raise SystemExit('checkpoint in %s used different settings: %s. Use a new --out.'
                             % (args.out, ', '.join(changed)))
        model.load_state_dict(s['model'])
        optimizer.load_state_dict(s['opt'])
        scheduler.load_state_dict(s['sched'])
        start = s['epoch'] + 1
        print('resumed from epoch', s['epoch'], flush=True)
    print('parameters', count_params(model), flush=True)

    def save_checkpoint(ep):
        torch.save({'model': model.state_dict(), 'opt': optimizer.state_dict(),
                    'sched': scheduler.state_dict(), 'epoch': ep, 'args': vars(args),
                    'a_norm': (a_norm.mean.cpu(), a_norm.std.cpu()),
                    'y_norm': (y_norm.mean.cpu(), y_norm.std.cpu())}, ckpt_path + '.tmp')
        os.replace(ckpt_path + '.tmp', ckpt_path)

    myloss = LpLoss(size_average=False)
    print('epoch seconds train_mse train_l2 test_l2 eta_hours', flush=True)
    for ep in range(start, args.epochs):
        model.train()
        t1 = default_timer()
        train_mse, train_l2, nbatch = 0.0, 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            b = x.shape[0]
            optimizer.zero_grad()
            out = model(x).view(b, S, S, T)
            mse = F.mse_loss(out, y, reduction='mean')
            l2 = myloss(y_norm.decode(out).view(b, -1), y_norm.decode(y).view(b, -1))
            l2.backward()
            optimizer.step()
            train_mse += mse.item()
            train_l2 += l2.item()
            nbatch += 1
        scheduler.step()

        if not math.isfinite(train_l2):
            raise SystemExit('non-finite loss at epoch %d; the last checkpoint was kept' % ep)

        test_l2, _ = evaluate(model, test_loader, y_norm, myloss, device, S, T)
        sync(device)
        seconds = default_timer() - t1
        print(ep, '%.2f' % seconds, train_mse / nbatch, train_l2 / args.ntrain, test_l2 / args.ntest,
              '%.1f' % (seconds * (args.epochs - ep - 1) / 3600), flush=True)
        if (ep + 1) % args.ckpt_every == 0 or ep == args.epochs - 1:
            save_checkpoint(ep)

    if not args.no_pred and start < args.epochs:
        test_l2, pred = evaluate(model, test_loader, y_norm, myloss, device, S, T)
        print('final test rel. L2', test_l2 / args.ntest, flush=True)
        scipy.io.savemat(os.path.join(args.out, 'pred.mat'),
                         mdict={'pred': pred.numpy(), 'u': test_u.numpy()})


if __name__ == '__main__':
    main()
