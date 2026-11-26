"""
Cluster-ready version of fourier_3d.py (FNO-3D for 2D Navier-Stokes, space + time).

Same model as fourier_3d.py, plus:
  - command-line settings (data path, output dir, modes, width, epochs, ...)
  - checkpoint every N epochs and automatic resume from <out>/checkpoint.pt
  - normaliser statistics saved in the checkpoint, so infer_fno3d.py can decode outputs
  - zero-copy .expand() of the input in time instead of .repeat()
  - optional multi-GPU data-parallel training when launched with torchrun

Single GPU:  python fourier_3d_gadi.py --data NavierStokes_V1e-3_N5000_T50.mat --out runs/v1e-3
Multi GPU:   torchrun --standalone --nproc_per_node=8 fourier_3d_gadi.py --data ... --out ...
See gadi/ for PBS job scripts and docs/fno3d_gadi.html for the full write-up.
"""

import argparse
import os
from timeit import default_timer

import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from utilities3 import MatReader, UnitGaussianNormalizer, LpLoss, count_params
from Adam import Adam


################################################################
# 3d fourier layers (same as fourier_3d.py)
################################################################

class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x


class FNO3d(nn.Module):
    """
    input:  first T_in snapshots repeated along time + (x, y, t) grid, shape (batch, x, y, t, T_in + 3)
    output: next T snapshots, shape (batch, x, y, t, 1)
    The unused BatchNorm layers of fourier_3d.py are left out (they would break DDP).
    """
    def __init__(self, modes1, modes2, modes3, width, t_in=10):
        super(FNO3d, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.padding = 6  # pad the time axis, which is not periodic
        self.fc0 = nn.Linear(t_in + 3, self.width)

        self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        self.w2 = nn.Conv3d(self.width, self.width, 1)
        self.w3 = nn.Conv3d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = F.pad(x, [0, self.padding])

        x = F.gelu(self.conv0(x) + self.w0(x))
        x = F.gelu(self.conv1(x) + self.w1(x))
        x = F.gelu(self.conv2(x) + self.w2(x))
        x = self.conv3(x) + self.w3(x)

        x = x[..., :-self.padding]
        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)


################################################################
# training
################################################################

def parse_args():
    p = argparse.ArgumentParser(description="Train FNO-3D on Navier-Stokes data")
    p.add_argument('--data', required=True, help=".mat file with field 'u' of shape [N, S, S, T_total]")
    p.add_argument('--out', required=True, help="run directory (checkpoint, predictions)")
    p.add_argument('--ntrain', type=int, default=1000)
    p.add_argument('--ntest', type=int, default=200)
    p.add_argument('--sub', type=int, default=1, help="spatial subsampling factor")
    p.add_argument('--t-in', type=int, default=10)
    p.add_argument('--t', type=int, default=40, help="number of output time steps")
    p.add_argument('--modes', type=int, default=8)
    p.add_argument('--width', type=int, default=20)
    p.add_argument('--epochs', type=int, default=500)
    p.add_argument('--batch', type=int, default=10, help="batch size per GPU")
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--step', type=int, default=100, help="StepLR step size (epochs)")
    p.add_argument('--gamma', type=float, default=0.5)
    p.add_argument('--ckpt-every', type=int, default=10)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--no-pred', action='store_true', help="skip writing test predictions at the end")
    return p.parse_args()


def evaluate(net, loader, y_norm, myloss, device, S, T):
    net.eval()
    total, preds = 0.0, []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = y_norm.decode(net(x).view(x.shape[0], S, S, T))
            total += myloss(out.view(x.shape[0], -1), y.view(x.shape[0], -1)).item()
            preds.append(out.cpu())
    return total, torch.cat(preds)


def main():
    args = parse_args()

    ddp = 'LOCAL_RANK' in os.environ
    if ddp:
        dist.init_process_group('nccl')
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        rank, world = dist.get_rank(), dist.get_world_size()
    else:
        local_rank, rank, world = 0, 0, 1
    is_main = rank == 0
    device = torch.device('cuda', local_rank) if torch.cuda.is_available() else torch.device('cpu')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True  # A100: TF32 for Linear/einsum; FFTs stay FP32
    torch.backends.cudnn.allow_tf32 = True
    if is_main:
        os.makedirs(args.out, exist_ok=True)
        print(vars(args), flush=True)

    ################################################################
    # data
    ################################################################
    t0 = default_timer()
    T_in, T, sub = args.t_in, args.t, args.sub
    u = MatReader(args.data).read_field('u')
    assert args.ntrain + args.ntest <= u.shape[0], "ntrain + ntest exceeds samples in file"
    assert T_in + T <= u.shape[-1], "file has fewer time steps than t_in + t"

    train_a = u[:args.ntrain, ::sub, ::sub, :T_in]
    train_u = u[:args.ntrain, ::sub, ::sub, T_in:T_in + T]
    test_a = u[-args.ntest:, ::sub, ::sub, :T_in]
    test_u = u[-args.ntest:, ::sub, ::sub, T_in:T_in + T]
    del u
    S = train_u.shape[1]

    a_norm = UnitGaussianNormalizer(train_a)
    y_norm = UnitGaussianNormalizer(train_u)
    train_a, test_a = a_norm.encode(train_a), a_norm.encode(test_a)
    train_u = y_norm.encode(train_u)
    y_norm.mean, y_norm.std = y_norm.mean.to(device), y_norm.std.to(device)

    # zero-copy broadcast along time (fourier_3d.py uses .repeat, a 40x copy)
    train_a = train_a.reshape(-1, S, S, 1, T_in).expand(-1, -1, -1, T, -1)
    test_a = test_a.reshape(-1, S, S, 1, T_in).expand(-1, -1, -1, T, -1)

    train_ds = TensorDataset(train_a, train_u)
    sampler = DistributedSampler(train_ds, shuffle=True, seed=args.seed) if ddp else None
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=(sampler is None), sampler=sampler)
    test_loader = DataLoader(TensorDataset(test_a, test_u), batch_size=args.batch, shuffle=False)
    if is_main:
        print('train', tuple(train_u.shape), 'test', tuple(test_u.shape),
              'preprocessing %.1fs' % (default_timer() - t0), flush=True)

    ################################################################
    # model, optimiser, resume
    ################################################################
    model = FNO3d(args.modes, args.modes, args.modes, args.width, t_in=T_in).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=args.gamma)

    ckpt_path = os.path.join(args.out, 'checkpoint.pt')
    start = 0
    if os.path.exists(ckpt_path):
        s = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(s['model'])
        optimizer.load_state_dict(s['opt'])
        scheduler.load_state_dict(s['sched'])
        start = s['epoch'] + 1
        if is_main:
            print('resumed from epoch', s['epoch'], flush=True)
    if is_main:
        print('parameters', count_params(model), flush=True)

    net = model
    if ddp:
        net = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    def save_checkpoint(ep):
        torch.save({'model': model.state_dict(), 'opt': optimizer.state_dict(),
                    'sched': scheduler.state_dict(), 'epoch': ep, 'args': vars(args),
                    'a_norm': (a_norm.mean.cpu(), a_norm.std.cpu()),
                    'y_norm': (y_norm.mean.cpu(), y_norm.std.cpu())}, ckpt_path + '.tmp')
        os.replace(ckpt_path + '.tmp', ckpt_path)  # atomic: never a half-written checkpoint

    ################################################################
    # training loop
    ################################################################
    myloss = LpLoss(size_average=False)
    for ep in range(start, args.epochs):
        if sampler is not None:
            sampler.set_epoch(ep)
        net.train()
        t1 = default_timer()
        train_mse, train_l2, nbatch = 0.0, 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            b = x.shape[0]
            optimizer.zero_grad()
            out = net(x).view(b, S, S, T)

            mse = F.mse_loss(out, y, reduction='mean')
            l2 = myloss(y_norm.decode(out).view(b, -1), y_norm.decode(y).view(b, -1))
            l2.backward()
            optimizer.step()

            train_mse += mse.item()
            train_l2 += l2.item()
            nbatch += 1
        scheduler.step()

        if ddp:
            stats = torch.tensor([train_mse, train_l2, nbatch], device=device, dtype=torch.float64)
            dist.all_reduce(stats)
            train_mse, train_l2, nbatch = stats.tolist()

        if is_main:
            test_l2, _ = evaluate(model, test_loader, y_norm, myloss, device, S, T)
            print(ep, '%.2f' % (default_timer() - t1), train_mse / nbatch,
                  train_l2 / args.ntrain, test_l2 / args.ntest, flush=True)
            if (ep + 1) % args.ckpt_every == 0 or ep == args.epochs - 1:
                save_checkpoint(ep)
        if ddp:
            dist.barrier()

    ################################################################
    # final predictions on the test set
    ################################################################
    if is_main and not args.no_pred:
        test_l2, pred = evaluate(model, test_loader, y_norm, myloss, device, S, T)
        print('final test rel. L2', test_l2 / args.ntest, flush=True)
        scipy.io.savemat(os.path.join(args.out, 'pred.mat'),
                         mdict={'pred': pred.numpy(), 'u': test_u.numpy()})

    if ddp:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
