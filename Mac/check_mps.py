"""
Check that FNO-3D training on MPS agrees with CPU. Exits non-zero if it does not.

Compares the forward output, every parameter's gradient, and the weights after one
optimiser step. Run it again after any PyTorch or macOS upgrade.

python Mac/check_mps.py
"""

import copy
import sys
import warnings

import torch

from fno3d_mac import MacAdam, build_model

TOLERANCE = 1e-2


def rel(a, b):
    return (torch.linalg.vector_norm((a - b).reshape(-1)) / torch.linalg.vector_norm(a.reshape(-1))).item()


def run(base, x, y, device):
    model = copy.deepcopy(base).to(device)
    optimizer = MacAdam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    out = model(x.to(device))
    loss = torch.norm((out - y.to(device)).reshape(x.shape[0], -1), dim=1).sum()
    loss.backward()
    grads = {n: p.grad.detach().cpu() for n, p in model.named_parameters()}
    optimizer.step()
    params = {n: p.detach().cpu() for n, p in model.named_parameters()}
    return out.detach().cpu(), grads, params


def main():
    warnings.filterwarnings('ignore', message='An output with one or more elements was resized')
    if not torch.backends.mps.is_available():
        print('FAIL: MPS is not available')
        return 1

    torch.manual_seed(0)
    base = build_model(modes=4, width=8, t_in=10)
    x, y = torch.randn(2, 32, 32, 20, 10), torch.randn(2, 32, 32, 20, 1)
    out_c, grad_c, par_c = run(base, x, y, torch.device('cpu'))
    out_m, grad_m, par_m = run(base, x, y, torch.device('mps'))

    output_diff = rel(out_c, out_m)
    grad_diff = max(rel(grad_c[n], grad_m[n]) for n in grad_c)
    param_diff = max(rel(par_c[n], par_m[n]) for n in par_c)
    finite = all(torch.isfinite(p.abs()).all() for p in par_m.values())

    print('torch %s | largest relative CPU/MPS difference: output %.1e, gradient %.1e, weights %.1e'
          % (torch.__version__, output_diff, grad_diff, param_diff))
    if finite and max(output_diff, grad_diff, param_diff) < TOLERANCE:
        print('PASS: MPS training agrees with CPU')
        return 0
    print('FAIL: do not train on MPS with this PyTorch version; use --device cpu')
    return 1


if __name__ == '__main__':
    sys.exit(main())
