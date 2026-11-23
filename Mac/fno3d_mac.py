"""
Apple Silicon support for the FNO-3D model in fourier_3d_gadi.py.

Two operations give wrong results on MPS (PyTorch 2.10, M1 Max) and are replaced here:
  - backpropagation through complex torch.einsum returns wrong gradients, so the spectral
    multiplication is written with real-valued einsums;
  - the complex second-moment update in Adam.py produces NaN weights for some very small
    gradients, so the second moment is stored as a real tensor, |g|^2.

Both replacements are mathematically identical to the originals. Parameter names and shapes
are unchanged, so checkpoints from Gadi and from this Mac hold the same model.
Run Mac/check_mps.py to confirm the replacements on the current PyTorch version.
"""

import math
import os
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Adam import Adam  # noqa: E402
from fourier_3d_gadi import FNO3d, SpectralConv3d  # noqa: E402


class SpectralConv3dMac(SpectralConv3d):
    def compl_mul3d(self, input, weights):
        eq = "bixyz,ioxyz->boxyz"
        real = torch.einsum(eq, input.real, weights.real) - torch.einsum(eq, input.imag, weights.imag)
        imag = torch.einsum(eq, input.real, weights.imag) + torch.einsum(eq, input.imag, weights.real)
        return torch.complex(real, imag)


def build_model(modes, width, t_in):
    model = FNO3d(modes, modes, modes, width, t_in=t_in)
    for conv in (model.conv0, model.conv1, model.conv2, model.conv3):
        conv.__class__ = SpectralConv3dMac
    return model


class MacAdam(Adam):
    """Same update as Adam.py, with the second moment held as a real tensor."""

    def __init__(self, params, **kwargs):
        if kwargs.get('amsgrad'):
            raise ValueError("MacAdam does not implement amsgrad")
        super().__init__(params, **kwargs)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p.real)
                elif state['exp_avg_sq'].is_complex():  # state saved by Adam.py
                    state['exp_avg_sq'] = state['exp_avg_sq'].real

                state['step'] += 1
                k = state['step']
                grad = p.grad
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                state['exp_avg'].mul_(beta1).add_(grad, alpha=1 - beta1)
                grad_sq = grad.real.square() + grad.imag.square() if grad.is_complex() else grad.square()
                state['exp_avg_sq'].mul_(beta2).add_(grad_sq, alpha=1 - beta2)

                denom = state['exp_avg_sq'].sqrt() / math.sqrt(1 - beta2 ** k) + group['eps']
                p.add_(state['exp_avg'] / denom, alpha=-group['lr'] / (1 - beta1 ** k))
        return loss


def get_device(name='auto'):
    if name == 'auto':
        name = 'mps' if torch.backends.mps.is_available() else 'cpu'
    if name == 'mps' and not torch.backends.mps.is_available():
        raise SystemExit("MPS was requested but is not available")
    return torch.device(name)


def sync(device):
    """Wait for queued GPU work, so timings are real."""
    if device.type == 'mps':
        torch.mps.synchronize()
