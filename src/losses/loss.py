import torch
import timm 
import random
from torch import nn
import numpy as np

"""
This script uses the LeJEPA architecture.

Reference:
    Balestriero, R., & LeCun, Y. (2025). 
    LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics. 
    arXiv preprint arXiv:2511.08544.
    
    https://github.com/galilai-group/lejepa
"""
class SIGReg(torch.nn.Module):
    def __init__(self, knots=17):
        super().__init__()
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj, M=256):
        # proj: (N*V, D) - flattened batch of projections
        # with torch.no_grad():
            # step = dist.all_reduce(local_step,o)
        A = torch.randn(proj.size(-1), M, device=proj.device, dtype=proj.dtype) # (D, 256)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t # (N*V, 256, 1) * (knots,) -> (N*V, 256, knots)
        # err: Mean over batch (dim 0) -> (256, knots)
        err = (x_t.cos().mean(0) - self.phi).square() + x_t.sin().mean(0).square()
        statistic = (err @ self.weights) * proj.size(0) # (256,) * scalar -> scalar (after mean)
        return statistic.mean()

def LeJEPA(global_proj, all_proj, sigreg, lamb, global_step=None):
    """
    global_proj: (N, Vg, D) - Embeddings of global views
    all_proj: (N, V, D) - Embeddings of all views (global + local)
    lamb: scalar weight
    """
    # Centers from global views
    centers = global_proj.mean(dim=1, keepdim=True) # (N, 1, D)
    
    # Prediction loss (MSE between centers and all views)
    # (N, 1, D) - (N, V, D) -> (N, V, D) -> scalar mean
    sim_loss = (centers - all_proj).square().mean()
    
    sigreg_losses = []
    for i in range(all_proj.shape[1]):
        view_emb = all_proj[:, i, :] # (N, D)
        l = sigreg(view_emb) # scalar
        sigreg_losses.append(l)
    sigreg_loss = torch.stack(sigreg_losses).mean()
    
    return (1 - lamb) * sim_loss + lamb * sigreg_loss, sim_loss, sigreg_loss

