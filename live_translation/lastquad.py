"""Conditional last-quadratic PICNN energy for the translation NodeSolve.

The affine hidden skips / final activated diagonal quadratic / separate base
quadratic follow ICNN-DRO's NPF last_layer_diagonal construction. Conditional
gates follow the PICNN construction of Amos et al. (2017). This is a PyTorch
adaptation, not a copy of OTT's unconditional ICNN or its transport objective.
"""
from __future__ import annotations

import math
import torch
from torch import Tensor, nn
from torch.nn import functional as F


def lecun_(layer: nn.Linear, scale: float = 1.0) -> None:
    nn.init.normal_(layer.weight, std=scale / math.sqrt(layer.in_features))
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class PositiveDense(nn.Module):
    """Positive randomized weights with a fan-in-independent mean row sum.

    Lognormal coefficient of variation is one. Softplus parameterization
    retains gradients without projection or permanently dead ReLU weights.
    This is a mean-controlled initialization for softplus/gated layers, not
    a claim of the exact ReLU signal-propagation theory of Hoedt & Klambauer.
    """
    def __init__(self, width: int, output: int):
        super().__init__()
        log_var = math.log(2.0)
        effective = torch.empty(output, width).normal_(
            -math.log(width) - .5 * log_var, math.sqrt(log_var)).exp_()
        self.raw_weight = nn.Parameter(effective + torch.log(-torch.expm1(-effective)))

    def forward(self, value: Tensor) -> Tensor:
        return F.linear(value, F.softplus(self.raw_weight))


class LastQuadPICNN(nn.Module):
    """K=(lambda/s)[.5 <q,y²> + epsilon softplus(v) + <b(h),y>].

    v contains the positive convex readout and the only nonlinear input
    quadratic skip. q has a strict positive floor and is free to fall below
    two. Input-dependent gates receive only frozen context, never the current
    optimization variable. All dimensions are independent of utterance length.
    """
    def __init__(self, action_dim: int, context_dim: int, hidden_dim: int = 64,
                 depth: int = 3, curvature_floor: float = .1,
                 residual_scale: float = .01):
        super().__init__()
        if min(action_dim, context_dim, hidden_dim, depth) < 1:
            raise ValueError('PICNN dimensions and depth must be positive')
        if not 0 < curvature_floor < 2 or residual_scale <= 0:
            raise ValueError('Invalid PICNN curvature or residual scale')
        self.curvature_floor = curvature_floor
        self.residual_scale = residual_scale
        self.raw_curvature = nn.Parameter(torch.zeros(action_dim))
        self.action_layers = nn.ModuleList(nn.Linear(action_dim, hidden_dim, bias=False) for _ in range(depth))
        self.context_layers = nn.ModuleList(nn.Linear(context_dim, hidden_dim) for _ in range(depth))
        self.context_updates = nn.ModuleList(nn.Linear(context_dim, context_dim) for _ in range(depth-1))
        self.action_gates = nn.ModuleList(nn.Linear(context_dim, action_dim) for _ in range(depth))
        self.positive_layers = nn.ModuleList(PositiveDense(hidden_dim, hidden_dim) for _ in range(depth-1))
        self.positive_gates = nn.ModuleList(nn.Linear(context_dim, hidden_dim) for _ in range(depth-1))
        self.output_positive = PositiveDense(hidden_dim, 1)
        self.last_diagonal = nn.Linear(context_dim, action_dim)
        self.last_affine = nn.Linear(context_dim, action_dim)
        self.last_bias = nn.Linear(context_dim, 1)
        self.affine = nn.Linear(context_dim, action_dim)
        for layer in self.action_layers:
            lecun_(layer, .1)
        for layer in [*self.context_layers, *self.context_updates]:
            lecun_(layer)
        for layer in [*self.action_gates, *self.positive_gates, self.last_diagonal,
                      self.last_affine, self.last_bias, self.affine]:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        # Each quadratic coordinate initially contributes 1/d, avoiding a
        # dimension-sized scalar preactivation in the final softplus.
        self.diagonal_scale = 1.0 / action_dim

    def curvature(self) -> Tensor:
        return self.curvature_floor + (2-self.curvature_floor) * F.softplus(self.raw_curvature) / math.log(2.)

    def residual(self, action: Tensor, context: Tensor) -> Tensor:
        u, z = context, None
        for j, (a, h, g) in enumerate(zip(self.action_layers, self.context_layers, self.action_gates)):
            pre = a(action * (1 + .1 * torch.tanh(g(u)))) + h(u)
            if z is not None:
                gate = F.softplus(self.positive_gates[j-1](u)) / math.log(2.)
                # Centering is additive and therefore preserves convexity.
                pre = pre + self.positive_layers[j-1](z * gate) - math.log(2.)
            z = F.softplus(pre)
            if j < len(self.context_updates):
                u = torch.tanh(self.context_updates[j](u))
        diagonal = self.diagonal_scale * F.softplus(self.last_diagonal(u)) / math.log(2.)
        last = self.output_positive(z).squeeze(-1) - math.log(2.)
        last = last + .5*(diagonal*action.square()).sum(-1)
        last = last + (self.last_affine(u)*action).sum(-1) + self.last_bias(u).squeeze(-1)
        return self.residual_scale * F.softplus(last) + (self.affine(context)*action).sum(-1)

    def forward(self, action: Tensor, context: Tensor, lam: float = 1., cost_scale: float = 1.) -> Tensor:
        return lam/cost_scale*(.5*(self.curvature()*action.square()).sum(-1) + self.residual(action, context))

    def conjugate_objective(self, action: Tensor, context: Tensor, source: Tensor,
                            lam: float, cost_scale: float) -> Tensor:
        """K(y)-<2lambda/s x,y> with its large action-independent term removed.

        Computing in displacement coordinates avoids subtracting two O(||x||²)
        values when lambda=3 produces small moves in 4096 dimensions.
        """
        delta = action-source
        q = self.curvature()
        return lam/cost_scale*(.5*(q*delta.square()).sum(-1) +
            ((q-2)*source*delta).sum(-1) + self.residual(action, context))


def initialize_recurrent(cell: nn.GRUCell) -> None:
    for weight in cell.weight_ih.chunk(3, dim=0):
        nn.init.normal_(weight, std=1/math.sqrt(cell.input_size))
    for weight in cell.weight_hh.chunk(3, dim=0):
        nn.init.orthogonal_(weight)
    nn.init.zeros_(cell.bias_ih)
    nn.init.zeros_(cell.bias_hh)
    # PyTorch gate order is reset, update, new; z=1 retains more history.
    with torch.no_grad():
        cell.bias_ih[cell.hidden_size:2*cell.hidden_size].fill_(1.)
