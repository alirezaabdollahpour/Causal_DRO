"""Load a trained causal recurrent policy for live source-prefix attacks.

The pilot runner saves the attacker and its development-set calibration in a
single ``conditions/*.pt`` artifact. This loader uses only those frozen
parameters and metadata; it never loads training or evaluation examples.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from pathlib import Path

import torch
from torch import Tensor, nn

from .attacks import RecurrentAttacker


class CalibratedCausalRNN(nn.Module):
    """A frozen prefix map with one scalar fitted before live evaluation."""

    information = "causal"

    def __init__(self, attacker: RecurrentAttacker, scale: float,
                 lam: float, cost_scale: float) -> None:
        super().__init__()
        self.attacker = attacker.requires_grad_(False).eval()
        self.scale = scale
        self.lam = lam
        self.cost_scale = cost_scale

    def forward(self, x: Tensor, valid: Tensor) -> Tensor:
        raw, _ = self.attacker(x, valid, self.lam, self.cost_scale)
        return x + self.scale * (raw - x)


def load_causal_rnn(path: str | Path, state_dim: int, device: str,
                    wait_k: int) -> CalibratedCausalRNN:
    """Restore a pilot runner artifact for the matching source and wait-k.

    ``weights_only=True`` limits deserialization to tensors and simple data.
    The checkpoint contains no held-out utterances or reference text.
    """
    if (isinstance(state_dim, bool) or not isinstance(state_dim, int) or state_dim < 1
            or isinstance(wait_k, bool) or not isinstance(wait_k, int) or wait_k < 1):
        raise ValueError("Source dimension and wait-k must be positive")
    payload = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("Causal RNN artifact must be a mapping")
    if payload.get("threat") != "causal_rnn" or payload.get("wait_k") != wait_k:
        raise ValueError("Live RNN artifact must be causal and fitted at this wait-k")
    config = payload.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("Causal RNN artifact has no run configuration")
    configured_wait_k = config.get("wait_k")
    if not isinstance(configured_wait_k, (list, tuple)) or wait_k not in configured_wait_k:
        raise ValueError("Artifact wait-k is absent from its run configuration")

    def positive_float(key: str) -> float:
        try:
            value = float(config[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid RNN configuration value: {key}") from exc
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid RNN configuration value: {key}")
        return value

    def positive_int(key: str) -> int:
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"Invalid RNN configuration value: {key}")
        return value

    hidden_dim = positive_int("rnn_hidden_dim")
    head_dim = positive_int("rnn_head_dim")
    time_scale = positive_float("rnn_time_scale")
    lam, cost_scale = positive_float("lam"), positive_float("cost_scale")
    try:
        scale = float(payload["scale"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Calibrated RNN scale is missing or invalid") from exc
    if not math.isfinite(scale) or scale < 0:
        raise ValueError("Calibrated RNN scale must be finite and nonnegative")
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("Causal RNN artifact has no attacker state_dict")
    attacker = RecurrentAttacker(state_dim, hidden_dim=hidden_dim,
        head_dim=head_dim, time_scale=time_scale).to(device)
    attacker.load_state_dict(state_dict, strict=True)
    return CalibratedCausalRNN(attacker, scale, lam, cost_scale)
