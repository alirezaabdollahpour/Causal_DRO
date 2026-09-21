"""Restore a calibrated Adaptive Causal Duchi policy for incremental deployment.

Only the condition's summary/configuration and the saved *training* continuation
bank are used. Saved test records are not attack inputs. The caller supplies the
frozen defender with the same adapter used by the selected condition.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .attacks import AttackConfig
from .conditional import PrefixContinuationBank, streaming_duchi
from .data import EncodedBatch


class CalibratedAdaptiveCausalDuchi:
    """A frozen conditional bank and defender with one saved global scale.

    Recomputing from a growing prefix reproduces previously committed actions:
    neighbor selection, finite node solves, and calibration are deterministic
    functions of the observed prefix and frozen training artifacts. No fitting
    or calibration occurs in ``__call__``. The finite solver's diagnostics are
    available in ``last_diagnostics`` after a call.
    """

    information = "causal"

    def __init__(self, bank: EncodedBatch, backend, wait_k: int, scale: float,
                 config: AttackConfig, *, count: int, restarts: int, seed: int,
                 max_evaluations: int, provenance: dict[str, Any]):
        self.bank = bank
        self.backend = backend.requires_grad_(False).eval()
        self.wait_k, self.scale, self.config = wait_k, scale, config
        self.restarts, self.seed = restarts, seed
        self.max_evaluations = max_evaluations
        self.sampler = PrefixContinuationBank(bank, self.backend, wait_k, count)
        self.provenance = provenance
        self.last_diagnostics: dict[str, Any] | None = None

    def __call__(self, x: Tensor, valid: Tensor) -> Tensor:
        if (x.ndim != 3 or min(x.shape) < 1 or valid.shape != x.shape[:2]
                or valid.dtype != torch.bool):
            raise ValueError("Expected nonempty source prefixes [B,T,D] and bool mask [B,T]")
        if x.shape[-1] != self.bank.states.shape[-1]:
            raise ValueError("Source dimension differs from the saved training continuation bank")
        if x.device != self.bank.states.device or valid.device != x.device:
            raise ValueError("Source prefix, mask, and saved bank must use the backend device")
        if x.dtype != self.bank.states.dtype or not bool(torch.isfinite(x).all()):
            raise ValueError("Source prefixes must be finite and use the saved bank dtype")
        if not bool(valid[:, 0].all()) or bool((valid[:, 1:] & ~valid[:, :-1]).any()):
            raise ValueError("Validity masks must be nonempty right-padded prefixes")
        # SimulEval callers may disable gradients globally. Node optimization
        # still needs action gradients; cloning outside inference mode avoids
        # trying to attach them to an inference tensor supplied by the caller.
        with torch.inference_mode(False), torch.enable_grad():
            source = x.detach().clone()
            mask = valid.detach().clone()
            raw, diagnostics = streaming_duchi(source, mask, self.sampler,
                self.config, solver="adaptive", max_evaluations=self.max_evaluations,
                restarts=self.restarts, seed=self.seed)
            y = source + self.scale * (raw - source)
        self.last_diagnostics = {**diagnostics, "fixed_calibration_scale": self.scale}
        return y.detach()


def load_adaptive_causal_duchi(condition_json_path: str | Path, backend,
                               wait_k: int) -> CalibratedAdaptiveCausalDuchi:
    """Load the runner's ACD condition without dev/test data access.

    Required sibling artifacts are ``../config.json`` and
    ``../encoded_train.pt`` relative to the condition directory. PyTorch loading
    uses ``weights_only=True``. Explicit nontraining split metadata and encoder
    checkpoint mismatches are rejected; absent split metadata is recorded as
    unverified, not invented. The caller must load the matching defender adapter
    before constructing this attack.
    """
    path = Path(condition_json_path).resolve()
    condition = json.loads(path.read_text())
    summary = condition.get("summary")
    if not isinstance(summary, dict) or summary.get("attack") != "adaptive_causal_duchi":
        raise ValueError("Live ACD deployment requires an adaptive_causal_duchi condition")
    if wait_k < 1 or summary.get("wait_k") != wait_k:
        raise ValueError("Live Duchi condition was calibrated at a different wait-k")
    scale = float(summary["scale"])
    if not math.isfinite(scale) or scale < 0:
        raise ValueError("Saved global calibration scale must be finite and nonnegative")
    run_root = path.parent.parent
    config = json.loads((run_root / "config.json").read_text())
    if wait_k not in config.get("wait_k", []):
        raise ValueError("Condition wait-k is absent from the saved run configuration")
    restarts = int(config["adaptive_causal_duchi_restarts"])
    seed = int(config["seed"])
    max_evaluations = int(config.get("duchi_max_evaluations", 10000))
    count = int(config["conditional_samples"])
    if min(restarts, count, max_evaluations) < 1 or seed < 0:
        raise ValueError("Invalid saved ACD configuration")
    attack_config = AttackConfig(lam=float(config["lam"]),
        steps=int(config["attack_steps"]), tolerance=float(config["attack_tolerance"]),
        cost_scale=float(config["cost_scale"]))
    parameter = next(backend.parameters(), None)
    if parameter is None:
        raise ValueError("A translation backend with parameters is required")
    bank_path = run_root / "encoded_train.pt"
    payload = torch.load(bank_path, map_location=parameter.device, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Saved training bank must be an EncodedBatch field dictionary")
    bank = EncodedBatch(**payload).to(parameter.device)
    if bank.states.shape[-1] != getattr(backend, "state_dim", None):
        raise ValueError("Training bank dimension does not match the defender encoder")
    if bank.states.dtype != parameter.dtype:
        raise ValueError("Training bank dtype does not match the loaded defender")
    if not bool(torch.isfinite(bank.states).all()):
        raise ValueError("Training continuation bank contains nonfinite states")
    metadata = bank.metadata or {}
    splits = metadata.get("splits", [metadata["split"]] if "split" in metadata else [])
    if not isinstance(splits, list) or any(not isinstance(split, str) for split in splits):
        raise ValueError("Training-bank split provenance must be a list of strings")
    if any(split.lower() not in {"train", "training", "unspecified"} for split in splits):
        raise ValueError("Continuation bank explicitly contains a nontraining split")
    if config.get("backend") == "fairseq" and metadata.get("not_a_translation_benchmark"):
        raise ValueError("A synthetic continuation bank cannot deploy a speech-benchmark condition")
    backend_provenance = getattr(backend, "provenance", {})
    recorded_hash = metadata.get("checkpoint_sha256")
    if recorded_hash is not None and backend_provenance.get("checkpoint_sha256") != recorded_hash:
        raise ValueError("Training bank encoder checkpoint differs from the loaded backend")
    for name in ("pre_decision_ratio", "chunk_ms"):
        if name in metadata and metadata[name] != getattr(backend, name, None):
            raise ValueError(f"Training bank {name} differs from the loaded backend")
    provenance = {"condition": str(path), "defender": summary.get("defender"),
        "training_bank": str(bank_path), "training_bank_corpus": metadata.get("corpus"),
        "split_verified": bool(splits) and all(split.lower() in {"train", "training"} for split in splits),
        "heldout_references_used": False,
        "conditional_model": "training_prefix_nearest_neighbor_splice"}
    # Do not retain the condition object: its per-utterance evaluation records
    # may contain held-out references, and none belongs in a deployed policy.
    return CalibratedAdaptiveCausalDuchi(bank, backend, wait_k, scale, attack_config,
        count=count, restarts=restarts, seed=seed,
        max_evaluations=max_evaluations, provenance=provenance)
