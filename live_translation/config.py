from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import yaml


@dataclass
class ExperimentConfig:
    backend: str = "toy"
    output: str = "live_translation/runs/smoke"
    device: str = "cpu"
    seed: int = 2027
    wait_k: list[int] = field(default_factory=lambda: [1, 3, 5])
    rho: float = 0.2  # Target square root of mean transport cost.
    cost_scale: float = 1.0  # Shared cost scale for all examples and attacks.
    lam: float = 1.0
    radius_tolerance: float = 0.05
    batch_size: int = 8
    warmup_steps: int = 60
    defender_steps: int = 8
    defender_lr: float = 0.01
    adapter_rank: int = 4
    defenders: list[str] = field(default_factory=lambda: ["nominal", "causal_picnn", "causal_rnn", "anticipative_picnn"])
    attacks: list[str] = field(default_factory=lambda: ["clean", "causal_picnn", "causal_rnn", "anticipative_picnn", "adaptive_causal_duchi", "anticipative_oracle"])
    attack_fit_steps: int = 12
    attack_inner_steps: int = 1
    attack_lr: float = 0.01
    attack_steps: int = 12
    attack_tolerance: float = 1e-5
    picnn_solver_steps: int = 12
    picnn_width: int = 16
    picnn_depth: int = 2
    rnn_hidden_dim: int = 64
    rnn_head_dim: int = 64
    rnn_time_scale: float = 128.0
    conditional_samples: int = 4
    adaptive_causal_duchi_restarts: int = 3
    duchi_max_evaluations: int = 10000
    max_new_tokens: int = 24
    bootstrap_samples: int = 100
    train_size: int = 24
    dev_size: int = 8
    test_size: int = 8
    min_chunks: int = 4
    max_chunks: int = 6
    state_dim: int = 8
    checkpoint: str | None = None
    data_bin: str | None = None
    config_yaml: str = "config_st.yaml"
    user_dir: str | None = None
    train_manifest: str | None = None
    dev_manifest: str | None = None
    test_manifest: str | None = None
    pre_decision_ratio: int = 7
    chunk_ms: int = 280

    def __post_init__(self):
        if self.backend not in {"toy", "fairseq"}:
            raise ValueError("backend must be toy or fairseq")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")
        if not self.wait_k or any(k < 1 for k in self.wait_k) or len(set(self.wait_k)) != len(self.wait_k):
            raise ValueError("wait_k must contain distinct positive chunk counts")
        for name in ("rho", "cost_scale", "lam", "defender_lr", "attack_lr",
                     "attack_tolerance", "radius_tolerance"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not math.isfinite(self.rnn_time_scale) or self.rnn_time_scale <= 0:
            raise ValueError("rnn_time_scale must be positive and finite")
        for name in ("batch_size", "adapter_rank", "attack_fit_steps", "attack_steps",
                     "attack_inner_steps", "picnn_solver_steps", "conditional_samples",
                     "max_new_tokens", "train_size", "dev_size", "test_size",
                     "rnn_hidden_dim", "rnn_head_dim", "picnn_width", "picnn_depth",
                     "adaptive_causal_duchi_restarts", "duchi_max_evaluations",
                     "min_chunks", "max_chunks"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.min_chunks > self.max_chunks:
            raise ValueError("min_chunks cannot exceed max_chunks")
        if min(self.warmup_steps, self.defender_steps) < 0:
            raise ValueError("Training update counts cannot be negative")
        if not self.defenders or not self.attacks:
            raise ValueError("At least one defender and attack are required")
        if not set(self.defenders) <= {"nominal", "causal_picnn", "causal_rnn", "anticipative_picnn", "adaptive_causal_duchi"}:
            raise ValueError("Unknown defender")
        if not set(self.attacks) <= {"clean", "causal_picnn", "causal_rnn", "anticipative_picnn", "adaptive_causal_duchi", "anticipative_oracle"}:
            raise ValueError("Unknown attack")
        if self.backend == "fairseq":
            for name in ("checkpoint", "data_bin", "user_dir", "train_manifest", "dev_manifest", "test_manifest"):
                value = getattr(self, name)
                if not value or not Path(value).exists():
                    raise FileNotFoundError(f"Supply an existing {name}; got {value!r}")


def load_config(path: str | Path) -> ExperimentConfig:
    content = yaml.safe_load(Path(path).read_text())
    if not isinstance(content, dict):
        raise ValueError("Configuration must be a YAML mapping")
    return ExperimentConfig(**content)
