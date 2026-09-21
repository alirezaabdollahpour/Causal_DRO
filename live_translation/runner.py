"""Train, calibrate, and evaluate the synthetic or MuST-C pilot experiment."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import platform
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .attacks import (AttackConfig, PICNNAttacker, RecurrentAttacker,
                      anticipative_attack, transport_cost, train_picnn_step,
                      train_recurrent_step)
from .conditional import PrefixContinuationBank, streaming_duchi
from .config import ExperimentConfig
from .data import make_toy_batch, read_manifest
from .metrics import latency_scores, summarize, matched_comparisons, paired_bootstrap_difference


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


@contextmanager
def frozen(module):
    flags = [p.requires_grad for p in module.parameters()]
    for p in module.parameters():
        p.requires_grad_(False)
    try:
        yield
    finally:
        for p, flag in zip(module.parameters(), flags):
            p.requires_grad_(flag)


def fixed_scale(x, y, mask, rho, cost_scale):
    cost = float(transport_cost(x, y, mask, cost_scale).mean())
    if not math.isfinite(cost):
        raise FloatingPointError("Nonfinite calibration cost")
    if cost <= 1e-16:
        return 0.0, {"raw_dev_cost": cost, "scale": 0., "calibration_reached": False}
    scale = rho / math.sqrt(cost)
    return scale, {"raw_dev_cost": cost, "scale": scale, "calibration_reached": True}


class Experiment:
    def __init__(self, config: ExperimentConfig):
        self.c = config
        self.root = Path(config.output)
        if (self.root / "summary.json").exists():
            raise FileExistsError(f"Completed run already exists: {self.root}; choose a new output")
        self.root.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        self.generator = torch.Generator().manual_seed(config.seed)
        self.trace = []
        self.executed_source_hashes = {str(p): sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
        self.ac = AttackConfig(lam=config.lam, steps=config.attack_steps,
            tolerance=config.attack_tolerance, cost_scale=config.cost_scale)

    def load(self):
        from .backend import ToyBackend, FairseqBackend
        c = self.c
        if c.backend == "toy":
            self.backend = ToyBackend(state_dim=c.state_dim, seed=c.seed).to(c.device)
            self.train, self.dev, self.test = [make_toy_batch(n=n, state_dim=c.state_dim,
                seed=c.seed+i, min_chunks=c.min_chunks, max_chunks=c.max_chunks).to(c.device)
                for i, n in enumerate((c.train_size, c.dev_size, c.test_size))]
        else:
            self.backend = FairseqBackend(c.checkpoint, c.data_bin, c.user_dir,
                config_yaml=c.config_yaml, device=c.device)
            if (self.backend.pre_decision_ratio != c.pre_decision_ratio
                    or self.backend.chunk_ms != c.chunk_ms):
                raise ValueError("Checkpoint predecision interval differs from configuration")
            batches = []
            for path, n, split in ((c.train_manifest, c.train_size, "train"),
                                  (c.dev_manifest, c.dev_size, "dev"),
                                  (c.test_manifest, c.test_size, "tst-COMMON")):
                records = read_manifest(path)
                if any(r.get("corpus") != "MuST-C" or r.get("split") != split
                       or r.get("language") != "en-de" for r in records):
                    raise ValueError(f"Expected official MuST-C en-de {split} manifest; use live_translation.data")
                batches.append(self.backend.encode(records, limit=n))
            self.train, self.dev, self.test = batches
        self.backend.eval()
        ids = [set(b.ids) for b in (self.train, self.dev, self.test)]
        if ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]:
            raise ValueError("Train/calibration/test IDs overlap")
        if "adaptive_causal_duchi" in self.c.attacks or "adaptive_causal_duchi" in self.c.defenders:
            if max(int(self.dev.lengths.max()), int(self.test.lengths.max())) > int(self.train.lengths.max()):
                raise ValueError("Duchi's training continuation bank does not cover the evaluation horizon; "
                                 "increase train_size before fitting or reporting any conditions")
        # Save the encoded examples so the reported run can be inspected later.
        for name, batch in (("train", self.train), ("dev", self.dev), ("test", self.test)):
            torch.save(asdict(batch.to("cpu")), self.root / f"encoded_{name}.pt")
        self.trainable = self.backend.configure_adapters(c.adapter_rank)
        if not self.trainable:
            raise ValueError("Backend supplied no defender parameters")
        self.initial = copy.deepcopy({k: v.detach().cpu() for k, v in self.backend.state_dict().items()})

    def sample(self):
        indices = torch.randperm(len(self.train.ids), generator=self.generator)[:self.c.batch_size]
        return self.train.subset(indices)

    def picnn(self, threat, seed_offset=0):
        # Matched initial parameters for causal/full-context architecture.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.c.seed + seed_offset)
            attacker = PICNNAttacker(self.backend.state_dim, context_dim=self.c.picnn_width,
                hidden_dim=self.c.picnn_width, depth=self.c.picnn_depth, threat=threat,
                solver_steps=self.c.picnn_solver_steps, solver_tolerance=self.c.attack_tolerance)
        return attacker.to(self.c.device)

    def rnn(self, seed_offset=0):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.c.seed + seed_offset)
            attacker = RecurrentAttacker(self.backend.state_dim,
                hidden_dim=self.c.rnn_hidden_dim, head_dim=self.c.rnn_head_dim,
                time_scale=self.c.rnn_time_scale)
        return attacker.to(self.c.device)

    def fit_attacker(self, attacker, wait_k, steps, optimizer=None, label="eval_fit"):
        optimizer = optimizer or torch.optim.Adam(attacker.parameters(), lr=self.c.attack_lr)
        with frozen(self.backend):
            for step in range(steps):
                batch = self.sample()
                train_step = train_recurrent_step if isinstance(attacker, RecurrentAttacker) else train_picnn_step
                diagnostic = train_step(attacker, optimizer, batch.states, batch.mask,
                    lambda y: self.backend.loss(y, batch, wait_k), self.ac)
                self.trace.append(dict(stage=label, step=step, wait_k=wait_k,
                                       attack=getattr(attacker, "threat", "causal_rnn"), **diagnostic))
        return optimizer

    def raw_attack(self, name, batch, wait_k, attacker=None):
        if name == "clean":
            return batch.states, {"method": "identity"}
        with frozen(self.backend):
            if name.endswith("picnn") or name == "causal_rnn":
                return attacker(batch.states, batch.mask, self.c.lam, self.c.cost_scale)
            if name == "adaptive_causal_duchi":
                sampler = PrefixContinuationBank(self.train, self.backend, wait_k,
                                                  self.c.conditional_samples)
                return streaming_duchi(batch.states, batch.mask, sampler, self.ac,
                    solver="adaptive", max_evaluations=self.c.duchi_max_evaluations,
                    restarts=self.c.adaptive_causal_duchi_restarts, seed=self.c.seed)
            if name == "anticipative_oracle":
                result = anticipative_attack(batch.states, batch.mask,
                    lambda y: self.backend.loss(y, batch, wait_k), self.ac)
                result.diagnostics["reference_access"] = "full_heldout_reference_oracle"
                return result.y, result.diagnostics
        raise ValueError(name)

    def train_defender(self, name):
        self.backend.load_state_dict(self.initial)
        self.backend.eval()  # disable dropout, retain gradients
        self.generator.manual_seed(self.c.seed)
        optimizer = torch.optim.Adam(self.trainable, lr=self.c.defender_lr)
        attacker = None
        attack_optimizer = None
        if name.endswith("picnn"):
            attacker = self.picnn(name.split("_")[0])
            attack_optimizer = torch.optim.Adam(attacker.parameters(), lr=self.c.attack_lr)
        elif name == "causal_rnn":
            attacker = self.rnn()
            attack_optimizer = torch.optim.Adam(attacker.parameters(), lr=self.c.attack_lr)
        for step in range(self.c.warmup_steps + self.c.defender_steps):
            k = self.c.wait_k[step % len(self.c.wait_k)]
            # The same outer batch sequence is used by every defender, irrespective
            # of how many additional batches the attacker consumes during fitting.
            outer_gen = torch.Generator().manual_seed(self.c.seed + 10000 + step)
            ix = torch.randperm(len(self.train.ids), generator=outer_gen)[:self.c.batch_size]
            batch = self.train.subset(ix)
            y = batch.states
            scale = 0.
            if step >= self.c.warmup_steps and name != "nominal":
                if attacker is not None:
                    self.fit_attacker(attacker, k, self.c.attack_inner_steps,
                                      attack_optimizer, label=f"train_{name}")
                raw, _ = self.raw_attack(name, batch, k, attacker)
                dev_raw, _ = self.raw_attack(name, self.dev, k, attacker)
                scale, _ = fixed_scale(self.dev.states, dev_raw, self.dev.mask,
                                        self.c.rho, self.c.cost_scale)
                y = (batch.states + scale * (raw - batch.states)).detach()
            optimizer.zero_grad(set_to_none=True)
            loss = self.backend.loss(y.detach(), batch, k).mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite defender loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.trainable, 5.)
            optimizer.step()
            self.trace.append(dict(stage="defender", defender=name, step=step,
                wait_k=k, ce=float(loss.detach()), scale=scale,
                warmup=step < self.c.warmup_steps))
        trainable_names = {n for n, p in self.backend.named_parameters() if p.requires_grad}
        torch.save(dict(adapter_state={n: p.detach().cpu() for n, p in self.backend.named_parameters()
                                      if n in trainable_names}, optimizer=optimizer.state_dict(),
                        config=asdict(self.c), defender=name), self.root / f"defender_{name}.pt")

    def evaluate(self, name, wait_k):
        all_rows, all_records = [], {}
        for attack_name in self.c.attacks:
            started = time.monotonic()
            attacker = None
            if attack_name.endswith("picnn"):
                self.generator.manual_seed(self.c.seed + 20000 + wait_k)
                attacker = self.picnn(attack_name.split("_")[0], 100 + wait_k)
                self.fit_attacker(attacker, wait_k, self.c.attack_fit_steps)
            elif attack_name == "causal_rnn":
                self.generator.manual_seed(self.c.seed + 20000 + wait_k)
                attacker = self.rnn(100 + wait_k)
                self.fit_attacker(attacker, wait_k, self.c.attack_fit_steps)
            raw_dev, dev_diagnostic = self.raw_attack(attack_name, self.dev, wait_k, attacker)
            if attack_name == "clean":
                scale, calibration = 0., {"raw_dev_cost": 0., "scale": 0., "calibration_reached": True}
            else:
                scale, calibration = fixed_scale(self.dev.states, raw_dev, self.dev.mask,
                                                   self.c.rho, self.c.cost_scale)
            raw_test, test_diagnostic = self.raw_attack(attack_name, self.test, wait_k, attacker)
            y = (self.test.states + scale * (raw_test - self.test.states)).detach()
            with torch.no_grad():
                losses = self.backend.loss(y, self.test, wait_k)
                generations = self.backend.decode(y, self.test, wait_k,
                                                  max_new_tokens=self.c.max_new_tokens)
            costs = transport_cost(self.test.states, y, self.test.mask, self.c.cost_scale)
            records = []
            for i, generation in enumerate(generations):
                latency = latency_scores(generation.word_delays_ms, float(self.test.durations_ms[i]),
                                         len(self.test.references[i].split()))
                records.append(dict(id=self.test.ids[i], reference=self.test.references[i],
                    hypothesis=generation.text, finished=generation.finished,
                    token_ids=generation.token_ids, token_delays_ms=generation.token_delays_ms,
                    eos_delay_ms=generation.eos_delay_ms,
                    word_delays_ms=generation.word_delays_ms, duration_ms=float(self.test.durations_ms[i]),
                    ce=float(losses[i]), cost=float(costs[i]), **latency))
            row = dict(defender=name, attack=attack_name, wait_k=wait_k,
                       rho_target=self.c.rho, cost_target=self.c.rho**2,
                       **summarize(records), **calibration)
            row["radius_matched_on_test"] = (attack_name != "clean" and
                abs(row["cost"] - self.c.rho**2) <= self.c.radius_tolerance * self.c.rho**2)
            row["wall_seconds"] = time.monotonic() - started
            key = f"{name}_k{wait_k}_{attack_name}"
            write_json(self.root / "conditions" / f"{key}.json",
                dict(summary=row, records=records, dev_diagnostics=dev_diagnostic,
                     test_diagnostics=test_diagnostic))
            if attacker is not None:
                torch.save(dict(state_dict=attacker.state_dict(), scale=scale,
                    config=asdict(self.c), wait_k=wait_k,
                    threat=getattr(attacker, "threat", "causal_rnn")),
                    self.root / "conditions" / f"{key}.pt")
            print(json.dumps({k: row[k] for k in ("defender", "attack", "wait_k", "BLEU", "AL", "cost")}), flush=True)
            all_rows.append(row)
            all_records[attack_name] = records
        bootstrap = []
        if self.c.bootstrap_samples >= 2 and "anticipative_picnn" in all_records:
            for causal in ("causal_picnn", "causal_rnn", "adaptive_causal_duchi"):
                if causal in all_records:
                    bootstrap.append(dict(defender=name, wait_k=wait_k, causal_method=causal,
                        kind="paired_same_k_not_automatically_matched_radius_or_latency",
                        BLEU_c_minus_a=paired_bootstrap_difference(all_records[causal],
                            all_records["anticipative_picnn"], samples=self.c.bootstrap_samples,
                            seed=self.c.seed)))
        return all_rows, bootstrap

    def run(self):
        started = time.monotonic()
        write_json(self.root / "config.json", asdict(self.c))
        self.load()
        rows, bootstrap = [], []
        for name in self.c.defenders:
            print(f"Training {name}", flush=True)
            self.train_defender(name)
            for k in self.c.wait_k:
                new_rows, new_bootstrap = self.evaluate(name, k)
                rows.extend(new_rows)
                bootstrap.extend(new_bootstrap)
                write_json(self.root / "training_trace.json", self.trace)
        matched = matched_comparisons(rows, self.c.rho, self.c.radius_tolerance)
        with (self.root / "summary.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        source_root = Path(__file__).parent
        final_source_hashes = {str(p): sha256(p) for p in sorted(source_root.glob("*.py"))}
        inputs = dict(self.executed_source_hashes)
        for name in ("checkpoint", "train_manifest", "dev_manifest", "test_manifest"):
            value = getattr(self.c, name)
            if value:
                inputs[value] = sha256(Path(value))
        summary = dict(status="completed_synthetic_audit" if self.c.backend == "toy" else "completed_mustc_pilot",
            is_mustc_result=self.c.backend == "fairseq", config=asdict(self.c), rows=rows,
            matched_comparisons=matched, paired_same_k_bootstrap=bootstrap,
            wall_seconds=time.monotonic()-started, environment=dict(python=platform.python_version(),
                torch=torch.__version__, cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name() if self.c.device.startswith("cuda") else None),
            source_sha256=inputs, split_counts={n: len(b.ids) for n, b in
                (("train", self.train), ("dev", self.dev), ("test", self.test))},
            sources_unchanged_during_run=self.executed_source_hashes == final_source_hashes,
            backend_provenance=getattr(self.backend, "provenance", {"backend": "synthetic_random_frozen_feature_decoder"}),
            limitations=["Finite optimized attack candidates, no certified Wasserstein supremum.",
                "Global calibration scales fit on dev; test radii are measured, never reprojected.",
                "ACD uses a training-only prefix nearest-neighbor continuation approximation.",
                "Anticipative oracle additionally observes held-out reference labels.",
                "Latency excludes encoder, attack, and decoder computation time."])
        write_json(self.root / "summary.json", summary)
        return summary
