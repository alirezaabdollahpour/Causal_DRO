"""Estimate possible speech continuations from training examples.

For each observed audio prefix, this module finds similar training prefixes
and uses their later audio and Arabic references as possible futures. This is
an approximation based on neighbors, not a speech generator. The lookup does
not receive test references, future audio, durations, or sample IDs.
"""
from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
import hashlib
import math
import os
from pathlib import Path
import uuid
import numpy as np
import torch

from .data import EncodedBatch
from .attacks import ScenarioTree, AttackConfig, causal_duchi_step, prefix_node_ids
from .benchmark_data import EncodedDataset, collate_batches


class _ProjectedPrefixIndex:
    """Full training-law index with approximate shortlist and exact rerank.

    A query at time t compares only anchors in its observed prefix. The
    shortlist is approximate, so global nearest neighbors are not certified.
    Every official training utterance remains eligible by its true length.
    """

    ANCHORS = (0, 1, 2, 3, 7, 15, 31, 63, 127, 255, 511)
    WIDTHS = (16, 16, 16, 16, 8, 8, 8, 8, 8, 8, 8)
    SEED = 20260921
    VERSION = 2

    def __init__(self, bank: EncodedDataset):
        self.bank = bank
        self.path = Path(bank.cache_dir) / "conditional_prefix_index.pt"
        self.states_path = Path(bank.cache_dir) / "conditional_prefix_states.f32"
        self.checksum = self._item_checksum(bank)
        self.projection = None
        self.features = None
        self.lengths = None
        self.offsets = None
        self.packed = None
        self._load_or_build()

    @staticmethod
    def _file_checksum(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _item_checksum(bank: EncodedDataset) -> str:
        digest = hashlib.sha256()
        for entry in bank.index["items"]:
            digest.update(entry["id"].encode("utf-8"))
            digest.update(entry["sha256"].encode("ascii"))
        return digest.hexdigest()

    def _load(self) -> bool:
        if not self.path.is_file():
            return False
        try:
            payload = torch.load(self.path, map_location="cpu", weights_only=True)
            if (payload.get("version") != self.VERSION or payload.get("fingerprint") != self.bank.fingerprint
                    or payload.get("item_checksum") != self.checksum
                    or payload.get("anchors") != list(self.ANCHORS)
                    or payload.get("widths") != list(self.WIDTHS)):
                return False
            projection, features, lengths, offsets = (
                payload[name] for name in ("projection", "features", "lengths", "offsets"))
            total_blocks = payload["total_blocks"]
            if (projection.shape != (self.bank.state_dim, max(self.WIDTHS))
                    or features.shape != (len(self.bank), sum(self.WIDTHS))
                    or features.dtype != torch.float16 or lengths.shape != (len(self.bank),)
                    or lengths.dtype != torch.int32 or offsets.shape != (len(self.bank),)
                    or offsets.dtype != torch.int64 or not bool((lengths > 0).all())
                    or not bool(torch.isfinite(features).all())
                    or not bool(torch.isfinite(projection).all())
                    or int(offsets[0]) != 0
                    or not bool((offsets[1:] == offsets[:-1] + lengths[:-1]).all())
                    or int(offsets[-1] + lengths[-1]) != total_blocks
                    or self.states_path.stat().st_size != total_blocks * self.bank.state_dim * 4
                    or self._file_checksum(self.states_path) != payload["states_sha256"]):
                return False
            packed = np.memmap(self.states_path, mode="r", dtype="<f4",
                               shape=(total_blocks, self.bank.state_dim))
            self.projection, self.features, self.lengths = projection, features, lengths
            self.offsets, self.packed = offsets, packed
            return True
        except (OSError, RuntimeError, ValueError, KeyError, TypeError, IndexError):
            return False

    def _load_or_build(self) -> None:
        if self._load():
            return
        import fcntl

        with (self.path.parent / ".conditional_prefix_index.lock").open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if self._load():
                return
            generator = torch.Generator(device="cpu").manual_seed(self.SEED)
            projection = torch.randn(self.bank.state_dim, max(self.WIDTHS), generator=generator)
            projection /= math.sqrt(self.bank.state_dim)
            features = torch.zeros(len(self.bank), sum(self.WIDTHS), dtype=torch.float16)
            lengths = torch.empty(len(self.bank), dtype=torch.int32)
            offsets = torch.empty(len(self.bank), dtype=torch.int64)
            packed_hash = hashlib.sha256()
            total_blocks = 0
            states_temporary = self.states_path.with_name(
                f".{self.states_path.name}.{uuid.uuid4().hex}.tmp")
            try:
                with states_temporary.open("wb") as packed_output:
                    for i in range(len(self.bank)):
                        item = self.bank[i]
                        length = int(item.lengths[0])
                        lengths[i] = length
                        offsets[i] = total_blocks
                        states = item.states[0, :length].contiguous()
                        raw = states.numpy().astype("<f4", copy=False).tobytes()
                        packed_output.write(raw)
                        packed_hash.update(raw)
                        total_blocks += length
                        start = 0
                        for anchor, width in zip(self.ANCHORS, self.WIDTHS):
                            if anchor < length:
                                features[i, start:start + width] = (
                                    states[anchor] @ projection[:, :width]).half()
                            start += width
                os.replace(states_temporary, self.states_path)
            finally:
                states_temporary.unlink(missing_ok=True)
            payload = dict(version=self.VERSION, fingerprint=self.bank.fingerprint,
                           item_checksum=self.checksum, anchors=list(self.ANCHORS),
                           widths=list(self.WIDTHS), projection=projection,
                           features=features, lengths=lengths, offsets=offsets,
                           total_blocks=total_blocks, states_sha256=packed_hash.hexdigest())
            temporary = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
            try:
                torch.save(payload, temporary)
                os.replace(temporary, self.path)
            finally:
                temporary.unlink(missing_ok=True)
            packed = np.memmap(self.states_path, mode="r", dtype="<f4",
                               shape=(total_blocks, self.bank.state_dim))
            self.projection, self.features, self.lengths = projection, features, lengths
            self.offsets, self.packed = offsets, packed

    def count_at_least(self, t: int) -> int:
        return int((self.lengths >= t).sum())

    @lru_cache(maxsize=256)
    def item(self, position: int) -> EncodedBatch:
        """Reuse immutable CPU donor items across adjacent source prefixes."""
        return self.bank[position]

    def exact_prefix_distance(self, position: int, observed: np.ndarray) -> float:
        """Float64 reduction of original float32 state squared distances."""
        t, dimension = observed.shape
        start = int(self.offsets[position])
        difference = self.packed[start:start + t] - observed
        return float(np.einsum("ij,ij->", difference, difference,
                               dtype=np.float64) / (t * dimension))

    def shortlist(self, prefix: torch.Tensor, size: int) -> list[int]:
        prefix = prefix.detach().to(device="cpu", dtype=torch.float32)
        t = len(prefix)
        eligible = self.lengths >= t
        available = int(eligible.sum())
        if available == 0:
            raise ValueError("Complete training law has no source as long as the observed prefix")
        score = torch.zeros(len(self.bank), dtype=torch.float32)
        start = 0
        for anchor, width in zip(self.ANCHORS, self.WIDTHS):
            if anchor < t:
                query = prefix[anchor] @ self.projection[:, :width]
                difference = self.features[:, start:start + width].float() - query
                score += difference.square().mean(dim=1)
            start += width
        score[~eligible] = torch.inf
        limit = min(size, available)
        if limit == available:
            return torch.nonzero(eligible).flatten().tolist()
        # Partition in O(N) work, then settle the cutoff tie by official row.
        # Full sorting 212,085 donors at every online decision is prohibitive.
        cutoff = torch.topk(score, limit, largest=False, sorted=False).values.max()
        strictly_better = torch.nonzero(score < cutoff).flatten()
        tied = torch.nonzero(score == cutoff).flatten()[:limit - len(strictly_better)]
        selected = torch.cat((strictly_better, tied))
        order = torch.argsort(score[selected], stable=True)
        return selected[order].tolist()


class PrefixContinuationBank:
    def __init__(self, bank: EncodedBatch | EncodedDataset, backend, wait_k: int,
                 count: int = 4, shortlist: int = 128):
        if count < 1 or wait_k < 1:
            raise ValueError("Conditional sample count and wait-k must be positive")
        if shortlist < count:
            raise ValueError("Projected donor shortlist must contain at least count candidates")
        self.bank, self.backend, self.wait_k, self.count = bank, backend, wait_k, count
        self.shortlist_size = shortlist
        self.index = None
        self.last_donor_diagnostics = {}
        if isinstance(bank, EncodedDataset):
            if not hasattr(bank, "_projected_prefix_index"):
                bank._projected_prefix_index = _ProjectedPrefixIndex(bank)
            self.index = bank._projected_prefix_index

    def count_at_least(self, t: int) -> int:
        if self.index is not None:
            return self.index.count_at_least(t)
        return int((self.bank.lengths >= t).sum())

    def __call__(self, prefix: torch.Tensor):
        dimension = self.bank.state_dim if self.index is not None else self.bank.states.shape[2]
        if prefix.ndim != 2 or len(prefix) < 1 or prefix.shape[1] != dimension:
            raise ValueError("Expected a nonempty observed source prefix [t,D]")
        t = len(prefix)
        if self.index is None:
            # The small in-memory path remains exact for legacy experiments.
            eligible = torch.nonzero(self.bank.lengths >= t).flatten()
            if not len(eligible):
                raise ValueError("Training continuation bank has no source this long. "
                                 "Increase bank coverage; do not substitute the test suffix.")
            source = self.bank.states[eligible, :t]
            distance = (source - prefix[None]).square().mean((1, 2))
            nearest = eligible[torch.argsort(distance, stable=True)[:self.count]]
            sampled = self.bank.subset(nearest)
            self.last_donor_diagnostics = dict(search="full_in_memory_exact",
                eligible=int(len(eligible)), shortlisted=int(len(eligible)), effective_donors=len(sampled.ids))
        else:
            # Eligibility is known from training metadata alone. The held-out
            # utterance's total length, suffix, and reference are never used.
            candidates = self.index.shortlist(prefix, self.shortlist_size)
            observed = prefix.detach().to(device="cpu", dtype=torch.float32).numpy()
            ranked = []
            for i in candidates:
                distance = self.index.exact_prefix_distance(i, observed)
                ranked.append((distance, i))
            ranked.sort(key=lambda item: (item[0], item[1]))
            chosen = [self.index.item(i) for _, i in ranked[:self.count]]
            sampled = collate_batches(chosen, self.bank.pad_id).to(prefix.device)
            self.last_donor_diagnostics = dict(
                search="projected_prefix_shortlist_then_exact_rerank",
                global_nearest_certified=False, bank_population=len(self.bank),
                eligible=self.index.count_at_least(t), shortlisted=len(candidates),
                effective_donors=len(chosen),
                donor_ids=[item.ids[0] for item in chosen])
        x = sampled.states.clone()
        x[:, :t] = prefix[None]
        sampled = replace(sampled, states=x)
        tree = ScenarioTree(x, sampled.mask, prefix_node_ids(x, sampled.mask))
        return tree, lambda y: self.backend.loss(y, sampled, self.wait_k)


def streaming_duchi(x: torch.Tensor, valid: torch.Tensor, sampler,
                    config: AttackConfig, *, solver: str = "adaptive",
                    max_evaluations: int = 10000,
                    restarts: int = 3, seed: int = 0) -> tuple[torch.Tensor, dict]:
    """Commit each ACD action before exposing the next source block.

    ``restarts`` and ``seed`` control prefix-measurable adaptive starts. The
    training-only sampler supplies the fixed conditional tree at each node.
    Legacy solvers remain selectable to replay archived experiments.
    """
    output, diagnostics = x.detach().clone(), []
    for i in range(len(x)):
        actions = []
        for t in range(int(valid[i].sum())):
            previous = torch.stack(actions) if actions else x.new_empty((0, x.shape[-1]))
            action, diagnostic = causal_duchi_step(x[i, :t+1], previous, sampler,
                config, solver=solver, max_evaluations=max_evaluations,
                restarts=restarts, seed=seed)
            if getattr(sampler, "last_donor_diagnostics", None):
                diagnostic = {**diagnostic, "donor_search": dict(sampler.last_donor_diagnostics)}
            actions.append(action.detach())
            diagnostics.append(diagnostic)
        output[i, :len(actions)] = torch.stack(actions)
    return output, {"method": ("adaptive_causal_duchi" if solver == "adaptive"
                                else "legacy_causal_duchi"),
                    "conditional_model": "training_prefix_nearest_neighbor_splice",
                    "reference_access": "training_bank_only",
                    "donor_search": ("full_train_projected_shortlist_exact_rerank"
                                     if getattr(sampler, "index", None) is not None
                                     else "in_memory_exact"),
                    "solver": solver,
                    "restarts": restarts if solver == "adaptive" else None,
                    "seed": seed if solver == "adaptive" else None,
                    "nodes": diagnostics}
