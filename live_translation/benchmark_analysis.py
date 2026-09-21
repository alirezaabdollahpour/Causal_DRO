"""Paired talk-cluster analysis for the real speech-translation experiment.

The unit resampled is an entire TED talk, never an independently selected
utterance. Corpus BLEU/chrF are recomputed from summed sufficient statistics,
not averaged sentence scores. In multi-seed analysis, scores are first computed
for each model seed and then averaged: seed replicas are not extra test speech.

Required per-utterance fields are id, talk_id, reference, hypothesis, ce, cost,
and finished. Latency can be supplied as AL/LAAL/AP/DAL or reconstructed from
word_delays_ms and duration_ms. Text is scored exactly as supplied: there is no
undocumented Arabic normalization, stemming, punctuation removal, or filtering.
Secondary talk-concatenated scores require a complete set of segments: a
within-talk zero-based segment_index and expected_talk_segments on each record.
Explicit source offsets can provide ordering when ordinals are entirely absent.
Incomplete talks are identified and excluded only from the secondary metric,
which is always labeled as scoring the selected complete-talk subset.

Primary scoring configuration: SacreBLEU 13a and chrF2 (nc=6,nw=0,beta=2).
The upstream toolkit has no automatic Arabic-specific BLEU tokenizer:
https://github.com/mjpost/sacrebleu#languages--preprocessing
Caching corpus sufficient statistics follows its own significance module:
https://github.com/mjpost/sacrebleu/blob/master/sacrebleu/significance.py
The private statistic interface is checked against public corpus_score on each
prepared corpus; both complete metric signatures are retained in every result.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .metrics import latency_scores


DEFENDERS = ("nominal", "causal_picnn", "causal_rnn", "anticipative_picnn",
             "adaptive_causal_duchi", "anticipative_duchi")
ATTACKS = ("clean", "causal_picnn", "causal_rnn", "anticipative_picnn",
           "adaptive_causal_duchi", "anticipative_duchi")
ATTACK_INFORMATION = {
    "clean": "identity",
    "causal_picnn": "source_prefix_only_frozen_learned_policy",
    "causal_rnn": "source_prefix_only_frozen_recurrent_policy",
    "anticipative_picnn": "full_source_only_frozen_learned_policy",
    "adaptive_causal_duchi": "source_prefix_and_training_conditional_reference_model",
    "anticipative_duchi": "full_source_and_heldout_reference_direct_ascent",
}
NUMERIC_METRICS = ("ce", "cost", "AL", "LAAL", "AP", "DAL")
LATENCY_METRICS = ("AL", "LAAL", "AP", "DAL")


@dataclass(frozen=True)
class MetricConfig:
    bleu_tokenizer: str = "13a"
    bleu_lowercase: bool = False


@dataclass(frozen=True)
class MatchingConfig:
    """Point-estimate tolerances declared before inspecting test outcomes.

    rho is a radius, so the mean squared-transport target is rho**2.
    Latency tolerance is absolute in milliseconds. These tests do not establish
    population equivalence; uncertainty in cost and latency is retained in the
    paired bootstrap. They never select a subset of the scored utterances.
    """
    rho: float
    cost_relative_tolerance: float = 0.05
    cost_absolute_tolerance: float = 1e-8
    latency_metric: str = "AL"
    latency_tolerance_ms: float = 50.0

    def __post_init__(self):
        for name in ("rho", "cost_relative_tolerance",
                     "cost_absolute_tolerance", "latency_tolerance_ms"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.latency_metric not in {"AL", "LAAL", "DAL"}:
            raise ValueError("Latency matching requires AL, LAAL, or DAL in ms")


def _finite(value: Any, field: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a real number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite real number") from exc
    if not math.isfinite(result) or (nonnegative and result < 0):
        raise ValueError(f"{field} must be finite" +
                         (" and nonnegative" if nonnegative else ""))
    return result


def validate_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Validate all examples and align by ID without dropping any row.

    talk_id must be explicit. Inferring it from an arbitrary segment identifier
    could quietly turn a cluster bootstrap back into an utterance bootstrap.
    """
    if not records:
        raise ValueError("A nonempty corpus of records is required")
    checked, seen = [], set()
    for raw in records:
        record = dict(raw)
        for field in ("id", "talk_id"):
            value = record.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Every record needs a nonempty explicit {field}")
        if record["id"] in seen:
            raise ValueError(f"Duplicate utterance id: {record['id']}")
        seen.add(record["id"])
        if not isinstance(record.get("reference"), str) or not record["reference"].strip():
            raise ValueError(f"Nonempty reference required for {record['id']}")
        if not isinstance(record.get("hypothesis"), str):
            raise ValueError(f"String hypothesis required for {record['id']}")
        if not isinstance(record.get("finished"), bool):
            raise ValueError(f"Explicit boolean finished required for {record['id']}")
        for field in ("ce", "cost"):
            record[field] = _finite(record.get(field), field, nonnegative=True)
        for field, minimum in (("segment_index", 0), ("expected_talk_segments", 1)):
            if record.get(field) is not None:
                value = record[field]
                if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                    raise ValueError(f"{field} must be an integer >= {minimum}")
        if record.get("offset") is not None:
            record["offset"] = _finite(record["offset"], "offset", nonnegative=True)
        if "duration_ms" in record:
            record["duration_ms"] = _finite(record["duration_ms"], "duration_ms")
            if record["duration_ms"] <= 0:
                raise ValueError("duration_ms must be positive")
        if "word_delays_ms" in record:
            if "duration_ms" not in record:
                raise ValueError("word_delays_ms require source duration_ms")
            delays = record["word_delays_ms"]
            if len(delays) != len(record["hypothesis"].split()):
                raise ValueError("One delay is required per emitted whitespace word")
            computed = latency_scores(delays, record["duration_ms"],
                                      len(record["reference"].split()))
            for name in LATENCY_METRICS:
                previous = record.get(name)
                if previous is not None:
                    previous = _finite(previous, name)
                    if computed[name] is None or not math.isclose(
                            previous, computed[name], abs_tol=1e-4, rel_tol=1e-7):
                        raise ValueError(f"Saved {name} disagrees with emission trace")
                record[name] = computed[name]
        else:
            for name in LATENCY_METRICS:
                if record.get(name) is not None:
                    record[name] = _finite(record[name], name)
        if not record["hypothesis"].strip():
            if any(record.get(name) is not None for name in LATENCY_METRICS):
                raise ValueError("An empty hypothesis has no defined word latency")
            for name in LATENCY_METRICS:
                record[name] = None
        checked.append(record)
    return sorted(checked, key=lambda row: row["id"])


def _check_pair(left: Sequence[dict], right: Sequence[dict]) -> None:
    if [r["id"] for r in left] != [r["id"] for r in right]:
        raise ValueError("Paired systems must have exactly the same utterance IDs")
    for a, b in zip(left, right):
        for field in ("reference", "talk_id"):
            if a[field] != b[field]:
                raise ValueError(f"Paired {field} differs for utterance {a['id']}")
        for field in ("segment_index", "expected_talk_segments", "offset"):
            if a.get(field) != b.get(field):
                raise ValueError(f"Paired {field} differs for utterance {a['id']}")
        if ("duration_ms" in a) != ("duration_ms" in b):
            raise ValueError("Source duration metadata must be present on both sides")
        if "duration_ms" in a and not math.isclose(
                a["duration_ms"], b["duration_ms"], rel_tol=0, abs_tol=1e-6):
            raise ValueError(f"Source duration differs for utterance {a['id']}")


def _reference_hash(records: Sequence[dict]) -> str:
    payload = [[r["id"], r["talk_id"], r["reference"], r.get("duration_ms"),
                r.get("segment_index"), r.get("expected_talk_segments"), r.get("offset")]
               for r in records]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def _concatenate_validated(records: Sequence[dict]) -> dict[str, Any]:
    groups: dict[str, list[dict]] = {}
    for record in records:
        groups.setdefault(record["talk_id"], []).append(record)
    concatenated, coverage = [], []
    for talk_id, rows in sorted(groups.items()):
        declared = {r["expected_talk_segments"] for r in rows
                    if r.get("expected_talk_segments") is not None}
        if len(declared) > 1:
            raise ValueError(f"Inconsistent expected_talk_segments for {talk_id}")
        expected = next(iter(declared)) if declared else None
        if expected is not None and len(rows) > expected:
            raise ValueError(f"Observed segment count exceeds expected count for {talk_id}")
        has_ordinal = [r.get("segment_index") is not None for r in rows]
        known_ordinals = [r["segment_index"] for r in rows if r.get("segment_index") is not None]
        if len(set(known_ordinals)) != len(known_ordinals):
            raise ValueError(f"Duplicate segment_index within {talk_id}")
        if expected is not None and any(index >= expected for index in known_ordinals):
            raise ValueError(f"segment_index outside declared within-talk range for {talk_id}")
        reasons = []
        if any(r.get("expected_talk_segments") is None for r in rows):
            reasons.append("missing_expected_talk_segments")
        if expected is not None and len(rows) != expected:
            reasons.append("missing_segments")
        ordering = None
        ordered = rows
        if all(has_ordinal):
            ordered = sorted(rows, key=lambda row: row["segment_index"])
            ordering = "within_talk_zero_based_segment_index"
            if all(r.get("offset") is not None for r in ordered):
                offsets = [r["offset"] for r in ordered]
                if any(b < a - 1e-9 for a, b in zip(offsets, offsets[1:])):
                    raise ValueError(f"Segment indices conflict with chronological offsets for {talk_id}")
        elif any(has_ordinal):
            reasons.append("incomplete_segment_index_metadata")
        elif all(r.get("offset") is not None for r in rows):
            if len({r["offset"] for r in rows}) != len(rows):
                raise ValueError(f"Ambiguous duplicate source offsets without ordinals for {talk_id}")
            ordered = sorted(rows, key=lambda row: row["offset"])
            ordering = "source_offset"
        else:
            reasons.append("missing_segment_order")
        missing_indices = (sorted(set(range(expected)) - set(known_ordinals))
                           if expected is not None and all(has_ordinal) else None)
        complete = not reasons
        item = {
            "talk_id": talk_id, "observed_segments": len(rows),
            "expected_talk_segments": expected,
            "observed_fraction": len(rows) / expected if expected is not None else None,
            "complete": complete, "reasons": reasons, "ordering": ordering,
            "missing_segment_indices": missing_indices,
            "completeness_evidence": ("exact_declared_ordinal_set" if ordering ==
                "within_talk_zero_based_segment_index" else "declared_count_and_unique_offsets"
                if ordering == "source_offset" else "unavailable"),
        }
        coverage.append(item)
        if complete:
            concatenated.append({
                "talk_id": talk_id,
                "segment_ids": [r["id"] for r in ordered],
                "reference": " ".join(r["reference"].strip() for r in ordered),
                "hypothesis": " ".join(r["hypothesis"].strip() for r in ordered),
                "n_segments": len(ordered),
                "unfinished_segments": sum(not r["finished"] for r in ordered),
                "empty_hypotheses": sum(not r["hypothesis"].strip() for r in ordered),
            })
    expected_total = (sum(r["expected_talk_segments"] for r in coverage)
                      if all(r["expected_talk_segments"] is not None for r in coverage) else None)
    return {
        "records": concatenated,
        "coverage": {
            "observed_talks": len(groups), "complete_talks": len(concatenated),
            "incomplete_talks": len(groups) - len(concatenated),
            "observed_segments": len(records),
            "scored_segments": sum(r["n_segments"] for r in concatenated),
            "expected_segments_across_observed_talks": expected_total,
            "observed_fraction_of_declared_segments": len(records) / expected_total if expected_total else None,
            "complete_talk_ids": [r["talk_id"] for r in concatenated],
            "selected_complete_talk_subset": len(concatenated) != len(groups),
            "talks": coverage,
        },
    }


def concatenate_complete_talks(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return ordered complete-talk text and explicit coverage of every talk.

    Completeness concerns source-segment coverage, not successful generation.
    Empty or unfinished segment hypotheses are retained in complete-talk text
    and counted. Reference segments are joined in source order even if their
    original semantic alignment is imperfect; this does not repair CE labels.
    """
    return _concatenate_validated(validate_records(records))


def _checked_statistics(metric, hypotheses: list[str], references: list[list[str]]) -> np.ndarray:
    public_score = float(metric.corpus_score(hypotheses, references).score)
    if not all(hasattr(metric, attr) for attr in
               ("_extract_corpus_statistics", "_compute_score_from_stats")):
        raise RuntimeError("Unsupported SacreBLEU statistic interface")
    stats = np.asarray(metric._extract_corpus_statistics(
        hypotheses, references), dtype=np.int64)
    if stats.ndim != 2 or stats.shape[0] != len(hypotheses):
        raise RuntimeError("Unexpected SacreBLEU sufficient-statistic shape")
    aggregate_score = float(metric._compute_score_from_stats(
        stats.sum(axis=0).tolist()).score)
    if not math.isclose(public_score, aggregate_score, rel_tol=0, abs_tol=1e-10):
        raise RuntimeError("Cached statistics disagree with public corpus score")
    return stats


class _PreparedCorpus:
    """Validated corpus with per-talk sufficient statistics for fast resampling."""
    def __init__(self, records: Sequence[Mapping[str, Any]],
                 metric_config: MetricConfig):
        from sacrebleu.metrics import BLEU, CHRF

        self.records = validate_records(records)
        self.talk_ids = sorted({r["talk_id"] for r in self.records})
        talk_index = {talk: i for i, talk in enumerate(self.talk_ids)}
        indices = np.asarray([talk_index[r["talk_id"]] for r in self.records])
        self.counts = np.bincount(indices, minlength=len(self.talk_ids)).astype(np.int64)
        self.scorers = {
            "BLEU": BLEU(tokenize=metric_config.bleu_tokenizer,
                         lowercase=metric_config.bleu_lowercase,
                         effective_order=False, smooth_method="exp"),
            "chrF2": CHRF(char_order=6, word_order=0, beta=2,
                          lowercase=False, whitespace=False, eps_smoothing=False),
        }
        hypotheses = [r["hypothesis"] for r in self.records]
        references = [[r["reference"] for r in self.records]]
        concatenated = _concatenate_validated(self.records)
        self.concat_records = concatenated["records"]
        self.concat_coverage = concatenated["coverage"]
        self.complete_talk_ids = [r["talk_id"] for r in self.concat_records]
        self.concat_statistics: dict[str, np.ndarray] = {}
        self.statistics: dict[str, np.ndarray] = {}
        self.signatures: dict[str, str] = {}
        for name, metric in self.scorers.items():
            stats = _checked_statistics(metric, hypotheses, references)
            grouped = np.zeros((len(self.talk_ids), stats.shape[1]), dtype=np.int64)
            np.add.at(grouped, indices, stats)
            self.statistics[name] = grouped
            self.signatures[name] = str(metric.get_signature())
            if self.concat_records:
                self.concat_statistics[name] = _checked_statistics(
                    metric, [r["hypothesis"] for r in self.concat_records],
                    [[r["reference"] for r in self.concat_records]])
        self.numeric: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name in NUMERIC_METRICS:
            sums, counts = np.zeros(len(self.talk_ids)), np.zeros(len(self.talk_ids), dtype=np.int64)
            for record, group in zip(self.records, indices):
                if record.get(name) is not None:
                    sums[group] += record[name]
                    counts[group] += 1
            self.numeric[name] = (sums, counts)

    def scores(self, multiplicities: np.ndarray | None = None, *,
               complete_latency: bool = False) -> dict[str, float | None]:
        weights = np.ones(len(self.talk_ids), dtype=np.int64) if multiplicities is None else multiplicities
        n = int(weights @ self.counts)
        if n < 1:
            raise ValueError("Resampled corpus cannot be empty")
        result: dict[str, float | None] = {}
        for name, metric in self.scorers.items():
            statistics = weights @ self.statistics[name]
            result[name] = float(metric._compute_score_from_stats(statistics.tolist()).score)
        for name, (sums, counts) in self.numeric.items():
            valid_n = int(weights @ counts)
            result[name] = (float(weights @ sums) / valid_n
                            if valid_n and (not complete_latency or name not in LATENCY_METRICS or valid_n == n)
                            else None)
        return result

    def concat_scores(self, multiplicities: np.ndarray | None = None) -> dict[str, float | None]:
        if not self.concat_records:
            return {"BLEU": None, "chrF2": None}
        weights = (np.ones(len(self.complete_talk_ids), dtype=np.int64)
                   if multiplicities is None else multiplicities)
        if int(weights.sum()) < 1:
            raise ValueError("Complete-talk resample cannot be empty")
        return {name: float(metric._compute_score_from_stats(
            (weights @ self.concat_statistics[name]).tolist()).score)
                for name, metric in self.scorers.items()}

    def concat_summary(self) -> dict[str, Any]:
        return {
            "status": "scored_complete_talk_subset" if self.concat_records else "no_complete_talks",
            **self.concat_scores(), "coverage": self.concat_coverage,
            "metric_signatures": dict(self.signatures),
            "aggregation": "corpus_statistics_on_one_concatenated_text_pair_per_complete_talk",
            "inference_scope": "selected_complete_talk_subset_only",
            "fewer_than_three_complete_talks": len(self.complete_talk_ids) < 3,
            "unfinished_segments": sum(r["unfinished_segments"] for r in self.concat_records),
            "empty_segment_hypotheses": sum(r["empty_hypotheses"] for r in self.concat_records),
        }

    def summary(self) -> dict[str, Any]:
        scored = self.scores()
        count = len(self.records)
        return {
            "n": count, "n_talks": len(self.talk_ids), **scored,
            "achieved_radius": math.sqrt(float(scored["cost"])),
            "metric_signatures": dict(self.signatures),
            "reference_set_sha256": _reference_hash(self.records),
            "empty_hypotheses": sum(not r["hypothesis"].strip() for r in self.records),
            "unfinished": sum(not r["finished"] for r in self.records),
            "latency_valid_n": {name: int(self.numeric[name][1].sum()) for name in LATENCY_METRICS},
            "hypothesis_words": sum(len(r["hypothesis"].split()) for r in self.records),
            "reference_words": sum(len(r["reference"].split()) for r in self.records),
            "ce_aggregation": "mean_of_per_utterance_token_mean_cross_entropy",
            "quality_aggregation": "corpus_statistics_including_empty_and_unfinished_outputs",
            "talk_concatenated": self.concat_summary(),
        }


def summarize_records(records: Sequence[Mapping[str, Any]], *,
                      metric_config: MetricConfig | None = None) -> dict[str, Any]:
    """Corpus scores and explicit failure/latency denominators on every record."""
    return _PreparedCorpus(records, metric_config or MetricConfig()).summary()


def assess_matching(left: Mapping[str, Any], right: Mapping[str, Any],
                    config: MatchingConfig) -> dict[str, Any]:
    """Report all failures of predeclared empirical cost/latency tolerances."""
    if (not left.get("reference_set_sha256")
            or left.get("reference_set_sha256") != right.get("reference_set_sha256")
            or left["n"] != right["n"]):
        raise ValueError("Matching requires summaries of the identical reference population")
    target = config.rho ** 2
    allowance = max(config.cost_absolute_tolerance,
                    config.cost_relative_tolerance * target)
    reasons = []
    for side, row in (("left", left), ("right", right)):
        cost = _finite(row["cost"], f"{side} cost", nonnegative=True)
        if abs(cost - target) > allowance:
            reasons.append(f"{side}_transport_cost_outside_tolerance")
        if row["empty_hypotheses"]:
            reasons.append(f"{side}_empty_hypotheses")
        if row["unfinished"]:
            reasons.append(f"{side}_unfinished_hypotheses")
        if row["latency_valid_n"][config.latency_metric] != row["n"]:
            reasons.append(f"{side}_latency_missing_for_some_utterances")
    a, b = left.get(config.latency_metric), right.get(config.latency_metric)
    difference = (None if a is None or b is None else
                  _finite(a, "left latency") - _finite(b, "right latency"))
    if difference is None:
        reasons.append("undefined_latency")
    elif abs(difference) > config.latency_tolerance_ms:
        reasons.append("latency_difference_exceeds_tolerance")
    return {
        "eligible": not reasons, "reasons": reasons, "target_radius": config.rho,
        "target_cost": target, "cost_tolerance_absolute": allowance,
        "left_cost": float(left["cost"]), "right_cost": float(right["cost"]),
        "latency_metric": config.latency_metric,
        "latency_difference_ms": difference,
        "latency_tolerance_ms": config.latency_tolerance_ms,
        "basis": "observed_point_estimates_with_fixed_test_population",
        "population_equivalence_certified": False,
    }


def _interval(point: float | None, values: list[float], confidence: float) -> dict[str, Any]:
    if point is None:
        return {"estimate": None, "low": None, "high": None,
                "reason": "metric_not_defined_on_every_paired_utterance"}
    if not values:
        return {"estimate": point, "low": None, "high": None,
                "reason": "insufficient_talk_clusters"}
    tail = (1.0 - confidence) / 2
    low, high = np.quantile(values, (tail, 1 - tail), method="linear")
    return {"estimate": point, "low": float(low), "high": float(high)}


def _paired_concat_prepared(left: Mapping[str, _PreparedCorpus],
                            right: Mapping[str, _PreparedCorpus], *,
                            samples: int, seed: int, confidence: float,
                            resample_seeds: bool) -> dict[str, Any]:
    seeds = sorted(left)
    anchor = left[seeds[0]]
    talks = anchor.complete_talk_ids
    if any(c.complete_talk_ids != talks for c in [*left.values(), *right.values()]):
        raise ValueError("Paired secondary metrics need identical complete-talk coverage")
    count = len(talks)
    point = {
        metric: float(np.mean([left[s].concat_scores()[metric] -
                               right[s].concat_scores()[metric] for s in seeds]))
        if count else None
        for metric in ("BLEU", "chrF2")
    }
    draws: dict[str, list[float]] = {"BLEU": [], "chrF2": []}
    vary_seeds = resample_seeds and len(seeds) > 1
    # Independent deterministic stream keeps the primary bootstrap unchanged.
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    if count >= 2:
        for _ in range(samples):
            weights = np.bincount(rng.integers(0, count, count), minlength=count)
            selected = (rng.integers(0, len(seeds), len(seeds))
                        if vary_seeds else np.arange(len(seeds)))
            values: dict[str, list[float]] = {"BLEU": [], "chrF2": []}
            for index in selected:
                name = seeds[int(index)]
                a, b = left[name].concat_scores(weights), right[name].concat_scores(weights)
                for metric in values:
                    values[metric].append(a[metric] - b[metric])
            for metric in values:
                draws[metric].append(float(np.mean(values[metric])))
    metrics = {name: _interval(point[name], draws[name], confidence) for name in point}
    if count == 0:
        for value in metrics.values():
            value["reason"] = "no_complete_talks"
    return {
        "difference": "left_minus_right", "metrics": metrics,
        "coverage": anchor.concat_coverage,
        "metric_signatures": dict(anchor.signatures),
        "bootstrap": {
            "unit": "complete_TED_talk",
            "method": "crossed_paired_seed_and_complete_talk_percentile" if vary_seeds
            else "paired_complete_talk_percentile_conditional_on_trained_models",
            "replicates": samples if count >= 2 else 0, "requested_replicates": samples,
            "confidence": confidence, "n_complete_talks": count,
            "rng_seed_components": [seed, 1], "training_seeds": seeds,
            "resamples_training_seeds": vary_seeds,
            "intervals_adjusted_for_multiple_comparisons": False,
        },
        "inference_scope": "selected_complete_talk_subset_only",
        "selection_uncertainty_included": False,
        "matched_transport_or_latency_claim": False,
        "cluster_count_note": (
            "No complete talks; secondary scores are unavailable." if count == 0 else
            "One complete talk; no talk-bootstrap interval can be estimated." if count == 1 else
            "Only two complete talks; percentile intervals have very limited resolution and do not quantify talk-selection uncertainty."
            if count == 2 else
            "Intervals are conditional on the supplied complete-talk subset and do not establish its representativeness."),
    }


def _paired_prepared(left: Mapping[str, _PreparedCorpus],
                     right: Mapping[str, _PreparedCorpus], *,
                     samples: int, seed: int, confidence: float,
                     resample_seeds: bool,
                     matching: MatchingConfig | None) -> dict[str, Any]:
    if samples < 2 or not 0 < confidence < 1:
        raise ValueError("At least two replicates and confidence in (0,1) required")
    if not left or set(left) != set(right):
        raise ValueError("Paired methods must have exactly the same nonempty seed set")
    seeds = sorted(left)
    anchor = left[seeds[0]].records
    for name in seeds:
        _check_pair(anchor, left[name].records)
        _check_pair(anchor, right[name].records)
    g = len(left[seeds[0]].talk_ids)
    scores_left = [left[name].scores(complete_latency=True) for name in seeds]
    scores_right = [right[name].scores(complete_latency=True) for name in seeds]
    names = ("BLEU", "chrF2", *NUMERIC_METRICS)
    point = {}
    for metric in names:
        values = [None if a[metric] is None or b[metric] is None
                  else float(a[metric]) - float(b[metric])
                  for a, b in zip(scores_left, scores_right)]
        point[metric] = None if any(v is None for v in values) else float(np.mean(values))
    replicates: dict[str, list[float]] = {name: [] for name in names}
    rng = np.random.default_rng(seed)
    vary_seeds = resample_seeds and len(seeds) > 1
    if g >= 2:
        for _ in range(samples):
            # One common talk resample is used for both methods and all seeds.
            multiplicities = np.bincount(rng.integers(0, g, g), minlength=g)
            selected = (rng.integers(0, len(seeds), len(seeds))
                        if vary_seeds else np.arange(len(seeds)))
            differences = {name: [] for name in names if point[name] is not None}
            for index in selected:
                name = seeds[int(index)]
                a = left[name].scores(multiplicities, complete_latency=True)
                b = right[name].scores(multiplicities, complete_latency=True)
                for metric in differences:
                    differences[metric].append(float(a[metric]) - float(b[metric]))
            for metric, values in differences.items():
                replicates[metric].append(float(np.mean(values)))
    matches = {}
    if matching is not None:
        matches = {name: assess_matching(left[name].summary(), right[name].summary(), matching)
                   for name in seeds}
    return {
        "difference": "left_minus_right",
        "metrics": {name: _interval(point[name], replicates[name], confidence) for name in names},
        "bootstrap": {
            "unit": "TED_talk_cluster",
            "method": "crossed_paired_seed_and_talk_percentile" if vary_seeds else "paired_talk_percentile_conditional_on_trained_models",
            "replicates": samples if g >= 2 else 0, "requested_replicates": samples,
            "confidence": confidence, "rng_seed": seed, "n_talks": g,
            "n_utterances": len(anchor), "training_seeds": seeds,
            "resamples_training_seeds": vary_seeds,
            "seed_aggregation": "mean_of_seedwise_corpus_scores",
            "intervals_adjusted_for_multiple_comparisons": False,
        },
        "metric_signatures": dict(left[seeds[0]].signatures),
        "reference_set_sha256": _reference_hash(anchor),
        "talk_concatenated": _paired_concat_prepared(
            left, right, samples=samples, seed=seed, confidence=confidence,
            resample_seeds=resample_seeds),
        "matching_by_seed": matches,
        "matched_interpretation_eligible": bool(matches) and all(m["eligible"] for m in matches.values()),
        "estimand": "raw_paired_condition_contrast_at_the_supplied_operating_points",
        "population_causal_gap_certified": False,
    }


def paired_talk_bootstrap(left: Sequence[Mapping[str, Any]],
                          right: Sequence[Mapping[str, Any]], *,
                          samples: int = 2000, seed: int = 2027,
                          confidence: float = 0.95,
                          matching: MatchingConfig | None = None,
                          metric_config: MetricConfig | None = None) -> dict[str, Any]:
    """Paired corpus-score differences, resampling whole talks with replacement.

    Unfinished/empty hypotheses remain in quality scores and CE. Matching is
    separately rejected when such failures occur; bootstrap samples are never
    filtered on whether their cost/latency happens to satisfy a tolerance.
    """
    config = metric_config or MetricConfig()
    return _paired_prepared({"fixed": _PreparedCorpus(left, config)},
                            {"fixed": _PreparedCorpus(right, config)},
                            samples=samples, seed=seed, confidence=confidence,
                            resample_seeds=False, matching=matching)


def paired_multiseed_bootstrap(
        left_by_seed: Mapping[int | str, Sequence[Mapping[str, Any]]],
        right_by_seed: Mapping[int | str, Sequence[Mapping[str, Any]]], *,
        samples: int = 2000, seed: int = 2027, confidence: float = 0.95,
        resample_seeds: bool = True, matching: MatchingConfig | None = None,
        metric_config: MetricConfig | None = None) -> dict[str, Any]:
    """Crossed paired seed/talk bootstrap on one identical held-out corpus.

    Seeds and talk clusters are independently resampled. Each seed contributes
    one corpus metric on the common resampled talks; those seed scores are then
    averaged. One training seed yields a talk-only interval and is labeled so.
    """
    config = metric_config or MetricConfig()
    if len({str(k) for k in left_by_seed}) != len(left_by_seed) or len(
            {str(k) for k in right_by_seed}) != len(right_by_seed):
        raise ValueError("Seed keys collide after conversion to string")
    left = {str(k): _PreparedCorpus(v, config) for k, v in left_by_seed.items()}
    right = {str(k): _PreparedCorpus(v, config) for k, v in right_by_seed.items()}
    return _paired_prepared(left, right, samples=samples, seed=seed,
                            confidence=confidence, resample_seeds=resample_seeds,
                            matching=matching)


def analyze_matrix(conditions: Sequence[Mapping[str, Any]], *,
                   matching: MatchingConfig, samples: int = 2000, seed: int = 2027,
                   metric_config: MetricConfig | None = None,
                   defenders: Sequence[str] = DEFENDERS,
                   attacks: Sequence[str] = ATTACKS) -> dict[str, Any]:
    """Summarize the requested method matrix and prespecified paired contrasts.

    A condition is {seed, wait_k, defender, attack, records}. Every condition
    must score the identical corpus. Missing matrix cells are listed explicitly.
    Contrasts require both methods at every supplied training seed for that k;
    no favorable subset of completed seeds is selected.
    """
    if not conditions:
        raise ValueError("At least one condition is required")
    if (not defenders or not attacks or len(set(defenders)) != len(defenders)
            or len(set(attacks)) != len(attacks)
            or not set(defenders) <= set(DEFENDERS)
            or not set(attacks) <= set(ATTACKS)):
        raise ValueError("Invalid declared defender/attack design")
    config = metric_config or MetricConfig()
    prepared: dict[tuple[str, int, str, str], _PreparedCorpus] = {}
    rows = []
    anchor = None
    for condition in conditions:
        for field in ("seed", "wait_k", "defender", "attack", "records"):
            if field not in condition:
                raise ValueError(f"Condition missing {field}")
        defender, attack = condition["defender"], condition["attack"]
        if defender not in defenders or attack not in attacks:
            raise ValueError("Unknown defender/attack in the declared design")
        k = condition["wait_k"]
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("wait_k must be a positive integer")
        key = (str(condition["seed"]), k, defender, attack)
        if key in prepared:
            raise ValueError(f"Duplicate condition: {key}")
        corpus = _PreparedCorpus(condition["records"], config)
        if anchor is not None:
            _check_pair(anchor, corpus.records)
        else:
            anchor = corpus.records
        prepared[key] = corpus
        rows.append({"seed": key[0], "wait_k": k, "defender": defender,
                     "attack": attack, "attack_information": ATTACK_INFORMATION[attack],
                     **corpus.summary()})
    seeds = sorted({key[0] for key in prepared})
    waits = sorted({key[1] for key in prepared})
    missing = [{"seed": s, "wait_k": k, "defender": d, "attack": a}
               for s in seeds for k in waits for d in defenders for a in attacks
               if (s, k, d, a) not in prepared]
    comparisons, unavailable = [], []

    def contrast(k: int, left_condition: tuple[str, str],
                 right_condition: tuple[str, str], kind: str):
        left = {s: prepared[(s, k, *left_condition)] for s in seeds
                if (s, k, *left_condition) in prepared}
        right = {s: prepared[(s, k, *right_condition)] for s in seeds
                 if (s, k, *right_condition) in prepared}
        metadata = {"kind": kind, "wait_k": k,
                    "left": dict(zip(("defender", "attack"), left_condition)),
                    "right": dict(zip(("defender", "attack"), right_condition))}
        if len(left) != len(seeds) or len(right) != len(seeds):
            unavailable.append({**metadata, "reason": "missing_condition_for_one_or_more_training_seeds"})
            return
        pair_matching = replace(matching, rho=0.0) if left_condition[1] == right_condition[1] == "clean" else matching
        value = _paired_prepared(left, right, samples=samples, seed=seed,
                                 confidence=0.95, resample_seeds=True,
                                 matching=pair_matching)
        comparisons.append({**metadata, **value})

    for k in waits:
        for attack in attacks:
            for defender in defenders[1:]:
                contrast(k, (defender, attack), ("nominal", attack),
                         "defender_difference_under_common_attack_class")
            for causal, anticipative in (("causal_picnn", "anticipative_picnn"),
                                        ("adaptive_causal_duchi", "anticipative_duchi")):
                if causal in defenders and anticipative in defenders:
                    contrast(k, (causal, attack), (anticipative, attack),
                             "defender_training_information_contrast_under_common_attack")
            for causal in ("causal_picnn", "adaptive_causal_duchi"):
                if "causal_rnn" in defenders and causal in defenders:
                    contrast(k, ("causal_rnn", attack), (causal, attack),
                             "causal_defender_method_contrast_under_common_attack")
        for defender in defenders:
            if "causal_picnn" in attacks and "anticipative_picnn" in attacks:
                contrast(k, (defender, "causal_picnn"), (defender, "anticipative_picnn"),
                         "source_information_attack_contrast")
            if "adaptive_causal_duchi" in attacks and "anticipative_duchi" in attacks:
                contrast(k, (defender, "adaptive_causal_duchi"), (defender, "anticipative_duchi"),
                         "source_and_reference_information_attack_contrast")
            for causal in ("causal_picnn", "adaptive_causal_duchi"):
                if "causal_rnn" in attacks and causal in attacks:
                    contrast(k, (defender, "causal_rnn"), (defender, causal),
                             "causal_attack_method_contrast")
    return {
        "schema_version": 1, "design": {"defenders": list(defenders), "attacks": list(attacks),
                                        "training_seeds": seeds, "wait_k": waits},
        "rows": sorted(rows, key=lambda row: (row["seed"], row["wait_k"],
                       defenders.index(row["defender"]), attacks.index(row["attack"]))),
        "complete_matrix": not missing, "missing_conditions": missing,
        "comparisons": comparisons, "unavailable_comparisons": unavailable,
        "matching_configuration": asdict(matching),
        "scoring_configuration": asdict(config),
        "interpretation": [
            "BLEU and chrF2 differences are left-minus-right: positive favors left.",
            "CE differences are left-minus-right: negative favors left.",
            "Anticipative Duchi receives held-out references; its contrast includes label information.",
            "All intervals are raw, unadjusted paired contrasts; no multiple-testing significance claim.",
            "Matching checks observed mean cost and latency only, with no population-equivalence certificate.",
            "No test curve interpolation or extrapolation, post-hoc filtering, or best-k selection is performed.",
        ],
    }


def load_conditions(paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    """Load explicit condition lists or runner condition JSON files/directories."""
    result = []
    for item in paths:
        path = Path(item)
        if path.is_dir():
            config_path = path / "config.json"
            config = json.loads(config_path.read_text()) if config_path.exists() else {}
            files = sorted((path / "conditions").glob("*.json"))
            if not files:
                raise ValueError(f"No condition JSON files found in {path}")
            for file in files:
                value = json.loads(file.read_text())
                metadata = dict(value.get("summary", {}))
                metadata.update({name: value[name] for name in
                                 ("seed", "wait_k", "defender", "attack") if name in value})
                if "seed" not in metadata and "seed" in config:
                    metadata["seed"] = config["seed"]
                result.append({**metadata, "records": value["records"]})
        else:
            value = json.loads(path.read_text())
            if isinstance(value, list):
                result.extend(value)
            elif "conditions" in value:
                result.extend(value["conditions"])
            elif "records" in value:
                result.append({**value.get("summary", {}), **value})
            else:
                raise ValueError("Input JSON must contain conditions or a condition with records")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--rho", required=True, type=float)
    parser.add_argument("--cost-relative-tolerance", type=float, default=0.05)
    parser.add_argument("--latency-tolerance-ms", type=float, default=50.0)
    parser.add_argument("--latency-metric", choices=["AL", "LAAL", "DAL"], default="AL")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=2027)
    parser.add_argument("--bleu-tokenizer", default="13a")
    args = parser.parse_args()
    output = analyze_matrix(load_conditions(args.inputs),
                            matching=MatchingConfig(
                                rho=args.rho,
                                cost_relative_tolerance=args.cost_relative_tolerance,
                                latency_metric=args.latency_metric,
                                latency_tolerance_ms=args.latency_tolerance_ms),
                            samples=args.bootstrap_samples, seed=args.bootstrap_seed,
                            metric_config=MetricConfig(bleu_tokenizer=args.bleu_tokenizer))
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(destination)


if __name__ == "__main__":
    main()
