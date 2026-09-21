"""Corpus quality and computation-unaware speech latency, in milliseconds.

AL uses reference word count; DAL uses hypothesis word count, matching
facebookresearch/SimulEval's latency_scorer.py. Empty hypotheses have undefined
latency and are counted explicitly rather than silently assigned zero lag.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Sequence

import numpy as np


def latency_scores(delays_ms: Sequence[float], duration_ms: float,
                   reference_words: int) -> dict[str, float | None]:
    if not math.isfinite(duration_ms) or duration_ms <= 0 or reference_words < 1:
        raise ValueError("Positive duration and reference word count required")
    d = np.asarray(delays_ms, dtype=float)
    if (not np.isfinite(d).all() or np.any(d < 0)
            or np.any(np.diff(d) < -1e-7) or np.any(d > duration_ms + 1e-4)):
        raise ValueError("Content delays must be monotone and inside the utterance")
    if not len(d):
        return dict(AL=None, LAAL=None, AP=None, DAL=None)
    complete = np.flatnonzero(d >= duration_ms - 1e-5)
    tau = int(complete[0] + 1) if len(complete) else len(d)
    indices = np.arange(tau)
    al = np.mean(d[:tau] - indices * duration_ms / reference_words)
    laal = np.mean(d[:tau] - indices * duration_ms / max(reference_words, len(d)))
    step = duration_ms / len(d)
    g_prime = d[0]
    dal = float(g_prime)
    for i, delay in enumerate(d[1:], 1):
        g_prime = max(float(delay), g_prime + step)
        dal += g_prime - i * step
    # SimulEval's reference-length default also applies to AP.
    return dict(AL=float(al), LAAL=float(laal),
                AP=float(d.sum() / (duration_ms * reference_words)), DAL=dal / len(d))


def corpus_bleu(hypotheses: Sequence[str], references: Sequence[str]) -> tuple[float, str]:
    if len(hypotheses) != len(references) or not references:
        raise ValueError("A nonempty paired corpus is required")
    try:
        from sacrebleu.metrics import BLEU
    except ImportError as exc:
        raise RuntimeError("Install live_translation/requirements.txt for SacreBLEU") from exc
    scorer = BLEU(tokenize="13a", lowercase=False, effective_order=False)
    score = scorer.corpus_score(list(hypotheses), [list(references)])
    return float(score.score), str(scorer.get_signature())


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    bleu, signature = corpus_bleu([r["hypothesis"] for r in records],
                                  [r["reference"] for r in records])
    result: dict[str, Any] = {"n": len(records), "BLEU": bleu,
                             "sacrebleu_signature": signature}
    for name in ("ce", "cost", "AL", "LAAL", "AP", "DAL"):
        values = [float(r[name]) for r in records if r.get(name) is not None]
        result[name] = float(np.mean(values)) if values else None
    result["empty_hypotheses"] = sum(not r["hypothesis"].strip() for r in records)
    result["unfinished"] = sum(not r["finished"] for r in records)
    result["latency_valid_n"] = sum(r.get("AL") is not None for r in records)
    result["hypothesis_words"] = sum(len(r["hypothesis"].split()) for r in records)
    result["reference_words"] = sum(len(r["reference"].split()) for r in records)
    return result


def paired_bootstrap_difference(left: list[dict], right: list[dict], *,
                                metric: str = "BLEU", samples: int = 1000,
                                seed: int = 2027) -> dict[str, float]:
    """Resample paired utterances, recomputing corpus BLEU each time (left-right)."""
    if samples < 2 or not left or [r["id"] for r in left] != [r["id"] for r in right]:
        raise ValueError("Bootstrap needs matching ordered IDs and >=2 replicates")
    rng = np.random.default_rng(seed)
    def difference(ix):
        if metric == "BLEU":
            ref = [left[i]["reference"] for i in ix]
            return (corpus_bleu([left[i]["hypothesis"] for i in ix], ref)[0]
                    - corpus_bleu([right[i]["hypothesis"] for i in ix], ref)[0])
        return float(np.mean([left[i][metric] - right[i][metric] for i in ix]))
    values = [difference(rng.integers(0, len(left), len(left))) for _ in range(samples)]
    low, high = np.quantile(values, [0.025, 0.975])
    return dict(estimate=difference(range(len(left))), low=float(low), high=float(high))


def interpolate_latency(curve: list[dict], target_al: float, *,
                        radius: float, radius_tolerance: float = 0.05) -> dict | None:
    """Descriptive interpolation between measured, budget-compatible points.

    Never extrapolates, never chooses the best BLEU at a duplicate latency, and
    excludes rows with missing hypotheses or excessive realized-cost mismatch.
    This is an interpolation of corpus statistics, not a measured mixed policy.
    """
    rows = [r for r in curve if r.get("AL") is not None
            and not r.get("empty_hypotheses", 0) and not r.get("unfinished", 0)
            and abs(r["cost"] - radius**2) <= max(1e-8, radius_tolerance * radius**2)]
    rows.sort(key=lambda r: (r["AL"], r["wait_k"]))
    if not rows or target_al < rows[0]["AL"] or target_al > rows[-1]["AL"]:
        return None
    for row in rows:
        if abs(target_al - row["AL"]) < 1e-8:
            return {"AL": target_al, "BLEU": row["BLEU"], "ce": row["ce"],
                    "cost": row["cost"], "k_pair": [row["wait_k"]]}
    for a, b in zip(rows, rows[1:]):
        if a["AL"] < target_al < b["AL"]:
            w = (target_al - a["AL"]) / (b["AL"] - a["AL"])
            return {"AL": target_al, **{m: (1-w)*a[m] + w*b[m]
                    for m in ("BLEU", "ce", "cost")},
                    "k_pair": [a["wait_k"], b["wait_k"]], "weight": w}
    return None


def matched_comparisons(rows: list[dict], radius: float,
                        tolerance: float = 0.05) -> list[dict]:
    """Report empirical learned-attack differences only on overlapping AL support."""
    by_defender: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_defender[row["defender"]][row["attack"]].append(row)
    output = []
    for defender, conditions in by_defender.items():
        for causal in ("causal_picnn", "causal_rnn", "adaptive_causal_duchi"):
            a, c = conditions.get("anticipative_picnn", []), conditions.get(causal, [])
            points = sorted({r["AL"] for r in a+c if r.get("AL") is not None})
            for lag in points:
                ai = interpolate_latency(a, lag, radius=radius, radius_tolerance=tolerance)
                ci = interpolate_latency(c, lag, radius=radius, radius_tolerance=tolerance)
                if ai is not None and ci is not None:
                    output.append(dict(defender=defender, causal_method=causal,
                        AL=lag, empirical_CE_difference=ai["ce"]-ci["ce"],
                        empirical_BLEU_difference=ci["BLEU"]-ai["BLEU"],
                        anticipative=ai, causal=ci, kind="descriptive_interpolation"))
    return output
