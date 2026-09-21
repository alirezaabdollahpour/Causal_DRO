"""Render measured translation results and paired comparisons.

Corpus scores are recomputed independently for every trained seed. The
arithmetic mean and sample standard deviation across those seed-level scores
are reported; sentence BLEU is never averaged and seed replicas are never
concatenated into an enlarged evaluation corpus. Missing seeds remain explicit.
Curves use observed AL, without matching by interpolation or extrapolation.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import csv
import json
import math
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

from .benchmark_analysis import (ATTACKS, DEFENDERS, MatchingConfig, MetricConfig,
                                 assess_matching, load_conditions, summarize_records)

LABELS = {"nominal": "Nominal", "clean": "Clean", "causal_picnn": "C-PICNN",
          "causal_rnn": "C-RNN", "anticipative_picnn": "A-PICNN", "adaptive_causal_duchi": "ACD",
          "anticipative_duchi": "A-Duchi"}
METRICS = ("BLEU", "chrF2", "ce", "cost", "achieved_radius", "AL", "LAAL", "AP", "DAL")
TABLE_METRICS = {"BLEU": ("bleu", 2), "chrF2": ("chrf2", 2),
                 "ce": ("ce", 4), "cost": ("cost", 3), "AL": ("latency_al", 1)}


def _key(row):
    return str(row["seed"]), int(row["wait_k"]), row["defender"], row["attack"]


def _mean_sd(values):
    if not values or any(value is None for value in values):
        return None, None
    numeric = [float(value) for value in values]
    if any(not math.isfinite(value) for value in numeric):
        raise ValueError("Nonfinite values cannot be rendered")
    return statistics.mean(numeric), statistics.stdev(numeric) if len(numeric) > 1 else None


def aggregate_conditions(conditions: Sequence[Mapping[str, Any]], analysis: Mapping[str, Any],
                         *, expected_seeds: Sequence[int | str] | None = None,
                         expected_waits: Sequence[int] | None = None,
                         defenders: Sequence[str] = DEFENDERS,
                         attacks: Sequence[str] = ATTACKS) -> dict[str, Any]:
    """Verify the analysis against raw records before displaying its findings."""
    if not conditions or analysis.get("schema_version") != 1:
        raise ValueError("Nonempty measured conditions and analysis schema version1 required")
    metric_config = MetricConfig(**analysis["scoring_configuration"])
    matching = MatchingConfig(**analysis["matching_configuration"])
    analyzed = {_key(row): row for row in analysis["rows"]}
    if len(analyzed) != len(analysis["rows"]):
        raise ValueError("Analysis contains duplicate condition keys")
    by_seed, reference_hash, signatures = {}, None, None
    for condition in conditions:
        key = _key(condition)
        if key in by_seed:
            raise ValueError(f"Duplicate raw condition: {key}")
        if key not in analyzed:
            raise ValueError(f"Raw condition absent from supplied analysis: {key}")
        if key[2] not in defenders or key[3] not in attacks or key[1] < 1:
            raise ValueError("Condition is outside the declared method design")
        fresh = summarize_records(condition["records"], metric_config=metric_config)
        saved = analyzed[key]
        for name in (*METRICS, "n", "n_talks", "unfinished", "empty_hypotheses"):
            left, right = fresh[name], saved.get(name)
            if (left is None) != (right is None) or (left is not None and
                    not math.isclose(float(left), float(right), abs_tol=1e-8, rel_tol=1e-9)):
                raise ValueError(f"Stale or incompatible analysis for {key}: {name}")
        for name in ("reference_set_sha256", "latency_valid_n", "metric_signatures"):
            if fresh[name] != saved.get(name):
                raise ValueError(f"Analysis provenance disagrees with raw records: {key}: {name}")
        if reference_hash is not None and fresh["reference_set_sha256"] != reference_hash:
            raise ValueError("All conditions must evaluate the identical reference population")
        if signatures is not None and fresh["metric_signatures"] != signatures:
            raise ValueError("Scoring signatures differ across conditions")
        reference_hash, signatures = fresh["reference_set_sha256"], fresh["metric_signatures"]
        by_seed[key] = {"seed": key[0], "wait_k": key[1], "defender": key[2],
                        "attack": key[3], **fresh}
    if set(by_seed) != set(analyzed):
        raise ValueError("Analysis contains conditions not supplied as raw records")
    seeds = sorted({str(seed) for seed in (expected_seeds or analysis["design"]["training_seeds"])})
    waits = sorted(set(expected_waits or analysis["design"]["wait_k"]))
    if not {key[0] for key in by_seed} <= set(seeds) or not {key[1] for key in by_seed} <= set(waits):
        raise ValueError("Observed conditions lie outside the expected seed/wait-k design")
    rows = []
    for k in waits:
        for defender in defenders:
            for attack in attacks:
                available = [by_seed[(seed, k, defender, attack)] for seed in seeds
                             if (seed, k, defender, attack) in by_seed]
                present = [row["seed"] for row in available]
                target = 0.0 if attack == "clean" else matching.rho ** 2
                tolerance = max(matching.cost_absolute_tolerance, matching.cost_relative_tolerance * target)
                row = {"wait_k": k, "defender": defender, "attack": attack,
                       "seeds": present, "n_seeds": len(available), "expected_n_seeds": len(seeds),
                       "missing_seeds": [seed for seed in seeds if seed not in present],
                       "complete_seed_set": len(available) == len(seeds),
                       "n_per_seed": available[0]["n"] if available else None,
                       "n_talks": available[0]["n_talks"] if available else None,
                       "evaluated_seed_utterances": sum(r["n"] for r in available),
                       "finished_seed_utterances": sum(r["n"] - r["unfinished"] for r in available),
                       "unfinished_seed_utterances": sum(r["unfinished"] for r in available),
                       "empty_seed_utterances": sum(r["empty_hypotheses"] for r in available),
                       "latency_valid_seed_utterances": sum(r["latency_valid_n"]["AL"] for r in available),
                       "all_generations_finished": bool(available) and all(not r["unfinished"] for r in available),
                       "no_empty_hypotheses": bool(available) and all(not r["empty_hypotheses"] for r in available),
                       "latency_defined_for_every_utterance": bool(available) and all(r["latency_valid_n"]["AL"] == r["n"] for r in available),
                       "target_cost": target,
                       "cost_within_tolerance_all_seeds": bool(available) and all(abs(r["cost"] - target) <= tolerance for r in available)}
                for name in METRICS:
                    row[name + "_mean"], row[name + "_seed_sd"] = _mean_sd([r[name] for r in available])
                rows.append(row)
    return {"schema_version": 1, "training_seeds": seeds, "wait_k": waits,
            "complete_matrix": all(row["complete_seed_set"] for row in rows),
            "metric_signatures": signatures, "reference_set_sha256": reference_hash,
            "matching_configuration": analysis["matching_configuration"],
            "aggregation": "mean and sample SD (ddof=1) of seedwise corpus scores; not sentence-score averaging",
            "standard_deviation_scope": "between trained seeds; not a confidence interval; undefined for one seed",
            "failure_count_scope": "seed-utterance evaluations, not distinct speech examples",
            "operating_points": "observed fixed wait-k; no latency interpolation or extrapolation",
            "rows": rows, "rows_by_seed": list(by_seed.values())}


def _paired_comparison(analysis, k, attack, left, right, expected_seeds):
    for comparison in analysis.get("comparisons", []):
        if comparison["wait_k"] != k or comparison["left"]["attack"] != attack or comparison["right"]["attack"] != attack:
            continue
        pair = comparison["left"]["defender"], comparison["right"]["defender"]
        if pair not in ((left, right), (right, left)):
            continue
        if {str(seed) for seed in comparison["bootstrap"]["training_seeds"]} != set(expected_seeds):
            raise ValueError("Paired interval uses a different set of trained seeds")
        value = dict(comparison["metrics"]["BLEU"])
        if pair == (right, left):
            value = {**value, "estimate": -value["estimate"] if value["estimate"] is not None else None,
                     "low": -value["high"] if value["high"] is not None else None,
                     "high": -value["low"] if value["low"] is not None else None}
        return {**value, "confidence": comparison["bootstrap"]["confidence"],
                "intervals_adjusted_for_multiple_comparisons": False}
    return None


def rank_fixed_conditions(aggregated, analysis):
    """Descriptive leaders at each shared attack/k; no cross-attack ranking."""
    by_seed = {_key(row): row for row in aggregated["rows_by_seed"]}
    matching = MatchingConfig(**aggregated["matching_configuration"])
    rankings = []
    for k in aggregated["wait_k"]:
        for attack in ATTACKS:
            rows = [row for row in aggregated["rows"] if row["wait_k"] == k and row["attack"] == attack]
            missing = [row["defender"] for row in rows if not row["complete_seed_set"]]
            metadata = {"wait_k": k, "attack": attack, "scope": "descriptive_fixed_k_common_attack_class",
                        "statistical_winner_claim": False}
            if missing:
                rankings.append({**metadata, "available": False, "missing_defenders": missing})
                continue
            rows.sort(key=lambda row: (-row["BLEU_mean"], DEFENDERS.index(row["defender"])))
            best, runner = rows[:2]
            leaders = [row["defender"] for row in rows if math.isclose(row["BLEU_mean"], best["BLEU_mean"], abs_tol=1e-10, rel_tol=0)]
            adversarial = [row for row in rows if row["defender"] != "nominal"]
            at_leaders = [row["defender"] for row in adversarial if math.isclose(row["BLEU_mean"], adversarial[0]["BLEU_mean"], abs_tol=1e-10, rel_tol=0)]
            pair_config = replace(matching, rho=0.0) if attack == "clean" else matching
            matched = {seed: assess_matching(by_seed[(seed, k, best["defender"], attack)],
                                             by_seed[(seed, k, runner["defender"], attack)], pair_config)
                       for seed in aggregated["training_seeds"]}
            interval = _paired_comparison(analysis, k, attack, best["defender"], runner["defender"], aggregated["training_seeds"])
            if interval is not None and not math.isclose(interval["estimate"], best["BLEU_mean"] - runner["BLEU_mean"], abs_tol=1e-8, rel_tol=1e-9):
                raise ValueError("Paired analysis interval disagrees with measured leader contrast")
            rankings.append({**metadata, "available": True, "leaders": leaders,
                             "adversarial_training_leaders": at_leaders,
                             "ordered_defenders": [row["defender"] for row in rows],
                             "leader_bleu": best["BLEU_mean"], "leader_bleu_seed_sd": best["BLEU_seed_sd"],
                             "runner_up": runner["defender"], "runner_up_bleu": runner["BLEU_mean"],
                             "leader_minus_runner_up_bleu": best["BLEU_mean"] - runner["BLEU_mean"],
                             "paired_interval": interval, "matching_by_seed": matched,
                             "matched_interpretation_eligible": all(value["eligible"] for value in matched.values()),
                             "post_selection_note": "Leader chosen from the measured table; intervals are unadjusted pairwise contrasts, not selection-adjusted winner tests"})
    return rankings


def _csv(path, rows):
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                             for name, value in row.items()})


def _cell(row, metric, digits, *, tex=False):
    mean, sd = row[metric + "_mean"], row[metric + "_seed_sd"]
    if mean is None:
        return "--"
    value = f"{mean:.{digits}f}"
    if sd is not None:
        value += (r" \pm " if tex else " ± ") + f"{sd:.{digits}f}"
    if tex:
        value = "$" + value + "$"
    if not row["complete_seed_set"]:
        value += (r"\textsuperscript{*}" if tex else "*")
    return value


def _write_tables(output, result, *, defenders=DEFENDERS, attacks=ATTACKS):
    index = {(row["wait_k"], row["defender"], row["attack"]): row for row in result["rows"]}
    for metric, (stem, digits) in TABLE_METRICS.items():
        lines = ["% Measured seedwise corpus/utterance means; plus-minus denotes sample SD across seeds.",
                 "% A star marks an incomplete expected seed set; -- means unavailable.",
                 r"\begin{tabular}{ll" + "r" * len(attacks) + "}", r"\toprule",
                 "$k$ & Defender & " + " & ".join(LABELS[a] for a in attacks) + r" \\", r"\midrule"]
        for ki, k in enumerate(result["wait_k"]):
            matrix = []
            for defender in defenders:
                cells = [index[(k, defender, attack)] for attack in attacks]
                matrix.append({"defender": LABELS[defender], **{LABELS[a]: _cell(row, metric, digits) for a, row in zip(attacks, cells)}})
                lines.append(f"{k} & {LABELS[defender]} & " + " & ".join(_cell(row, metric, digits, tex=True) for row in cells) + r" \\")
            _csv(output / f"matrix_{stem}_k{k}.csv", matrix)
            if ki < len(result["wait_k"]) - 1:
                lines.append(r"\midrule")
        lines.extend([r"\bottomrule", r"\end{tabular}"])
        (output / f"table_{stem}.tex").write_text("\n".join(lines) + "\n")
    lines = ["% Each cell: finished / evaluated seed-utterances; empty hypotheses in parentheses.",
             "% Counts sum over trained seeds; they are not counts of distinct speech examples.",
             r"\begin{tabular}{ll" + "r" * len(attacks) + "}", r"\toprule", "$k$ & Defender & " + " & ".join(LABELS[a] for a in attacks) + r" \\", r"\midrule"]
    for ki, k in enumerate(result["wait_k"]):
        for defender in defenders:
            cells = []
            for attack in attacks:
                row = index[(k, defender, attack)]
                value = "--" if not row["n_seeds"] else f"{row['finished_seed_utterances']}/{row['evaluated_seed_utterances']} ({row['empty_seed_utterances']})"
                if row["n_seeds"] and not row["complete_seed_set"]:
                    value += r"\textsuperscript{*}"
                cells.append(value)
            lines.append(f"{k} & {LABELS[defender]} & " + " & ".join(cells) + r" \\")
        if ki < len(result["wait_k"]) - 1:
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (output / "table_generation.tex").write_text("\n".join(lines) + "\n")


def _write_curves(output, result, label, *, defenders=DEFENDERS, attacks=ATTACKS):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    palette = {"nominal":"#666666", "causal_picnn":"#0072B2",
               "causal_rnn":"#D55E00", "anticipative_picnn":"#E69F00",
               "adaptive_causal_duchi":"#009E73", "anticipative_duchi":"#CC79A7"}
    colors = tuple(palette[name] for name in defenders)
    style = {"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
             "font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42}
    with plt.rc_context(style):
        for metric, stem, ylabel in (("BLEU", "bleu", "Corpus BLEU"), ("chrF2", "chrf2", "Corpus chrF2")):
            columns = 3 if len(attacks) <= 5 else 4
            fig, axes = plt.subplots(2, columns, figsize=(12.6 if columns==3 else 16.4, 7.1), squeeze=False)
            for attack, ax in zip(attacks, axes.flat):
                for defender, color in zip(defenders, colors):
                    points = sorted([row for row in result["rows"] if row["defender"] == defender and row["attack"] == attack], key=lambda row: row["wait_k"])
                    x, y = [], []
                    for row in points:
                        usable = row["complete_seed_set"] and row["AL_mean"] is not None and row[metric + "_mean"] is not None
                        x.append(row["AL_mean"] if usable else float("nan"))
                        y.append(row[metric + "_mean"] if usable else float("nan"))
                    ax.plot(x, y, color=color, linewidth=1.2)
                    for row, xpos, ypos in zip(points, x, y):
                        if not math.isfinite(xpos) or not math.isfinite(ypos):
                            continue
                        successful = row["all_generations_finished"] and row["no_empty_hypotheses"] and row["latency_defined_for_every_utterance"]
                        ax.errorbar(xpos, ypos, xerr=row["AL_seed_sd"], yerr=row[metric + "_seed_sd"],
                                    fmt="o", color=color, markersize=4.5, linewidth=.8,
                                    markerfacecolor=color if successful else "white", capsize=2)
                        ax.annotate(str(row["wait_k"]), (xpos, ypos), xytext=(3, 4), textcoords="offset points", fontsize=7, color=color)
                ax.set_title(LABELS[attack])
                ax.set_xlabel("Observed Average Lagging (ms)")
                ax.set_ylabel(ylabel)
                ax.grid(alpha=.2, linewidth=.5)
                ax.spines[["top", "right"]].set_visible(False)
            legend = axes.flat[-1]
            legend.axis("off")
            for unused in axes.flat[len(attacks):-1]:
                unused.axis("off")
            handles = [Line2D([], [], color=color, marker="o", markersize=4.5, label=LABELS[d]) for d, color in zip(defenders, colors)]
            legend.legend(handles=handles, loc="upper left", frameon=False)
            notes = ("Numbers beside points: wait-k.\n"
                     "Bars: sample SD across trained seeds.\n"
                     "Open markers: empty/unfinished output\n"
                     "or incomplete latency coverage.\n"
                     "Only complete seed sets are plotted.\n"
                     "Lines connect observed operating points;\n"
                     "they do not imply matched latency.\n"
                     "A-Duchi also receives test references.")
            legend.text(.04, .40, notes, transform=legend.transAxes, va="top", fontsize=8, linespacing=1.5)
            fig.suptitle(label, y=.99, fontsize=12)
            fig.tight_layout(rect=(0, 0, 1, .97))
            fig.savefig(output / f"quality_latency_{stem}.pdf", bbox_inches="tight")
            fig.savefig(output / f"quality_latency_{stem}.png", dpi=220, bbox_inches="tight")
            plt.close(fig)


def _write_rankings(output, rankings):
    paragraphs = ["Rankings below are descriptive comparisons at the same attack class and wait-k. They use mean seedwise corpus BLEU. A highest observed score is not a selection-adjusted statistical winner. Attacks are evaluated at their measured transport cost and latency; matching flags report the declared empirical checks."]
    flat = []
    for row in rankings:
        prefix = f"At k={row['wait_k']} under {LABELS[row['attack']]}"
        if not row["available"]:
            paragraphs.append(prefix + ", the defender ranking is unavailable because one or more expected seed/defender conditions are missing.")
            flat.append({"wait_k": row["wait_k"], "attack": row["attack"], "available": False,
                         "leaders": [], "adversarial_training_leaders": [], "leader_bleu": None,
                         "runner_up": None, "difference": None, "interval_low": None, "interval_high": None,
                         "matched_interpretation_eligible": None})
            continue
        names = ", ".join(LABELS[name] for name in row["leaders"])
        at_names = ", ".join(LABELS[name] for name in row["adversarial_training_leaders"])
        verb = "has" if len(row["leaders"]) == 1 else "share"
        text = prefix + f", {names} {verb} the highest observed mean BLEU ({row['leader_bleu']:.2f}); the highest among adversarially trained defenders is {at_names}."
        interval = row["paired_interval"]
        if len(row["leaders"]) == 1:
            text += f" The difference from {LABELS[row['runner_up']]} is {row['leader_minus_runner_up_bleu']:.2f} BLEU"
            if interval and interval["low"] is not None:
                text += f" (unadjusted paired {100*interval['confidence']:.0f}% interval [{interval['low']:.2f}, {interval['high']:.2f}])"
            text += "."
        text += (" This pair passes the declared empirical cost, latency and generation checks for every seed."
                 if row["matched_interpretation_eligible"] else
                 " This pair fails one or more cost, latency or generation checks, so its fixed-k ordering does not establish superiority at matched operating points.")
        paragraphs.append(text)
        flat.append({"wait_k": row["wait_k"], "attack": row["attack"], "available": True,
                     "leaders": row["leaders"], "adversarial_training_leaders": row["adversarial_training_leaders"],
                     "leader_bleu": row["leader_bleu"], "runner_up": row["runner_up"],
                     "difference": row["leader_minus_runner_up_bleu"],
                     "interval_low": interval["low"] if interval else None,
                     "interval_high": interval["high"] if interval else None,
                     "matched_interpretation_eligible": row["matched_interpretation_eligible"]})
    text = "\n\n".join(paragraphs) + "\n"
    (output / "measured_rankings.txt").write_text(text)
    (output / "measured_rankings.tex").write_text(text.replace("%", r"\%"))
    _csv(output / "fixed_condition_rankings.csv", flat)
    (output / "fixed_condition_rankings.json").write_text(json.dumps(rankings, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def render(conditions, analysis, output, *, label="MuST-C English–Arabic: measured quality and latency",
           expected_seeds=None, expected_waits=None, require_complete=False, plots=True):
    """Write figures, full matrices, diagnostics and explicitly scoped rankings."""
    result = aggregate_conditions(conditions, analysis, expected_seeds=expected_seeds, expected_waits=expected_waits)
    if require_complete and not result["complete_matrix"]:
        raise ValueError("The expected defender × attack × seed × wait-k matrix is incomplete")
    rankings = rank_fixed_conditions(result, analysis)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "aggregated_conditions.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    _csv(output / "aggregated_conditions.csv", result["rows"])
    _csv(output / "conditions_by_seed.csv", result["rows_by_seed"])
    _write_tables(output, result)
    _write_rankings(output, rankings)
    if plots:
        _write_curves(output, result, label)
    (output / "README.md").write_text(
        f"# {label}\n\n"
        f"Expected training seeds: {', '.join(result['training_seeds'])}; wait-k: {result['wait_k']}. "
        f"Complete declared matrix: {result['complete_matrix']}.\n\n"
        "Every displayed quality score was recomputed from raw condition records using corpus statistics. "
        "Tables show the mean ± sample SD of seedwise corpus BLEU/chrF2 or seedwise mean CE, cost and AL. "
        "For a single seed, SD is unavailable and omitted. A star indicates an incomplete expected seed set. "
        "Curves omit incomplete seed sets and retain actual observed AL; they contain no latency matching by interpolation. "
        "Empty and unfinished hypotheses remain in quality scores. AL means use the explicit valid-output denominators "
        "reported in aggregated_conditions.csv; open plot markers identify generation or coverage failures.\n\n"
        "table_generation.tex cells give finished/evaluated seed-utterances followed by the empty-output count. "
        "These totals sum model-seed evaluations of the same speech, rather than distinct examples. "
        "table_cost.tex, table_latency_al.tex and table_ce.tex provide separate operating-point diagnostics.\n\n"
        "The measured ranking text is descriptive and restricted to a common attack and k. Its paired intervals, "
        "when available, come from benchmark_analysis and are not adjusted for choosing a leader or multiple comparisons. "
        "A positive gap alone does not establish a statistical winner. Matched flags require every seed to pass the "
        "declared empirical cost/latency/generation checks; they do not certify population equivalence. "
        "Anticipative Duchi receives held-out references, so its attack contrast also changes label information.\n\n"
        "Metric signatures:\n\n" + "\n".join(f"- {name}: `{signature}`" for name, signature in result["metric_signatures"].items()) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Seed run directories or explicit condition JSON files")
    parser.add_argument("--analysis", required=True, help="Matching benchmark_analysis JSON")
    parser.add_argument("--output", required=True)
    parser.add_argument("--label", default="MuST-C English–Arabic: measured quality and latency")
    parser.add_argument("--expected-seeds", nargs="+")
    parser.add_argument("--expected-wait-k", type=int, nargs="+")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    result = render(load_conditions(args.inputs), json.loads(Path(args.analysis).read_text()), args.output,
                    label=args.label, expected_seeds=args.expected_seeds, expected_waits=args.expected_wait_k,
                    require_complete=args.require_complete, plots=not args.no_plots)
    print(json.dumps({"output": str(Path(args.output).resolve()), "complete_matrix": result["complete_matrix"],
                      "training_seeds": result["training_seeds"], "wait_k": result["wait_k"]}))


if __name__ == "__main__":
    main()
