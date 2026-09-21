"""Render only measured artifacts, without substituting planned benchmark data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def render(directory: str | Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(directory)
    result = json.loads((root / "summary.json").read_text())
    rows = result["rows"]
    defenders = list(dict.fromkeys(row["defender"] for row in rows))
    attacks = list(dict.fromkeys(row["attack"] for row in rows))
    fig, axes = plt.subplots(1, len(defenders), figsize=(5 * len(defenders), 4), squeeze=False)
    for defender, ax in zip(defenders, axes[0]):
        for attack in attacks:
            data = [r for r in rows if r["defender"] == defender and r["attack"] == attack and r["AL"] is not None]
            data.sort(key=lambda r: r["wait_k"])
            if not data:
                continue
            line, = ax.plot([r["AL"] for r in data], [r["BLEU"] for r in data],
                            marker="o", markersize=4, label=attack.replace("_", " "))
            for r in data:
                ax.annotate(str(r["wait_k"]), (r["AL"], r["BLEU"]), fontsize=7,
                            xytext=(3, 4), textcoords="offset points")
        ax.set(title=defender.replace("_", " "), xlabel="Average Lagging (ms)", ylabel="Corpus SacreBLEU")
        ax.grid(alpha=.2)
    axes[0, -1].legend(fontsize=7)
    title = "Synthetic execution audit — not MuST-C" if not result["is_mustc_result"] else "MuST-C pilot: measured quality / latency"
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(root / "quality_latency.pdf", bbox_inches="tight")
    fig.savefig(root / "quality_latency.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    lines = ["% Generated from summary.json; no interpolated numbers in this table.",
             r"\begin{tabular}{lllrrrr}", r"\toprule",
             r"Defender & Attack & $k$ & CE & BLEU & AL (ms) & $\widehat D$ \\", r"\midrule"]
    for r in rows:
        al = "--" if r["AL"] is None else f'{r["AL"]:.1f}'
        label = lambda value: value.replace("_", r"\_")
        lines.append(f'{label(r["defender"])} & {label(r["attack"])} & {r["wait_k"]} & '
                     f'{r["ce"]:.4f} & {r["BLEU"]:.2f} & {al} & {r["cost"]:.5f} ' + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (root / "results_table.tex").write_text("\n".join(lines) + "\n")
    return root / "quality_latency.pdf"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    render(parser.parse_args().directory)
