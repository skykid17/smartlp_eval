#!/usr/bin/env python3
"""
Post-hoc query_count cap trade-off analysis for decomposed regex generation.

Reads the completed decomposed RAG results file and simulates the effect of
varying the query_count cap on latency, compilation rate, and functional
accuracy.  No LLM calls are made; all figures are derived from the existing
per-sample data.

Outputs (written to data/eval/output/tradeoff/):
    query_count_tradeoff_raw.json      – per-sample simulation detail
    query_count_tradeoff_summary.json  – one row per integer cap
    query_count_tradeoff.png           – dual-axis bar+line chart
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_FILE = REPO_ROOT / "data" / "eval" / "output" / "generator_results_decomposed_rag.json"
OUTPUT_DIR = REPO_ROOT / "data" / "eval" / "output" / "tradeoff"

FUNCTIONAL_ACCURACY_THRESHOLD = 0.5  # score >= 0.5 counts as functionally accurate


def load_results(path: Path) -> tuple[dict, dict]:
    """Return (samples_dict, metadata_dict) from the results file."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    samples = {}
    metadata = {}
    for key, val in raw.items():
        if isinstance(val, dict):
            samples[key] = val
        else:
            metadata[key] = val
    return samples, metadata


def simulate(samples: dict) -> tuple[dict, dict]:
    """Run the cap sweep and return (raw_per_sample, summary)."""

    # Determine max rounds observed
    rounds_used_map = {sid: s["query_count"] for sid, s in samples.items()}
    max_rounds = max(rounds_used_map.values())
    cap_range = list(range(1, max_rounds + 1))

    # Per-round latency estimate for each sample
    per_round_latency = {
        sid: s["generation_time"] / s["query_count"]
        for sid, s in samples.items()
    }

    raw: dict[str, dict] = {}   # cap -> log_id -> simulated record
    summary: dict[str, dict] = {}

    for cap in cap_range:
        cap_records = {}
        latencies = []
        compiled_count = 0
        accurate_count = 0
        rounds_list = []

        for sid, s in samples.items():
            actual_rounds = s["query_count"]
            effective_rounds = min(actual_rounds, cap)
            sim_latency = round(effective_rounds * per_round_latency[sid], 6)

            if cap >= actual_rounds:
                # Full run -- use actual recorded results
                sim_compiled = s["compilation_ratio"] == 1.0
                sim_func_acc = s["functional_accuracy"] >= FUNCTIONAL_ACCURACY_THRESHOLD
            else:
                # Truncated -- conservative: incomplete pattern
                sim_compiled = False
                sim_func_acc = False

            cap_records[sid] = {
                "actual_rounds": actual_rounds,
                "effective_rounds": effective_rounds,
                "simulated_latency": sim_latency,
                "compiled": sim_compiled,
                "functional_accuracy": sim_func_acc,
            }

            latencies.append(sim_latency)
            compiled_count += int(sim_compiled)
            accurate_count += int(sim_func_acc)
            rounds_list.append(effective_rounds)

        n = len(samples)
        summary[str(cap)] = {
            "mean_latency": round(sum(latencies) / n, 2),
            "compilation_rate": round(100.0 * compiled_count / n, 1),
            "functional_accuracy": round(100.0 * accurate_count / n, 1),
            "mean_rounds_used": round(sum(rounds_list) / n, 2),
        }
        raw[str(cap)] = cap_records

    full_summary = {
        "cap_range": cap_range,
        "max_rounds_observed": max_rounds,
        "results": summary,
        "methodology_note": (
            "Latency for each sample under a given cap is estimated by prorating "
            "the recorded total generation time uniformly across the rounds actually "
            "used (per_round_latency = generation_time / query_count), then "
            "multiplying by min(query_count, cap).  For caps below a sample's "
            "actual round count the pattern is treated as truncated: compiled is "
            "set to False and functional accuracy to 0, representing a conservative "
            "lower bound on the quality of an incomplete assembly."
        ),
    }
    return raw, full_summary


def plot(summary: dict, output_path: Path) -> None:
    """Produce the dual-axis bar + line chart."""
    sns.set_style("whitegrid")

    caps = summary["cap_range"]
    results = summary["results"]

    mean_lat = [results[str(c)]["mean_latency"] for c in caps]
    comp_rate = [results[str(c)]["compilation_rate"] for c in caps]
    func_acc = [results[str(c)]["functional_accuracy"] for c in caps]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    # Bars – mean latency
    bar_colour = "#a0cefa"
    bars = ax1.bar(
        caps, mean_lat, width=0.6, color=bar_colour, alpha=0.85,
        label="Mean Latency (s)", zorder=2,
    )
    for bar, val in zip(bars, mean_lat):
        ax1.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
            f"{val:.2f}", ha="center", va="bottom", fontsize=8, color="#333333",
        )

    # Lines – compilation rate & functional accuracy
    comp_colour = "#a31d18"
    acc_colour = "#157323"
    line1, = ax2.plot(
        caps, comp_rate, marker="o", markersize=6, linewidth=2,
        color=comp_colour, label="Compilation Rate (%)", zorder=3,
    )
    line2, = ax2.plot(
        caps, func_acc, marker="o", markersize=6, linewidth=2,
        color=acc_colour, label="Functional Accuracy (%)", zorder=3,
    )

    # Annotate line points
    for x, y in zip(caps, comp_rate):
        ax2.text(x, y + 1.8, f"{y:.1f}", ha="center", va="bottom",
                 fontsize=8, color=comp_colour)
    for x, y in zip(caps, func_acc):
        ax2.text(x, y - 3.5, f"{y:.1f}", ha="center", va="top",
                 fontsize=8, color=acc_colour)

    # Axes labels & ticks
    ax1.set_xlabel(r"Generation Round Cap ($\mathit{query\_count}$)", fontsize=11)
    ax1.set_ylabel("Mean Latency (s)", fontsize=11)
    ax2.set_ylabel("Rate (%)", fontsize=11)
    ax1.set_xticks(caps)
    ax1.xaxis.set_major_locator(mticker.FixedLocator(caps))
    ax2.set_ylim(0, 110)

    # Legend – combine both axes
    handles = [bars, line1, line2]
    labels = [h.get_label() for h in handles]
    ax1.legend(handles, labels, loc="upper left", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved to {output_path}")


def main() -> None:
    if not RESULTS_FILE.exists():
        print(f"ERROR: Results file not found: {RESULTS_FILE}", file=sys.stderr)
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    samples, metadata = load_results(RESULTS_FILE)
    print(f"  {len(samples)} samples loaded.  Rounds used per sample read from 'query_count' field.")

    print("Running cap sweep simulation...")
    raw, summary = simulate(samples)

    # Save JSON outputs
    raw_path = OUTPUT_DIR / "query_count_tradeoff_raw.json"
    summary_path = OUTPUT_DIR / "query_count_tradeoff_summary.json"
    with open(raw_path, "w", encoding="utf-8") as fh:
        json.dump(raw, fh, indent=2)
    print(f"  Raw results saved to {raw_path}")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"  Summary saved to {summary_path}")

    print("Generating plot...")
    plot(summary, OUTPUT_DIR / "query_count_tradeoff.png")

    # Print summary table
    print("\n=== Summary ===")
    print(f"{'Cap':>4} | {'Mean Lat (s)':>12} | {'Comp Rate %':>11} | {'Func Acc %':>10} | {'Mean Rounds':>11}")
    print("-" * 60)
    for cap in summary["cap_range"]:
        r = summary["results"][str(cap)]
        print(
            f"{cap:>4} | {r['mean_latency']:>12.2f} | {r['compilation_rate']:>10.1f}% "
            f"| {r['functional_accuracy']:>9.1f}% | {r['mean_rounds_used']:>11.2f}"
        )


if __name__ == "__main__":
    main()
