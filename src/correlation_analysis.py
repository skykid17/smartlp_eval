"""
Correlation analysis between retrieval metrics (contextual precision, recall, relevancy)
and generation quality metrics (faithfulness, conciseness) across multiple retrieval modes.

This script analyzes 364 samples (182 from hybrid mode + 182 from text mode).
Note: vector mode results do not contain faithfulness/conciseness metrics.
"""

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "data" / "eval" / "output"
OUTPUT_DIR = BASE_DIR / "data" / "eval" / "output" / "correlation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data from all modes
# ---------------------------------------------------------------------------
def load_retrieval_results(mode):
    """Load retrieval results for a given mode (hybrid, text, or vector)."""
    file_path = INPUT_DIR / f"retriever_results_{mode}.json"
    with open(file_path, "r", encoding="utf-8") as fh:
        return json.load(fh)

modes_with_metrics = ["hybrid", "text", "vector"]
all_data = []

for mode in modes_with_metrics:
    results = load_retrieval_results(mode)

    for idx, entry in enumerate(results):
        # Extract scores from nested dicts
        row = {
            "mode": mode,
            "index": idx,
            "input": entry["input"],
            "contextual_precision": entry["contextual_precision"]["score"],
            "contextual_recall": entry["contextual_recall"]["score"],
            "contextual_relevancy": entry["contextual_relevancy"]["score"],
            "faithfulness": entry["faithfulness"]["score"],
            "conciseness": entry["conciseness"],
        }
        all_data.append(row)

df = pd.DataFrame(all_data)
print(f"Loaded {len(df)} samples across {len(modes_with_metrics)} modes")
print(f"Modes: {', '.join(modes_with_metrics)}")
print(f"Breakdown: {df['mode'].value_counts().to_dict()}")
print(f"\nDataset statistics:")
print(df[["contextual_precision", "contextual_recall", "contextual_relevancy",
          "faithfulness", "conciseness"]].describe())

# Save combined dataset
df.to_csv(OUTPUT_DIR / "retrieval_correlation_dataset.csv", index=False)
print(f"\nSaved combined dataset to retrieval_correlation_dataset.csv")

# ---------------------------------------------------------------------------
# Correlation Analysis
# ---------------------------------------------------------------------------
def fmt_p(p: float) -> str:
    """Format p-value for display."""
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}"

def interpret(r: float, p: float) -> str:
    """Plain-English interpretation."""
    sig = "significant" if p < 0.05 else "not significant"
    abs_r = abs(r)
    if abs_r >= 0.6:
        strength = "strong"
    elif abs_r >= 0.4:
        strength = "moderate"
    elif abs_r >= 0.2:
        strength = "weak"
    else:
        strength = "negligible"
    direction = "positive" if r >= 0 else "negative"
    return f"{strength} {direction} correlation, {sig}"

def correlate(x, y, label_x, label_y):
    """Calculate Spearman and Pearson correlations."""
    sp_r, sp_p = stats.spearmanr(x, y)
    pe_r, pe_p = stats.pearsonr(x, y)
    return {
        "x_metric": label_x,
        "y_metric": label_y,
        "n": len(x),
        "spearman_r": round(sp_r, 4),
        "spearman_p": sp_p,
        "spearman_p_fmt": fmt_p(sp_p),
        "spearman_interpretation": interpret(sp_r, sp_p),
        "pearson_r": round(pe_r, 4),
        "pearson_p": pe_p,
        "pearson_p_fmt": fmt_p(pe_p),
        "pearson_interpretation": interpret(pe_r, pe_p),
    }

# Define the 6 correlation pairs (3 retrieval metrics × 2 generation metrics)
retrieval_metrics = ["contextual_precision", "contextual_recall", "contextual_relevancy"]
generation_metrics = ["faithfulness", "conciseness"]

correlations = []
for ret_metric in retrieval_metrics:
    for gen_metric in generation_metrics:
        x = df[ret_metric].values
        y = df[gen_metric].values
        corr = correlate(x, y, ret_metric, gen_metric)
        correlations.append(corr)

        print(f"\n{ret_metric} vs {gen_metric}")
        print(f"  Spearman r = {corr['spearman_r']}, {corr['spearman_p_fmt']}")
        print(f"  Pearson  r = {corr['pearson_r']}, {corr['pearson_p_fmt']}")
        print(f"  Interpretation: {corr['spearman_interpretation']}")

# ---------------------------------------------------------------------------
# Create Correlation Table
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("CORRELATION MATRIX (Spearman r)")
print("=" * 80)

# Create a matrix for visualization
correlation_matrix = pd.DataFrame(
    index=retrieval_metrics,
    columns=generation_metrics,
    dtype=float
)

p_value_matrix = pd.DataFrame(
    index=retrieval_metrics,
    columns=generation_metrics,
    dtype=float
)

for corr in correlations:
    correlation_matrix.loc[corr['x_metric'], corr['y_metric']] = corr['spearman_r']
    p_value_matrix.loc[corr['x_metric'], corr['y_metric']] = corr['spearman_p']

print(f"\nn = {len(df)} (hybrid + text modes)")
print("\nSpearman Correlation Coefficients:")
print(correlation_matrix.to_string())
print("\np-values:")
print(p_value_matrix.to_string())

# Create formatted table with significance markers
formatted_table = correlation_matrix.astype(object)
for ret in retrieval_metrics:
    for gen in generation_metrics:
        r = correlation_matrix.loc[ret, gen]
        p = p_value_matrix.loc[ret, gen]
        if p < 0.001:
            sig = "***"
        elif p < 0.01:
            sig = "**"
        elif p < 0.05:
            sig = "*"
        else:
            sig = ""
        formatted_table.loc[ret, gen] = f"{r:.3f}{sig}"

print("\nFormatted Table (*** p<0.001, ** p<0.01, * p<0.05):")
print(formatted_table.to_string())

# Save results
correlation_results = {
    "note": "Correlation between retrieval metrics and generation quality metrics",
    "n_samples": len(df),
    "modes_analyzed": modes_with_metrics,
    "retrieval_metrics": retrieval_metrics,
    "generation_metrics": generation_metrics,
    "correlations": correlations,
    "correlation_matrix": correlation_matrix.to_dict(),
    "p_value_matrix": p_value_matrix.to_dict(),
}

with open(OUTPUT_DIR / "correlation_results_retrieval.json", "w", encoding="utf-8") as fh:
    json.dump(correlation_results, fh, indent=2)
print(f"\nSaved correlation_results_retrieval.json")

# ---------------------------------------------------------------------------
# Scatter plots with OLS regression
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", font_scale=1.0)
plt.rcParams.update({"font.size": 10})

def make_scatter(x, y, xlabel, ylabel, title, filename, corr_dict):
    """Create scatter plot with OLS regression line."""
    fig, ax = plt.subplots(figsize=(6, 5))

    # Data points
    ax.scatter(x, y, alpha=0.5, edgecolors="steelblue", facecolors="lightskyblue",
               linewidths=0.6, s=40, zorder=3)

    # OLS regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    x_line = np.linspace(min(x), max(x), 200)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, color="firebrick", linewidth=2, zorder=4,
            label=f"OLS: y = {slope:.3f}x + {intercept:.3f}")

    # Annotation with correlation stats
    r_val = corr_dict["spearman_r"]
    p_str = corr_dict["spearman_p_fmt"]
    annotation = f"$r_s = {r_val:.3f}$\n{p_str}\nn = {corr_dict['n']}"
    ax.text(0.05, 0.95, annotation, transform=ax.transAxes,
            fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="grey", alpha=0.85))

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11, pad=10)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = OUTPUT_DIR / filename
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {filename}")

# Generate all 6 scatter plots
print("\nGenerating scatter plots...")
for corr in correlations:
    ret_metric = corr['x_metric']
    gen_metric = corr['y_metric']

    x = df[ret_metric].values
    y = df[gen_metric].values

    # Create readable labels
    xlabel_map = {
        "contextual_precision": "Contextual Precision",
        "contextual_recall": "Contextual Recall",
        "contextual_relevancy": "Contextual Relevancy"
    }
    ylabel_map = {
        "faithfulness": "Faithfulness",
        "conciseness": "Conciseness"
    }

    xlabel = xlabel_map[ret_metric]
    ylabel = ylabel_map[gen_metric]
    title = f"{xlabel} vs {ylabel}\n(n = {len(df)}, hybrid + text modes)"
    filename = f"correlation_{ret_metric}_vs_{gen_metric}.png"

    make_scatter(x, y, xlabel, ylabel, title, filename, corr)

print("\n" + "=" * 80)
print(f"Analysis complete! All outputs saved to: {OUTPUT_DIR}")
print("=" * 80)
