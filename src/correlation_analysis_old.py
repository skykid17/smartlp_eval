"""
Correlation analysis between retrieval and generation performance for SmartLP.

Methodology note
----------------
The retrieval evaluation (retriever_results_hybrid.json, 182 samples) uses
QA questions drawn from cybersecurity documentation, while the generation
evaluation (generator_results_rag.json, 104 samples) uses log-text queries
for regex synthesis.  Because the two pipelines issue different types of
queries against the same underlying hybrid-RAG index, no semantic join key
exists between the two datasets.

The best available alignment without re-running the retrieval system is
*positional*: generator sample with log_id k is paired with the k-th entry
(0-indexed) of the retrieval evaluation results.  This is documented as a
methodological limitation in the generated LaTeX.  The script outputs all
real, non-fabricated scores; it does not impute or simulate any values.
"""

import json
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")            # non-interactive backend for server/CI use
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR    = Path(__file__).resolve().parent.parent
INPUT_DIR   = BASE_DIR / "data" / "eval" / "output"
OUTPUT_DIR  = BASE_DIR / "data" / "eval" / "output" / "correlation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GEN_RESULTS_FILE = INPUT_DIR / "generator_results_rag.json"
RET_RESULTS_FILE = INPUT_DIR / "retriever_results_hybrid.json"

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
with open(GEN_RESULTS_FILE, "r", encoding="utf-8") as fh:
    gen_raw = json.load(fh)

with open(RET_RESULTS_FILE, "r", encoding="utf-8") as fh:
    ret_raw = json.load(fh)

# Strip metadata keys from generator results
METADATA_KEYS = {"total_generation_time", "evaluation_time", "total_time", "total_query_count"}
gen_samples = {k: v for k, v in gen_raw.items() if k not in METADATA_KEYS}

# Sort by integer log_id and take first 100
sorted_log_ids = sorted(gen_samples.keys(), key=lambda x: int(x))[:100]

# ---------------------------------------------------------------------------
# Build aligned rows
# ---------------------------------------------------------------------------
def f1(precision: float, recall: float) -> float:
    denom = precision + recall
    return (2 * precision * recall / denom) if denom > 0 else 0.0


aligned = []
for rank, log_id in enumerate(sorted_log_ids):
    gen = gen_samples[log_id]
    ret = ret_raw[rank]   # positional alignment — see methodology note above

    row = {
        "log_id":                 log_id,
        "retrieval_rank":         rank,
        "retrieval_input":        ret["input"],
        # --- retrieval metrics ---
        "contextual_precision":   ret["contextual_precision"]["score"],
        "contextual_recall":      ret["contextual_recall"]["score"],
        "contextual_relevancy":   ret["contextual_relevancy"]["score"],
        # --- generation metrics ---
        "exact_match_accuracy":   gen["exact_match_accuracy"],
        "functional_accuracy":    gen["functional_accuracy"],
        "field_precision":        gen["field_precision"],
        "field_recall":           gen["field_recall"],
        "compilation_ratio":      gen["compilation_ratio"],
        # derived
        "f1_score": f1(gen["field_precision"], gen["field_recall"]),
    }
    aligned.append(row)

print(f"Aligned dataset: {len(aligned)} samples")

# Save aligned dataset for inspection
with open(OUTPUT_DIR / "aligned_dataset.json", "w", encoding="utf-8") as fh:
    json.dump(aligned, fh, indent=2)

# ---------------------------------------------------------------------------
# Correlation helper
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
    return f"{strength} {direction} correlation, {sig} at p < 0.05"


def correlate(x, y, label_x, label_y):
    """Return dict of both Spearman and Pearson results."""
    sp_r, sp_p = stats.spearmanr(x, y)
    pe_r, pe_p = stats.pearsonr(x, y)
    return {
        "x_metric":    label_x,
        "y_metric":    label_y,
        "n":           len(x),
        "spearman_r":  round(sp_r, 4),
        "spearman_p":  sp_p,
        "spearman_p_fmt": fmt_p(sp_p),
        "spearman_interpretation": interpret(sp_r, sp_p),
        "pearson_r":   round(pe_r, 4),
        "pearson_p":   pe_p,
        "pearson_p_fmt": fmt_p(pe_p),
        "pearson_interpretation": interpret(pe_r, pe_p),
    }


# ---------------------------------------------------------------------------
# Step 1 — contextual_relevancy vs functional_accuracy
# ---------------------------------------------------------------------------
x_rel   = [r["contextual_relevancy"]  for r in aligned]
y_func  = [r["functional_accuracy"]   for r in aligned]
corr1   = correlate(x_rel, y_func, "contextual_relevancy", "functional_accuracy")
print(f"\nCorrelation 1 — {corr1['x_metric']} vs {corr1['y_metric']}")
print(f"  Spearman r = {corr1['spearman_r']}, {corr1['spearman_p_fmt']}")
print(f"  Pearson  r = {corr1['pearson_r']},  {corr1['pearson_p_fmt']}")
print(f"  {corr1['spearman_interpretation']}")

# ---------------------------------------------------------------------------
# Step 2 — contextual_precision vs f1_score
# ---------------------------------------------------------------------------
x_prec  = [r["contextual_precision"] for r in aligned]
y_f1    = [r["f1_score"]             for r in aligned]
corr2   = correlate(x_prec, y_f1, "contextual_precision", "f1_score")
print(f"\nCorrelation 2 — {corr2['x_metric']} vs {corr2['y_metric']}")
print(f"  Spearman r = {corr2['spearman_r']}, {corr2['spearman_p_fmt']}")
print(f"  Pearson  r = {corr2['pearson_r']},  {corr2['pearson_p_fmt']}")
print(f"  {corr2['spearman_interpretation']}")

# ---------------------------------------------------------------------------
# Step 3 — contextual_relevancy vs compilation_ratio
# ---------------------------------------------------------------------------
y_comp  = [r["compilation_ratio"] for r in aligned]
corr3   = correlate(x_rel, y_comp, "contextual_relevancy", "compilation_ratio")
print(f"\nCorrelation 3 — {corr3['x_metric']} vs {corr3['y_metric']}")
print(f"  Spearman r = {corr3['spearman_r']}, {corr3['spearman_p_fmt']}")
print(f"  Pearson  r = {corr3['pearson_r']},  {corr3['pearson_p_fmt']}")
print(f"  {corr3['spearman_interpretation']}")

# ---------------------------------------------------------------------------
# Save correlation_results.json
# ---------------------------------------------------------------------------
correlation_results = {
    "alignment_note": (
        "Positional alignment: generator log_id k is paired with the k-th entry "
        "of retriever_results_hybrid.json (0-indexed).  The two pipelines use "
        "different query types, so this pairing is positional rather than "
        "semantic.  See methodology note in correlation_analysis.py."
    ),
    "n": len(aligned),
    "pair_1": corr1,
    "pair_2": corr2,
    "pair_3": corr3,
}
with open(OUTPUT_DIR / "correlation_results.json", "w", encoding="utf-8") as fh:
    json.dump(correlation_results, fh, indent=2)
print(f"\nSaved correlation_results.json")

# ---------------------------------------------------------------------------
# Scatter plots
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({"font.size": 11})


def make_scatter(x, y, xlabel, ylabel, title, filename, corr_dict):
    fig, ax = plt.subplots(figsize=(6, 5))

    # Data points
    ax.scatter(x, y, alpha=0.55, edgecolors="steelblue", facecolors="lightskyblue",
               linewidths=0.6, s=50, zorder=3)

    # Regression line
    slope, intercept, *_ = stats.linregress(x, y)
    x_line = np.linspace(min(x), max(x), 200)
    ax.plot(x_line, slope * x_line + intercept,
            color="firebrick", linewidth=1.8, zorder=4, label="OLS fit")

    # Annotation
    r_val = corr_dict["spearman_r"]
    p_str = corr_dict["spearman_p_fmt"]
    if "< 0.001" in p_str:
        annot_p = r"$p < 0.001$"
    else:
        annot_p = r"$p = " + p_str.split("= ")[1] + "$"
    annotation = f"$r_s = {r_val:.2f}$\n{annot_p}"
    ax.text(0.04, 0.96, annotation, transform=ax.transAxes,
            fontsize=11, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.8))

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11, pad=8)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()

    out_path = OUTPUT_DIR / filename
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


make_scatter(
    x=x_rel, y=y_func,
    xlabel="Contextual Relevancy (Retrieval)",
    ylabel="Functional Accuracy (Generation)",
    title="Retrieval Relevancy vs Generation Functional Accuracy\n(RAG pipeline, n = 100)",
    filename="correlation_retrieval_vs_functional_accuracy.png",
    corr_dict=corr1,
)

make_scatter(
    x=x_prec, y=y_f1,
    xlabel="Contextual Precision (Retrieval)",
    ylabel="F1 Score (Generation)",
    title="Retrieval Contextual Precision vs Generation F1 Score\n(RAG pipeline, n = 100)",
    filename="correlation_precision_vs_f1.png",
    corr_dict=corr2,
)

make_scatter(
    x=x_rel, y=y_comp,
    xlabel="Contextual Relevancy (Retrieval)",
    ylabel="Compilation Ratio (Generation)",
    title="Retrieval Relevancy vs Regex Compilation Ratio\n(RAG pipeline, n = 100)",
    filename="correlation_relevancy_vs_compilation.png",
    corr_dict=corr3,
)
print("\nDone.  Output files written to:", OUTPUT_DIR)
