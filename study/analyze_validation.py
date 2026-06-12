#!/usr/bin/env python
"""
study/analyze_validation.py
============================
Validates the GPT-4o LLM-as-judge against 3 human expert raters for the
UQ BIT information assistant study. Addresses Limitation #1 in PAPER_CONTEXT.md.

Usage (from repo root):
    python study/analyze_validation.py

Outputs:
    study/validation_report.md   -- full narrative report with tables
    study/agreement_stats.csv    -- machine-readable flat stats table
    study/figures/               -- four PNG figures

Dependencies (must be installed):
    pip install pingouin krippendorff matplotlib seaborn
"""
import argparse
import pathlib
import warnings

import krippendorff
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
from scipy import stats
from sklearn.metrics import cohen_kappa_score

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── Dimension mapping ─────────────────────────────────────────────────────────
DIM_CODES   = ["FACT", "REL", "CONC", "NOHAL"]
DIM_LABELS  = {
    "FACT":  "Factual accuracy",
    "REL":   "Relevance",
    "CONC":  "Conciseness",
    "NOHAL": "No hallucination",
}
GEVAL_COLS  = {
    "FACT":  "factual_accuracy",
    "REL":   "relevance",
    "CONC":  "conciseness",
    "NOHAL": "no_hallucination",
}
# Composite weights (match G-Eval, PAPER_CONTEXT.md §7.3)
COMP_WEIGHTS = {"FACT": 0.55, "REL": 0.25, "CONC": 0.10, "NOHAL": 0.10}


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Human validation analysis.")
    p.add_argument("--tidy",       default="study/responses_tidy.csv")
    p.add_argument("--key",        default="study/answer_key.csv")
    p.add_argument("--out-report", default="study/validation_report.md")
    p.add_argument("--out-stats",  default="study/agreement_stats.csv")
    p.add_argument("--figdir",     default="study/figures")
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────
def safe_spearman(x, y):
    """Return (rho, pval) or (nan, nan) on insufficient data / zero variance."""
    mask = ~(np.isnan(x) | np.isnan(y))
    x2, y2 = x[mask], y[mask]
    if len(x2) < 4 or np.std(x2) == 0 or np.std(y2) == 0:
        return np.nan, np.nan
    r, p = stats.spearmanr(x2, y2)
    return float(r), float(p)


def safe_pearson(x, y):
    mask = ~(np.isnan(x) | np.isnan(y))
    x2, y2 = x[mask], y[mask]
    if len(x2) < 4 or np.std(x2) == 0 or np.std(y2) == 0:
        return np.nan, np.nan
    r, p = stats.pearsonr(x2, y2)
    return float(r), float(p)


def safe_qwk(x, y):
    """Quadratic-weighted kappa on rounded integer scores 0-5."""
    mask = ~(np.isnan(x) | np.isnan(y))
    x2 = np.round(x[mask]).astype(int).clip(0, 5)
    y2 = np.round(y[mask]).astype(int).clip(0, 5)
    if len(x2) < 4:
        return np.nan
    try:
        return float(cohen_kappa_score(x2, y2, weights="quadratic"))
    except Exception:
        return np.nan


def safe_wilcoxon(diff):
    """Wilcoxon signed-rank test on diff = a - b; return (stat, pval)."""
    diff = diff[~np.isnan(diff)]
    if len(diff) < 10 or np.all(diff == 0):
        return np.nan, np.nan
    try:
        res = stats.wilcoxon(diff, alternative="two-sided")
        return float(res.statistic), float(res.pvalue)
    except Exception:
        return np.nan, np.nan


def safe_ttest(diff):
    diff = diff[~np.isnan(diff)]
    if len(diff) < 4:
        return np.nan, np.nan
    res = stats.ttest_1samp(diff, 0)
    return float(res.statistic), float(res.pvalue)


def ci95(arr):
    """95% CI of the mean (normal approximation)."""
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2:
        return np.nan, np.nan
    m = arr.mean()
    se = arr.std(ddof=1) / np.sqrt(len(arr))
    return float(m - 1.96 * se), float(m + 1.96 * se)


def kendalls_w(ratings_matrix):
    """
    Kendall's W from an (n_subjects x n_raters) matrix.
    Returns W in [0,1], or nan on degenerate input.
    """
    mat = np.array(ratings_matrix, dtype=float)
    # Drop rows with any nan
    mat = mat[~np.isnan(mat).any(axis=1)]
    n, k = mat.shape
    if n < 2 or k < 2:
        return np.nan
    ranks = np.apply_along_axis(stats.rankdata, 0, mat)
    Ri = ranks.sum(axis=1)
    S = np.sum((Ri - Ri.mean()) ** 2)
    W = 12 * S / (k ** 2 * (n ** 3 - n))
    return float(np.clip(W, 0, 1))


def fmt(v, decimals=3):
    if np.isnan(v):
        return "N/A"
    return f"{v:.{decimals}f}"


def sig_stars(p):
    if np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


# ── Build per-dimension pivot tables ─────────────────────────────────────────
def build_pivots(tidy, key):
    """
    Returns dict: dim_code -> DataFrame with columns
        item_id, rater_1, rater_2, rater_3, geval, human_consensus
    Rater columns are named by email.
    """
    pivots = {}
    raters = sorted(tidy["rater"].unique())

    # item_id -> geval score (from key, each dim)
    key_idx = key.set_index("item_id")

    for dim in DIM_CODES:
        sub = tidy[tidy["dimension"] == dim][["rater", "item_id", "score"]].copy()
        pivot = sub.pivot(index="item_id", columns="rater", values="score")
        pivot.columns.name = None
        pivot = pivot.reset_index()

        # Join G-Eval score for this dimension
        geval_col = GEVAL_COLS[dim]
        pivot["geval"] = pivot["item_id"].map(key_idx[geval_col])

        # Human consensus = nan-aware mean of rater columns
        rater_cols = [c for c in pivot.columns if c in raters]
        pivot["human_consensus"] = pivot[rater_cols].mean(axis=1, skipna=True)

        pivots[dim] = pivot

    return pivots, raters


# ── Composite scores per item ─────────────────────────────────────────────────
def build_composites(pivots, key):
    """
    Returns DataFrame: item_id, human_composite, geval_composite,
    plus question_slot, model, state, variant, condition.
    """
    items = pivots["FACT"]["item_id"].values
    human_comp = np.zeros(len(items))
    for dim, w in COMP_WEIGHTS.items():
        hc = pivots[dim].set_index("item_id").loc[items, "human_consensus"].values
        human_comp += w * hc
    human_comp /= 5.0   # normalise to [0,1]

    key_idx = key.set_index("item_id")
    geval_comp = key_idx.loc[items, "composite_score"].values.astype(float)

    df = pd.DataFrame({
        "item_id":         items,
        "human_composite": human_comp,
        "geval_composite": geval_comp,
    })
    for col in ["question_slot", "model", "state", "variant", "condition"]:
        df[col] = key_idx.loc[items, col].values
    return df


# ── Inter-rater reliability ───────────────────────────────────────────────────
def interrater_stats(pivot, raters):
    """
    pivot: DataFrame with item_id + rater columns + geval + human_consensus
    Returns dict of metrics.
    """
    rater_cols = [c for c in pivot.columns if c in raters]
    mat = pivot[rater_cols].values.astype(float)  # (50, 3)

    result = {}

    # ICC(2,k) via pingouin
    long = pivot[["item_id"] + rater_cols].melt(
        id_vars="item_id", var_name="rater", value_name="score"
    ).dropna()
    if len(long) >= 6:
        icc = pg.intraclass_corr(
            data=long, targets="item_id", raters="rater", ratings="score", nan_policy="omit"
        ).set_index("Type")
        # ICC(A,k) = two-way random, absolute agreement, average measures (ICC2k equivalent)
        # Pingouin type labels: ICC(A,k), ICC(C,k), ICC(A,1), ICC(C,1), ICC(1,1), ICC(1,k)
        for type_label in ["ICC(A,k)", "ICC2k", "ICC2"]:
            if type_label in icc.index:
                r = icc.loc[type_label]
                result["ICC2k"]       = float(r["ICC"])
                ci = r["CI95"]  # list [lo, hi]
                result["ICC2k_ci_lo"] = float(ci[0])
                result["ICC2k_ci_hi"] = float(ci[1])
                break
        else:
            result["ICC2k"] = result["ICC2k_ci_lo"] = result["ICC2k_ci_hi"] = np.nan
    else:
        result["ICC2k"] = result["ICC2k_ci_lo"] = result["ICC2k_ci_hi"] = np.nan

    # Krippendorff's alpha (ordinal)
    try:
        result["kripp_alpha"] = float(krippendorff.alpha(
            mat.T, level_of_measurement="ordinal"
        ))
    except Exception:
        result["kripp_alpha"] = np.nan

    # Kendall's W
    result["kendalls_W"] = kendalls_w(mat)

    # Pairwise Spearman
    pairs = [(rater_cols[i], rater_cols[j])
             for i in range(len(rater_cols))
             for j in range(i + 1, len(rater_cols))]
    pw_rs = []
    for a, b in pairs:
        r, _ = safe_spearman(pivot[a].values, pivot[b].values)
        pw_rs.append(r)
    result["mean_pairwise_spearman"] = float(np.nanmean(pw_rs))
    result["pairwise_spearmans"]     = pw_rs  # list of 3

    # Exact agreement and within ±1
    exact_list, near_list = [], []
    for i in range(len(rater_cols)):
        for j in range(i + 1, len(rater_cols)):
            a = pivot[rater_cols[i]].values
            b = pivot[rater_cols[j]].values
            mask = ~(np.isnan(a) | np.isnan(b))
            if mask.sum() == 0:
                continue
            exact_list.append((a[mask] == b[mask]).mean())
            near_list.append((np.abs(a[mask] - b[mask]) <= 1).mean())
    result["exact_agreement_pct"] = float(np.nanmean(exact_list)) * 100
    result["within1_agreement_pct"] = float(np.nanmean(near_list)) * 100

    return result


# ── Human vs G-Eval agreement ─────────────────────────────────────────────────
def human_vs_geval_stats(pivot, raters):
    hc = pivot["human_consensus"].values.astype(float)
    ge = pivot["geval"].values.astype(float)

    result = {}
    r, p = safe_spearman(hc, ge)
    result["spearman_rho"], result["spearman_p"] = r, p
    r, p = safe_pearson(hc, ge)
    result["pearson_r"], result["pearson_p"] = r, p
    result["qwk"] = safe_qwk(hc, ge)

    diff = ge - hc
    mask = ~np.isnan(diff)
    result["bias_mean"]   = float(np.nanmean(diff))
    result["bias_sd"]     = float(np.nanstd(diff, ddof=1))
    lo, hi = ci95(diff)
    result["bias_ci_lo"], result["bias_ci_hi"] = lo, hi
    w, wp = safe_wilcoxon(diff)
    result["wilcoxon_stat"], result["wilcoxon_p"] = w, wp
    t, tp = safe_ttest(diff)
    result["ttest_t"], result["ttest_p"] = t, tp
    result["n"] = int(mask.sum())

    # Per-rater vs G-Eval (to show G-Eval within the human-human band)
    rater_cols = [c for c in pivot.columns if c in raters]
    per_rater = {}
    for rc in rater_cols:
        rv = pivot[rc].values.astype(float)
        rho, _ = safe_spearman(rv, ge)
        per_rater[rc] = rho
    result["per_rater_spearman"] = per_rater

    return result


# ── Per-question rank agreement ───────────────────────────────────────────────
def rank_agreement_per_question(composites):
    """
    For each question (5 answers), Spearman + Kendall tau between
    human_composite and geval_composite ranks.
    Returns DataFrame with columns: question_slot, spearman, kendall_tau.
    """
    rows = []
    for slot, grp in composites.groupby("question_slot"):
        h = grp["human_composite"].values.astype(float)
        g = grp["geval_composite"].values.astype(float)
        if len(h) < 2:
            continue
        r, _ = safe_spearman(h, g)
        tau, _ = stats.kendalltau(h, g, nan_policy="omit")
        rows.append({"question_slot": int(slot), "spearman": r, "kendall_tau": float(tau)})
    return pd.DataFrame(rows)


# ── Descriptive table ─────────────────────────────────────────────────────────
def descriptive_table(pivots, composites):
    rows = []
    for dim in DIM_CODES:
        hc = pivots[dim]["human_consensus"].values.astype(float)
        ge = pivots[dim]["geval"].values.astype(float)
        rows.append({
            "Dimension": DIM_LABELS[dim],
            "Human mean": fmt(np.nanmean(hc)),
            "G-Eval mean": fmt(np.nanmean(ge)),
            "Gap (G-Eval − Human)": fmt(np.nanmean(ge) - np.nanmean(hc)),
        })
    # Composite
    h = composites["human_composite"].values.astype(float)
    g = composites["geval_composite"].values.astype(float)
    rows.append({
        "Dimension": "Composite (0–1)",
        "Human mean": fmt(np.nanmean(h)),
        "G-Eval mean": fmt(np.nanmean(g)),
        "Gap (G-Eval − Human)": fmt(np.nanmean(g) - np.nanmean(h)),
    })
    return pd.DataFrame(rows)


# ── Condition-level comparison ────────────────────────────────────────────────
def condition_table(composites):
    rows = []
    for (model, state, variant), grp in composites.groupby(["model", "state", "variant"]):
        rows.append({
            "Model":           model,
            "State":           state,
            "Variant":         variant,
            "n":               len(grp),
            "Human composite": fmt(grp["human_composite"].mean()),
            "G-Eval composite":fmt(grp["geval_composite"].mean()),
        })
    return pd.DataFrame(rows).sort_values("G-Eval composite", ascending=False)


# ── Figures ───────────────────────────────────────────────────────────────────
def make_figures(pivots, composites, figdir, raters):
    figdir = pathlib.Path(figdir)
    figdir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.05)

    # 1. Per-dimension scatter: human consensus vs G-Eval
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    axes = axes.flatten()
    for i, dim in enumerate(DIM_CODES):
        ax = axes[i]
        hc = pivots[dim]["human_consensus"].values.astype(float)
        ge = pivots[dim]["geval"].values.astype(float)
        mask = ~(np.isnan(hc) | np.isnan(ge))
        ax.scatter(ge[mask], hc[mask], alpha=0.6, edgecolors="steelblue",
                   facecolors="lightblue", s=50, linewidths=0.8)
        ax.plot([0, 5], [0, 5], "k--", lw=1, label="y = x")
        # Regression line
        if mask.sum() > 3:
            slope, intercept, _, _, _ = stats.linregress(ge[mask], hc[mask])
            xr = np.linspace(ge[mask].min(), ge[mask].max(), 100)
            ax.plot(xr, slope * xr + intercept, "r-", lw=1.5, alpha=0.7, label="OLS")
        rho, _ = safe_spearman(hc, ge)
        ax.set_title(f"{DIM_LABELS[dim]}\n(ρ = {fmt(rho, 2)})", fontsize=11)
        ax.set_xlabel("G-Eval score (0–5)")
        ax.set_ylabel("Human consensus (0–5)")
        ax.set_xlim(-0.2, 5.2)
        ax.set_ylim(-0.2, 5.2)
        ax.legend(fontsize=8)
    fig.suptitle("Human Consensus vs G-Eval: Per-Dimension Scores", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(figdir / "scatter_human_vs_geval.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 2. Composite scatter
    fig, ax = plt.subplots(figsize=(6, 5.5))
    h = composites["human_composite"].values.astype(float)
    g = composites["geval_composite"].values.astype(float)
    mask = ~(np.isnan(h) | np.isnan(g))
    ax.scatter(g[mask], h[mask], alpha=0.65, edgecolors="darkblue",
               facecolors="cornflowerblue", s=60, linewidths=0.8)
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="y = x")
    if mask.sum() > 3:
        slope, intercept, _, _, _ = stats.linregress(g[mask], h[mask])
        xr = np.linspace(g[mask].min(), g[mask].max(), 100)
        ax.plot(xr, slope * xr + intercept, "r-", lw=1.8, alpha=0.75, label="OLS")
    rho, _ = safe_spearman(h, g)
    r, _ = safe_pearson(h, g)
    ax.set_title(f"Human vs G-Eval Composite Score\nSpearman ρ = {fmt(rho, 3)}, Pearson r = {fmt(r, 3)}", fontsize=12)
    ax.set_xlabel("G-Eval composite (0–1)")
    ax.set_ylabel("Human composite (0–1)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(figdir / "composite_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 3. Bias bar chart (G-Eval − human) per dimension
    bias_means, bias_lo, bias_hi, dim_names = [], [], [], []
    for dim in DIM_CODES:
        hc = pivots[dim]["human_consensus"].values.astype(float)
        ge = pivots[dim]["geval"].values.astype(float)
        diff = ge - hc
        bm = np.nanmean(diff)
        lo, hi = ci95(diff)
        bias_means.append(bm)
        bias_lo.append(bm - lo)
        bias_hi.append(hi - bm)
        dim_names.append(DIM_LABELS[dim])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["#d73027" if b > 0 else "#4575b4" for b in bias_means]
    bars = ax.bar(dim_names, bias_means, color=colors, alpha=0.8, edgecolor="white", linewidth=1.2)
    ax.errorbar(dim_names, bias_means,
                yerr=[bias_lo, bias_hi], fmt="none", color="black", capsize=5, linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.9, linestyle="--")
    ax.set_ylabel("Mean (G-Eval − Human) ± 95% CI")
    ax.set_title("Systematic Bias: G-Eval vs Human Experts per Dimension", fontsize=12)
    ax.set_xticks(range(len(dim_names)))
    ax.set_xticklabels(dim_names, rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(figdir / "bias_by_dimension.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 4. Rater × rater Spearman heatmap (including G-Eval as 4th column/row)
    rater_labels = [r.split("@")[0] for r in raters] + ["G-Eval"]
    n = len(rater_labels)
    mat = np.full((n, n), np.nan)
    np.fill_diagonal(mat, 1.0)

    # Build score arrays per rater (across all dims, 200 items each)
    # Use pooled item×dim vector
    all_items = pivots["FACT"]["item_id"].values
    rater_cols_ordered = raters  # email order

    # Build (200,) vectors per source
    def get_vec(source_label):
        parts = []
        for dim in DIM_CODES:
            piv = pivots[dim].set_index("item_id")
            if source_label == "geval":
                parts.append(piv.loc[all_items, "geval"].values.astype(float))
            else:
                parts.append(piv.loc[all_items, source_label].values.astype(float))
        return np.concatenate(parts)

    vecs = [get_vec(r) for r in rater_cols_ordered] + [get_vec("geval")]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            r, _ = safe_spearman(vecs[i], vecs[j])
            mat[i, j] = r

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    mask_diag = np.zeros_like(mat, dtype=bool)
    np.fill_diagonal(mask_diag, True)
    ann_mat = pd.DataFrame(mat, index=rater_labels, columns=rater_labels)
    sns.heatmap(ann_mat, annot=True, fmt=".2f", cmap="RdYlGn",
                vmin=0.3, vmax=1.0, linewidths=0.5,
                cbar_kws={"label": "Spearman ρ"}, ax=ax)
    ax.set_title("Inter-Rater Spearman ρ\n(pooled across all dimensions)", fontsize=11)
    plt.tight_layout()
    plt.savefig(figdir / "interrater_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()


# ── Collect all stats into a flat CSV ─────────────────────────────────────────
def collect_stats_csv(irr_by_dim, hvg_by_dim, irr_all, hvg_all,
                      rank_df, composites, raters):
    rows = []

    def add(metric, dim, value, ci_lo=np.nan, ci_hi=np.nan, p=np.nan):
        rows.append({"metric": metric, "dimension": dim,
                     "value": round(float(value), 6) if not np.isnan(value) else np.nan,
                     "ci_lo": round(float(ci_lo), 6) if not np.isnan(ci_lo) else np.nan,
                     "ci_hi": round(float(ci_hi), 6) if not np.isnan(ci_hi) else np.nan,
                     "p":     round(float(p), 6)     if not np.isnan(p)     else np.nan})

    for dim, s in {**irr_by_dim, "POOLED": irr_all}.items():
        add("ICC2k",                dim, s["ICC2k"],            s["ICC2k_ci_lo"], s["ICC2k_ci_hi"])
        add("Krippendorff_alpha",   dim, s["kripp_alpha"])
        add("Kendalls_W",           dim, s["kendalls_W"])
        add("mean_pairwise_Spearman", dim, s["mean_pairwise_spearman"])
        add("exact_agreement_pct",  dim, s["exact_agreement_pct"])
        add("within1_agreement_pct",dim, s["within1_agreement_pct"])

    for dim, s in {**hvg_by_dim, "COMPOSITE": hvg_all}.items():
        add("HvG_Spearman",   dim, s["spearman_rho"],  p=s["spearman_p"])
        add("HvG_Pearson",    dim, s["pearson_r"],      p=s["pearson_p"])
        add("HvG_QWK",        dim, s["qwk"])
        add("Bias_mean",      dim, s["bias_mean"], s["bias_ci_lo"], s["bias_ci_hi"])
        add("Bias_Wilcoxon_p",dim, s["wilcoxon_p"])
        add("Bias_ttest_p",   dim, s["ttest_p"])

    # Per-question rank agreement
    add("rank_Spearman_mean", "COMPOSITE",
        rank_df["spearman"].mean(), p=np.nan)
    add("rank_KendallTau_mean","COMPOSITE",
        rank_df["kendall_tau"].mean())

    return pd.DataFrame(rows)


# ── Markdown report writer ────────────────────────────────────────────────────
def write_report(path, tidy, key, pivots, composites, raters,
                 irr_by_dim, hvg_by_dim, irr_all, hvg_all,
                 rank_df, desc_df, cond_df):

    rater_names = {r: r.split("@")[0] for r in raters}

    lines = []
    def ln(s=""): lines.append(s)

    ln("# Human Validation Report")
    ln(f"*Generated by `study/analyze_validation.py`*")
    ln()
    ln("## 1. Study Overview")
    ln()
    ln("Three UQ academic experts rated 50 AI-generated answers (10 student questions "
       "× 5 answers each) on four dimensions using a 0–5 integer scale. "
       "The same four dimensions and scale were used by the GPT-4o judge (G-Eval). "
       "This report quantifies (a) inter-rater reliability among the experts and "
       "(b) agreement between the expert consensus and the automated GPT-4o judge.")
    ln()
    ln("**Raters:**")
    for r in raters:
        ln(f"- {r}")
    ln()
    # Completion table
    filled_counts = {r: int(tidy[tidy["rater"] == r]["score"].notna().sum()) for r in raters}
    ln("| Rater | Ratings filled / 200 |")
    ln("|-------|----------------------|")
    for r in raters:
        ln(f"| {r} | {filled_counts[r]} / 200 |")
    ln()
    ln("Missing scores (4 items from one rater) are excluded from aggregations "
       "on a per-metric basis; they do not affect other raters' scores.")
    ln()

    ln("---")
    ln()
    ln("## 2. Descriptive Statistics")
    ln()
    ln("Mean scores across the 50 rated items per dimension:")
    ln()
    ln(desc_df.to_markdown(index=False))
    ln()
    ln("> **Note:** G-Eval dimension scores are on the 0–5 scale (before composite weighting). "
       "Human means are the average of the 3 raters' consensus scores. "
       "Positive gap = G-Eval scores higher than humans.")
    ln()

    ln("---")
    ln()
    ln("## 3. Inter-Rater Reliability")
    ln()
    ln("### 3.1 Per-dimension")
    ln()
    ln("| Dimension | ICC(2,k) [95% CI] | Krippendorff α | Kendall's W | Mean pairwise ρ | Exact agree % | Within±1 % |")
    ln("|-----------|-------------------|----------------|-------------|-----------------|---------------|------------|")
    for dim in DIM_CODES:
        s = irr_by_dim[dim]
        icc_ci = f"[{fmt(s['ICC2k_ci_lo'])}, {fmt(s['ICC2k_ci_hi'])}]"
        ln(f"| {DIM_LABELS[dim]} "
           f"| {fmt(s['ICC2k'])} {icc_ci} "
           f"| {fmt(s['kripp_alpha'])} "
           f"| {fmt(s['kendalls_W'])} "
           f"| {fmt(s['mean_pairwise_spearman'])} "
           f"| {fmt(s['exact_agreement_pct'], 1)}% "
           f"| {fmt(s['within1_agreement_pct'], 1)}% |")
    ln()
    ln("### 3.2 Pooled (all dimensions)")
    ln()
    s = irr_all
    ln(f"- **ICC(2,k):** {fmt(s['ICC2k'])} [95% CI: {fmt(s['ICC2k_ci_lo'])}, {fmt(s['ICC2k_ci_hi'])}]")
    ln(f"- **Krippendorff's α:** {fmt(s['kripp_alpha'])}")
    ln(f"- **Kendall's W:** {fmt(s['kendalls_W'])}")
    ln(f"- **Mean pairwise Spearman:** {fmt(s['mean_pairwise_spearman'])}")
    ln(f"- **Exact agreement:** {fmt(s['exact_agreement_pct'], 1)}%")
    ln(f"- **Within ±1:** {fmt(s['within1_agreement_pct'], 1)}%")
    ln()

    # Interpretation guide
    def icc_interp(icc):
        if np.isnan(icc): return "N/A"
        if icc >= 0.90: return "excellent"
        if icc >= 0.75: return "good"
        if icc >= 0.50: return "moderate"
        return "poor"
    def alpha_interp(a):
        if np.isnan(a): return "N/A"
        if a >= 0.80: return "good"
        if a >= 0.67: return "tentative"
        return "unreliable"

    ln(f"> **Interpretation:** Pooled ICC(2,k) = {fmt(irr_all['ICC2k'])} "
       f"({icc_interp(irr_all['ICC2k'])} reliability, Koo & Mae 2016 guidelines). "
       f"Krippendorff's α = {fmt(irr_all['kripp_alpha'])} "
       f"({alpha_interp(irr_all['kripp_alpha'])}, Krippendorff 2004 threshold ≥0.80). "
       f"Within-±1 agreement is {fmt(irr_all['within1_agreement_pct'], 1)}%, "
       f"indicating raters rarely differed by more than one scale step.")
    ln()

    ln("### 3.3 Pairwise Spearman (per dimension)")
    ln()
    pair_labels = [f"{rater_names[raters[i]]} vs {rater_names[raters[j]]}"
                   for i in range(len(raters)) for j in range(i + 1, len(raters))]
    header = "| Dimension | " + " | ".join(pair_labels) + " |"
    sep    = "|-----------|" + "---------|" * len(pair_labels)
    ln(header); ln(sep)
    for dim in DIM_CODES:
        pw = irr_by_dim[dim]["pairwise_spearmans"]
        ln(f"| {DIM_LABELS[dim]} | " + " | ".join(fmt(r) for r in pw) + " |")
    ln()

    ln("---")
    ln()
    ln("## 4. Human vs G-Eval Agreement")
    ln()
    ln("### 4.1 Per-dimension (human consensus vs G-Eval)")
    ln()
    ln("| Dimension | Spearman ρ (p) | Pearson r (p) | QWK | Bias mean [95% CI] | Wilcoxon p | t-test p |")
    ln("|-----------|----------------|---------------|-----|-------------------|------------|----------|")
    for dim in DIM_CODES:
        s = hvg_by_dim[dim]
        bias_ci = f"[{fmt(s['bias_ci_lo'], 2)}, {fmt(s['bias_ci_hi'], 2)}]"
        ln(f"| {DIM_LABELS[dim]} "
           f"| {fmt(s['spearman_rho'])}{sig_stars(s['spearman_p'])} ({fmt(s['spearman_p'], 4)}) "
           f"| {fmt(s['pearson_r'])}{sig_stars(s['pearson_p'])} ({fmt(s['pearson_p'], 4)}) "
           f"| {fmt(s['qwk'])} "
           f"| {fmt(s['bias_mean'], 3)} {bias_ci} "
           f"| {fmt(s['wilcoxon_p'], 4)}{sig_stars(s['wilcoxon_p'])} "
           f"| {fmt(s['ttest_p'], 4)}{sig_stars(s['ttest_p'])} |")
    ln()
    ln("*Significance: \\* p<0.05, \\*\\* p<0.01, \\*\\*\\* p<0.001. Bias = G-Eval − Human.*")
    ln()

    ln("### 4.2 Composite score agreement")
    ln()
    s = hvg_all
    ln(f"- **Spearman ρ:** {fmt(s['spearman_rho'])} (p = {fmt(s['spearman_p'], 4)}{sig_stars(s['spearman_p'])})")
    ln(f"- **Pearson r:** {fmt(s['pearson_r'])} (p = {fmt(s['pearson_p'], 4)}{sig_stars(s['pearson_p'])})")
    ln(f"- **QWK (0–1 scale, rounded):** {fmt(s['qwk'])}")
    ln(f"- **Composite bias mean (G-Eval − Human):** {fmt(s['bias_mean'], 4)} "
       f"[95% CI: {fmt(s['bias_ci_lo'], 4)}, {fmt(s['bias_ci_hi'], 4)}]")
    ln(f"  Wilcoxon p = {fmt(s['wilcoxon_p'], 4)}{sig_stars(s['wilcoxon_p'])}, "
       f"paired t p = {fmt(s['ttest_p'], 4)}{sig_stars(s['ttest_p'])}")
    ln()

    ln("### 4.3 Individual rater vs G-Eval Spearman (pooled across all dimensions)")
    ln()
    ln("| Rater | Spearman ρ vs G-Eval |")
    ln("|-------|---------------------|")
    per_rater = hvg_by_dim["FACT"]["per_rater_spearman"]  # use pooled version from pooled calc
    # Recalculate pooled per-rater vs geval
    for r in raters:
        parts = []
        for dim in DIM_CODES:
            piv = pivots[dim].set_index("item_id")
            all_items = piv.index.values
            rv = piv.loc[all_items, r].values.astype(float) if r in piv.columns else np.full(50, np.nan)
            ge = piv.loc[all_items, "geval"].values.astype(float)
            parts.append((rv, ge))
        rv_all = np.concatenate([p[0] for p in parts])
        ge_all = np.concatenate([p[1] for p in parts])
        rho, _ = safe_spearman(rv_all, ge_all)
        ln(f"| {r} | {fmt(rho)} |")
    ln()
    hh_mean = irr_all["mean_pairwise_spearman"]
    ln(f"> Human–human mean pairwise ρ (pooled) = **{fmt(hh_mean)}**. "
       "If each rater's ρ against G-Eval exceeds this value, the judge agrees with each "
       "expert *more strongly* than the experts agree with each other — "
       "a key indicator that the automated judge is performing at or above inter-human level.")
    ln()

    ln("---")
    ln()
    ln("## 5. Per-Question Ranking Agreement")
    ln()
    ln("For each of the 10 questions, the 5 answers are ranked by human composite vs "
       "G-Eval composite. Spearman ρ and Kendall τ measure whether the judge orders "
       "answers the same way experts do.")
    ln()
    ln("| Question | Spearman ρ | Kendall τ |")
    ln("|----------|------------|-----------|")
    for _, row in rank_df.iterrows():
        ln(f"| Q{int(row.question_slot):02d} | {fmt(row.spearman)} | {fmt(row.kendall_tau)} |")
    ln()
    mean_rho = rank_df["spearman"].mean()
    mean_tau = rank_df["kendall_tau"].mean()
    sd_rho   = rank_df["spearman"].std(ddof=1)
    sd_tau   = rank_df["kendall_tau"].std(ddof=1)
    ln(f"**Mean Spearman ρ:** {fmt(mean_rho)} ± {fmt(sd_rho)} (SD)  "
       f"**Mean Kendall τ:** {fmt(mean_tau)} ± {fmt(sd_tau)}")
    ln()
    ln("> Ranking agreement is the most practically relevant metric for the paper's "
       "ablation study: it indicates whether G-Eval produces the same *relative* ordering "
       "of configurations as human experts, even if the absolute scores differ.")
    ln()

    ln("---")
    ln()
    ln("## 6. Condition-Level Comparison")
    ln()
    ln("Mean human composite vs mean G-Eval composite per condition "
       "(sorted by G-Eval composite, descending). Shows whether the headline paper "
       "rankings (e.g. Mistral base sysprompt_rag at 0.868) hold under human scoring.")
    ln()
    ln(cond_df.to_markdown(index=False))
    ln()
    ln("> **Note:** Each row represents at most 1 sampled item from this condition "
       "(the study sampled 10 questions × 5 spread-of-quality answers). "
       "The n column shows how many of the 50 sampled answers came from each condition. "
       "Condition-level comparisons are illustrative only; "
       "full ablation rankings require the complete 50-question G-Eval data in PAPER_CONTEXT.md.")
    ln()

    ln("---")
    ln()
    ln("## 7. Figures")
    ln()
    ln("| File | Description |")
    ln("|------|-------------|")
    ln("| `figures/scatter_human_vs_geval.png` | Per-dimension scatter: human consensus vs G-Eval |")
    ln("| `figures/composite_scatter.png` | Composite score scatter with OLS and y=x lines |")
    ln("| `figures/bias_by_dimension.png` | Mean bias (G-Eval − Human) ± 95% CI per dimension |")
    ln("| `figures/interrater_heatmap.png` | Pairwise Spearman heatmap (3 raters + G-Eval) |")
    ln()

    ln("---")
    ln()
    ln("## 8. Implications for Limitation #1 (PAPER_CONTEXT.md)")
    ln()
    ln("*This section summarises what these results mean for the paper's stated limitation "
       "about LLM-as-judge bias. The authors should revise the limitation text accordingly.*")
    ln()
    ln(f"1. **Inter-rater reliability of human experts:** ICC(2,k) = {fmt(irr_all['ICC2k'])} "
       f"({icc_interp(irr_all['ICC2k'])}). The expert panel itself shows "
       f"{'adequate' if irr_all['ICC2k'] >= 0.5 else 'limited'} agreement, "
       "which contextualises the upper bound on expected human–machine agreement.")
    ln()
    ln(f"2. **G-Eval correlation with human consensus:** Pooled composite Spearman ρ = "
       f"{fmt(hvg_all['spearman_rho'])} (p = {fmt(hvg_all['spearman_p'], 4)}). "
       f"{'This indicates substantial agreement.' if abs(hvg_all['spearman_rho']) >= 0.6 else 'Agreement is moderate.'}")
    ln()
    bias_dir = "above" if hvg_all['bias_mean'] > 0 else "below"
    bias_sig  = "statistically significant" if hvg_all['ttest_p'] < 0.05 else "not statistically significant"
    ln(f"3. **Systematic bias:** G-Eval composite scores average {fmt(abs(hvg_all['bias_mean']), 4)} points "
       f"{bias_dir} human consensus ({bias_sig}, paired t p = {fmt(hvg_all['ttest_p'], 4)}). "
       "This suggests " +
       ("the judge inflates scores relative to humans — a known risk for LLM-as-judge designs." if hvg_all['bias_mean'] > 0
        else "the judge rates more strictly than humans on average."))
    ln()
    ln(f"4. **Ranking agreement:** Mean per-question Spearman ρ = {fmt(mean_rho)}. "
       "The judge and experts "
       + (f"agree well on the relative ordering of answers." if mean_rho >= 0.6
          else "show limited agreement on answer ordering — interpret ablation ranking differences cautiously."))
    ln()

    ln("---")
    ln()
    ln("*Report generated automatically. All statistics are reproducible by re-running "
       "`python study/analyze_validation.py`.*")

    pathlib.Path(path).write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] validation_report.md written  -> {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    tidy = pd.read_csv(args.tidy)
    key  = pd.read_csv(args.key)
    raters = sorted(tidy["rater"].unique())

    print(f"[analyze_validation.py]  {len(tidy)} rows, {len(raters)} raters, "
          f"{int(tidy['score'].isna().sum())} missing scores.")
    print(f"  Raters: {raters}")
    print()

    # Reshape into per-dim pivot tables
    print("Building per-dimension pivots ...")
    pivots, raters = build_pivots(tidy, key)

    # Composite scores
    composites = build_composites(pivots, key)

    # ── Inter-rater reliability ───────────────────────────────────────────────
    print("Computing inter-rater reliability ...")
    irr_by_dim = {dim: interrater_stats(pivots[dim], raters) for dim in DIM_CODES}

    # Pooled IRR: concatenate all dims into one long matrix
    all_items_per_dim = [pivots[dim] for dim in DIM_CODES]
    rater_cols = [c for c in all_items_per_dim[0].columns if c in raters]
    pooled_mat = np.vstack([p[rater_cols].values.astype(float) for p in all_items_per_dim])
    pooled_piv = pd.DataFrame(pooled_mat, columns=rater_cols)
    pooled_piv["item_id"] = [f"{dim}_{row['item_id']}"
                              for dim in DIM_CODES
                              for _, row in pivots[dim].iterrows()]
    irr_all = interrater_stats(pooled_piv, raters)

    # ── Human vs G-Eval agreement ─────────────────────────────────────────────
    print("Computing human vs G-Eval agreement ...")
    hvg_by_dim = {dim: human_vs_geval_stats(pivots[dim], raters) for dim in DIM_CODES}

    # Composite level
    h = composites["human_composite"].values.astype(float)
    g = composites["geval_composite"].values.astype(float)
    dummy_piv = pd.DataFrame({"human_consensus": h, "geval": g,
                               "item_id": composites["item_id"].values})
    hvg_all = human_vs_geval_stats(dummy_piv, raters=[])
    hvg_all["per_rater_spearman"] = {}  # no per-rater for composite level

    # ── Per-question ranking ──────────────────────────────────────────────────
    print("Computing per-question rank agreement ...")
    rank_df = rank_agreement_per_question(composites)

    # ── Descriptives ──────────────────────────────────────────────────────────
    desc_df = descriptive_table(pivots, composites)
    cond_df = condition_table(composites)

    # ── Figures ───────────────────────────────────────────────────────────────
    print(f"Generating figures -> {args.figdir}/")
    make_figures(pivots, composites, args.figdir, raters)
    for f in ["scatter_human_vs_geval.png", "composite_scatter.png",
              "bias_by_dimension.png", "interrater_heatmap.png"]:
        print(f"  {args.figdir}/{f}")

    # ── Stats CSV ─────────────────────────────────────────────────────────────
    stats_df = collect_stats_csv(irr_by_dim, hvg_by_dim, irr_all, hvg_all,
                                  rank_df, composites, raters)
    stats_df.to_csv(args.out_stats, index=False)
    print(f"[OK] agreement_stats.csv written  -> {args.out_stats}")

    # ── Report ────────────────────────────────────────────────────────────────
    print("Writing validation_report.md ...")
    write_report(args.out_report, tidy, key, pivots, composites, raters,
                 irr_by_dim, hvg_by_dim, irr_all, hvg_all,
                 rank_df, desc_df, cond_df)

    print()
    print("Done.")
    print(f"  Report : {args.out_report}")
    print(f"  Stats  : {args.out_stats}")
    print(f"  Figures: {args.figdir}/")


if __name__ == "__main__":
    main()
