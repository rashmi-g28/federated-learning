"""
compare_all.py -- compares every variant of the 3-, 4- and 5-hidden-layer studies.

Run AFTER run_3_layer.py, run_4_layer.py and run_5_layer.py:
    python compare_all.py

It first VERIFIES that all three families were run on the identical data (fingerprint) with the identical
settings (epochs, batch, lr, scaler, contamination, seeds); otherwise it stops. Then it writes:
    results/comparison_summary.csv          mean/std over seeds for every variant
    results/comparison_per_attack_file.csv  recall / F1 / AUC per variant per attack file
    results/comparison_report.txt           the tables + decision guide printed below
    results/fig1_scores_by_variant.png      F1, FPR, AUC per variant (dot = mean, bar = std over seeds)
    results/fig2_f1_vs_size.png             accuracy vs model size, Pareto frontier = best trade-offs
    results/fig3_recall_by_attack_file.png  recall of each variant on each attack file
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

RESULTS = Path(__file__).resolve().parent / "results"

# ---- decision-rule thresholds (change if your project needs different bars) ----
TOL_F1 = 0.001          # variants within max(TOL_F1, 1 std) of the best F1 count as "effectively tied"
RECALL_FLOOR = 0.99     # for the "lowest false-alarm" pick
UNSTABLE_STD = 0.002    # F1 std across seeds above this = unstable training
COLLAPSE_RECALL = 0.90  # worst attack-file recall below this = the model misses a whole attack type
HIGH_FPR = 0.05

# ---- palette (reference palette: first three categorical slots validate for all-pairs use) ----
FAMILY_COLOR = {3: "#2a78d6", 4: "#eb6834", 5: "#1baf7a"}
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE_RAMP = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]

METRICS = ["all_recall", "all_precision", "normal_fpr", "all_f1", "all_auc", "mirai_recall", "gafgyt_recall",
           "worst_file_recall", "params", "size_kb", "latency_us", "train_s", "val_mse", "threshold"]


# ------------------------------------------------------------------ loading + safety checks
def load(allow_missing):
    runs, files, metas, missing = [], [], {}, []
    for n in (3, 4, 5):
        p = RESULTS / f"layers_{n}_runs.csv"
        if not p.exists():
            missing.append(n)
            continue
        runs.append(pd.read_csv(p))
        files.append(pd.read_csv(RESULTS / f"layers_{n}_per_file.csv"))
        metas[n] = json.loads((RESULTS / f"layers_{n}_meta.json").read_text())
    if missing and not (allow_missing and len(metas) >= 2):
        raise SystemExit("Missing results for: " + ", ".join(f"{n}-layer (run: python run_{n}_layer.py)" for n in missing)
                         + "\n(use --allow-missing to compare only the families that exist)")
    ref_n = next(iter(metas))
    for n, m in metas.items():
        for key in ("fingerprint", "settings", "n_features", "attack_files", "n_normal_test"):
            if m[key] != metas[ref_n][key]:
                raise SystemExit(f"REFUSING TO COMPARE: family {n} differs from family {ref_n} in '{key}':\n"
                                 f"  {n}: {m[key]}\n  {ref_n}: {metas[ref_n][key]}\n"
                                 f"Re-run the mismatching family with the same data and settings (same --epochs/--n-seeds).")
    return pd.concat(runs, ignore_index=True), pd.concat(files, ignore_index=True), metas


def aggregate(runs):
    g = runs.groupby(["family", "variant"], sort=True)
    mean, std = g[METRICS].mean(), g[METRICS].std(ddof=1).fillna(0.0)
    S = mean.copy()
    for m in METRICS:
        S[m + "_std"] = std[m]
    S["arch"] = g["arch"].first()
    S["n_seeds"] = g.size()
    S["worst_file"] = g["worst_file"].agg(lambda x: x.value_counts().index[0])
    return S.reset_index()


# ------------------------------------------------------------------ tables
def pm(mean, std, n_seeds, d=4):
    return f"{mean:.{d}f}" if n_seeds < 2 else f"{mean:.{d}f}±{std:.{d}f}"


def table(S):
    t = pd.DataFrame({
        "Variant": S["variant"], "Architecture": S["arch"], "Params": S["params"].round().astype(int).map("{:,}".format),
        "KB": S["size_kb"].map("{:.1f}".format),
        "Recall": [pm(r.all_recall, r.all_recall_std, r.n_seeds) for r in S.itertuples()],
        "Precision": [pm(r.all_precision, r.all_precision_std, r.n_seeds) for r in S.itertuples()],
        "FPR": [pm(r.normal_fpr, r.normal_fpr_std, r.n_seeds) for r in S.itertuples()],
        "F1": [pm(r.all_f1, r.all_f1_std, r.n_seeds) for r in S.itertuples()],
        "AUC": [pm(r.all_auc, r.all_auc_std, r.n_seeds) for r in S.itertuples()],
        "WorstFileRecall": S["worst_file_recall"].map("{:.4f}".format),
        "us/rec": S["latency_us"].map("{:.2f}".format),
    })
    return t.to_string(index=False)


def pareto(S):
    keep = []
    for i, a in S.iterrows():
        dominated = ((S["all_f1"] >= a.all_f1) & (S["params"] <= a.params) &
                     ((S["all_f1"] > a.all_f1) | (S["params"] < a.params))).any()
        if not dominated:
            keep.append(i)
    return S.loc[keep].sort_values("params")


def decision_guide(S):
    L = []
    best = S.loc[S["all_f1"].idxmax()]
    tol = max(TOL_F1, best.all_f1_std)
    tied = S[S["all_f1"] >= best.all_f1 - tol].sort_values(["params", "latency_us"])
    light = tied.iloc[0]
    L.append("DECISION GUIDE (rules are mechanical; read them with the tables above)")
    L.append("-" * 78)
    L.append(f"1. Highest F1 ............ {best.variant}  {best.arch}  F1={best.all_f1:.4f}  recall={best.all_recall:.4f}  "
             f"FPR={best.normal_fpr:.4f}  params={int(best.params):,}")
    L.append(f"2. Effectively tied with #1 (F1 within {tol:.4f}): " + ", ".join(tied["variant"]))
    L.append(f"3. LIGHTEST of the tied .. {light.variant}  {light.arch}  params={int(light.params):,} ({light.size_kb:.1f} KB per FL "
             f"upload)  F1={light.all_f1:.4f}  <- default pick for the 'lightweight FL' objective")
    a = S.loc[S["all_auc"].idxmax()]
    if S["all_auc"].max() - S["all_auc"].min() < 5e-4:
        L.append(f"4. AUC (threshold-free quality) is saturated: every variant scores {S['all_auc'].min():.4f}-{S['all_auc'].max():.4f}. "
                 "All models separate normal from attack; the F1/FPR differences above come from where the threshold lands.")
    else:
        L.append(f"4. Best AUC (threshold-free quality) .. {a.variant}  AUC={a.all_auc:.5f}"
                 + ("" if a.variant == best.variant else
                    f"   (differs from #1 -> {best.variant}'s edge is threshold placement, not better separation)"))
    ok = S[S["all_recall"] >= RECALL_FLOOR]
    if len(ok):
        f = ok.loc[ok["normal_fpr"].idxmin()]
        L.append(f"5. Lowest false-alarm rate with recall >= {RECALL_FLOOR} .. {f.variant}  FPR={f.normal_fpr:.4f}  recall={f.all_recall:.4f}")
    else:
        L.append(f"5. No variant reaches recall >= {RECALL_FLOOR}.")
    P = pareto(S)
    L.append("6. Pareto frontier (nothing is both smaller AND more accurate): " +
             ", ".join(f"{r.variant}({int(r.params):,}p, F1 {r.all_f1:.4f})" for r in P.itertuples()))
    L.append("7. Best of each depth:")
    for fam, grp in S.groupby("family"):
        b = grp.loc[grp["all_f1"].idxmax()]
        L.append(f"     {fam} hidden layers -> {b.variant}  F1={b.all_f1:.4f}±{b.all_f1_std:.4f}  FPR={b.normal_fpr:.4f}  params={int(b.params):,}")
    fam_best = S.groupby("family")["all_f1"].max()
    spread = fam_best.max() - fam_best.min()
    L.append(f"   Depth effect: best-F1 spread across depths = {spread:.4f} "
             + ("-> depth barely matters; choose by size/latency." if spread <= max(TOL_F1, best.all_f1_std)
                else f"-> depth {int(fam_best.idxmax())} is genuinely better."))
    L.append("8. Warnings:")
    w = 0
    for r in S.itertuples():
        msgs = []
        if r.worst_file_recall < COLLAPSE_RECALL:
            msgs.append(f"misses an attack type (worst-file recall {r.worst_file_recall:.3f} on {r.worst_file})")
        if r.all_f1_std > UNSTABLE_STD:
            msgs.append(f"unstable across seeds (F1 std {r.all_f1_std:.4f})")
        if r.normal_fpr > HIGH_FPR:
            msgs.append(f"high false-alarm rate ({r.normal_fpr:.3f})")
        if msgs:
            w += 1
            L.append(f"     {r.variant}: " + "; ".join(msgs))
    if not w:
        L.append("     none")
    if S["n_seeds"].min() < 2:
        L.append("   NOTE: only 1 seed -> no std; differences of ~0.001 are NOT reliable. Use the full run (3 seeds) for decisions.")
    return L


# ------------------------------------------------------------------ figures
def style_ax(ax):
    ax.set_facecolor(SURFACE)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(color=GRID, lw=0.6, zorder=0)


def fig_scores(S):
    S = S.sort_values(["family", "variant"]).reset_index(drop=True)
    y = np.arange(len(S))[::-1]
    fig, axes = plt.subplots(1, 3, figsize=(13, 0.42 * len(S) + 2.2), sharey=True, facecolor=SURFACE)
    panels = [("all_f1", "F1  (higher is better)"), ("normal_fpr", "False positive rate  (lower is better)"),
              ("all_auc", "AUC  (threshold-free, higher is better)")]
    for ax, (col, title) in zip(axes, panels):
        style_ax(ax)
        for i, r in S.iterrows():
            c = FAMILY_COLOR[int(r.family)]
            ax.errorbar(r[col], y[i], xerr=r[col + "_std"], fmt="o", ms=8, color=c, mec=SURFACE, mew=1.5,
                        ecolor=c, elinewidth=1.5, capsize=0, zorder=3)
            ax.annotate(f"{r[col]:.4f}", (r[col], y[i]), xytext=(9, 0), textcoords="offset points",
                        va="center", fontsize=8, color=INK2)
        lo, hi = S[col].min(), S[col].max()
        pad = max((hi - lo) * 0.25, 1e-3)
        ax.set_xlim(lo - pad, hi + pad * 2.2)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([f"{r.variant}  {'-'.join(map(str, r.arch.split('-')[1:-1]))}" for r in S.itertuples()],
                            fontsize=9, color=INK)
    handles = [Line2D([], [], marker="o", ls="", color=FAMILY_COLOR[f], mec=SURFACE, ms=8, label=f"{f} hidden layers")
               for f in sorted(S["family"].unique())]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9, labelcolor=INK2)
    fig.suptitle("Every variant on the same data: dot = mean over seeds, bar = ±1 std", x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(RESULTS / "fig1_scores_by_variant.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def fig_size(S):
    best = S["all_f1"].max()
    P = pareto(S)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), facecolor=SURFACE)
    for ax, zoom in zip(axes, (False, True)):
        style_ax(ax)
        D = S[S["all_f1"] >= best - 0.02] if zoom else S
        for r in D.itertuples():
            ax.scatter(r.params, r.all_f1, s=70, color=FAMILY_COLOR[int(r.family)], edgecolor=SURFACE, linewidth=1.5, zorder=3)
            ax.annotate(r.variant, (r.params, r.all_f1), xytext=(7, 5), textcoords="offset points", fontsize=9, color=INK)
        Pd = P[P["variant"].isin(D["variant"])]
        if len(Pd) > 1:
            ax.step(Pd["params"], Pd["all_f1"], where="post", color=MUTED, lw=1.2, ls="--", zorder=2)
        ax.set_xscale("log")
        ax.set_xlabel("Parameters (log scale)  -  smaller = cheaper FL round / edge inference", color=INK2, fontsize=9)
        ax.set_ylabel("F1", color=INK2, fontsize=9)
        ax.set_title("All variants" if not zoom else f"Zoom: within 0.02 of the best F1", loc="left", fontsize=10, color=INK)
    handles = [Line2D([], [], marker="o", ls="", color=FAMILY_COLOR[f], mec=SURFACE, ms=8, label=f"{f} hidden layers")
               for f in sorted(S["family"].unique())]
    handles.append(Line2D([], [], color=MUTED, ls="--", label="Pareto frontier"))
    axes[0].legend(handles=handles, frameon=False, fontsize=9, labelcolor=INK2, loc="lower right")
    fig.suptitle("Accuracy vs model size: variants on the dashed frontier are the best trade-offs", x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(RESULTS / "fig2_f1_vs_size.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def fig_heat(S, files):
    order = S.sort_values(["family", "variant"])["variant"].tolist()
    H = files.groupby(["variant", "category", "file"])["recall"].mean().reset_index()
    cols = [(c, f) for c in ("Mirai", "Gafgyt") for f in sorted(H[H["category"] == c]["file"].unique())]
    M = np.array([[H[(H.variant == v) & (H.category == c) & (H.file == f)]["recall"].mean() for c, f in cols] for v in order])
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    fig, ax = plt.subplots(figsize=(1.0 + 0.9 * len(cols), 0.45 * len(order) + 2.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    vmin = min(0.5, float(np.nanmin(M)))
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=1.0, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=7.5,
                    color="#ffffff" if (M[i, j] - vmin) / (1 - vmin) > 0.55 else INK)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([f"{c}\n{f}" for c, f in cols], fontsize=8, color=INK2, rotation=0)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=9, color=INK)
    n_mirai = sum(1 for c, _ in cols if c == "Mirai")
    ax.axvline(n_mirai - 0.5, color=SURFACE, lw=4)
    fams = S.sort_values(["family", "variant"])["family"].tolist()
    for i in range(1, len(fams)):
        if fams[i] != fams[i - 1]:
            ax.axhline(i - 0.5, color=SURFACE, lw=4)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("recall (mean over seeds)", color=INK2, fontsize=9)
    cb.ax.tick_params(colors=INK2, labelsize=8)
    ax.set_title("Recall of each variant on each attack file (a pale cell = that attack is being missed)",
                 loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig3_recall_by_attack_file.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-missing", action="store_true", help="compare only the families that have been run")
    a = ap.parse_args()

    runs, files, metas = load(a.allow_missing)
    S = aggregate(runs)
    m0 = next(iter(metas.values()))
    out = []
    out.append("=" * 78)
    out.append("AUTOENCODER DEPTH STUDY - 3 vs 4 vs 5 hidden layers, all variants, identical data")
    out.append("=" * 78)
    out.append(f"data fingerprint {m0['fingerprint']} | normal-test records {m0['n_normal_test']} | "
               f"features {m0['n_features']} | attack files {len(m0['attack_files'])}")
    st = m0["settings"]
    out.append(f"epochs {st['epochs']} | batch {st['batch_size']} | lr {st['lr']} | scaler {st['scaler']} | "
               f"IF contamination {st['if_contamination']} | seeds {st['seeds']}  (mean±std over seeds)")
    out.append("Attack metrics pool ALL attack files (Mirai + Gafgyt) against the 20% normal hold-out; FPR is on that hold-out.")
    out.append("")
    for fam, grp in S.groupby("family"):                       # one table per family: never mixed
        out.append(f"--- {fam} HIDDEN LAYERS " + "-" * 56)
        out.append(table(grp))
        for v, spec in metas[fam]["variants"].items():
            out.append(f"   {v}: {spec['why']}")
        out.append("")
    out.append("--- ALL VARIANTS RANKED BY F1 (separate-family tables above; this is the cross-family view) " + "-" * 0)
    out.append(table(S.sort_values("all_f1", ascending=False)))
    out.append("")
    out += decision_guide(S)
    report = "\n".join(out)
    print(report)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "comparison_report.txt").write_text(report + "\n")
    S.to_csv(RESULTS / "comparison_summary.csv", index=False)
    (files.groupby(["family", "variant", "category", "file"])[["recall", "precision", "f1", "auc", "median_re"]]
          .mean().reset_index().to_csv(RESULTS / "comparison_per_attack_file.csv", index=False))
    fig_scores(S); fig_size(S); fig_heat(S, files)
    print("\nSaved to results/: comparison_report.txt, comparison_summary.csv, comparison_per_attack_file.csv, "
          "fig1_scores_by_variant.png, fig2_f1_vs_size.png, fig3_recall_by_attack_file.png")


if __name__ == "__main__":
    main()