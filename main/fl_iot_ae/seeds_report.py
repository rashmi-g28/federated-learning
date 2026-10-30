"""
SEEDS REPORT -- mean +/- std over repeated federated runs (results/fl_run_seed<k>.json).

    python seeds_report.py

Reports, for each seed and then mean / std / min / max (sample std, n-1):
  per-record metrics at the 1% budget (pooled and strict 'max' thresholds), AUC, worst attack-file AUC,
  window-level false-alarm rate and recall (W = 50, 100, 200), and whether Gafgyt tcp/udp were caught.
Writes results/seeds_summary.csv and results/seeds_summary.txt.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def cfg_value(key):
    txt = (Path(__file__).resolve().parent / "pyproject.toml").read_text()
    m = re.search(rf'^{key}\s*=\s*"(.*)"\s*$', txt, flags=re.M)
    return m.group(1) if m else ""


def main():
    rdir = Path(cfg_value("results-dir") or Path(__file__).resolve().parent / "results")
    files = sorted(rdir.glob("fl_run_seed*.json"), key=lambda p: int(re.search(r"seed(\d+)", p.name).group(1)))
    if len(files) < 2:
        raise SystemExit(f"Need at least 2 files fl_run_seed*.json in {rdir}; found {len(files)}.")
    rows = []
    for f in files:
        r = json.loads(f.read_text())
        ev = r["evaluation"]
        row = {"seed": r["config"].get("seed"), "AUC (all)": ev["auc_all"], "AUC worst file": ev["auc_worst_file"],
               "AUC Gafgyt tcp": ev["auc_per_file"].get("Gafgyt__tcp"), "AUC Gafgyt udp": ev["auc_per_file"].get("Gafgyt__udp")}
        for name in ("FL pooled 1%", "FL max 1%"):
            op = ev["operating_points"][name]
            n0, n1 = ev["n_normal"], ev["n_attack"]
            row[f"[{name}] FPR"] = op["fpr"]
            row[f"[{name}] recall"] = op["recall"]
            row[f"[{name}] precision"] = op["precision"]
            row[f"[{name}] F1"] = op["f1"]
            row[f"[{name}] balanced acc"] = ((1 - op["fpr"]) + op["recall"]) / 2
            row[f"[{name}] worst-file recall"] = op["worst_file_recall"]
        for W, w in (r.get("window") or {}).items():
            row[f"[window {W}] FPR"] = w["window_fpr"]["any"]
            row[f"[window {W}] recall"] = w["window_recall"]["any"]
            row[f"[window {W}] worst-file recall"] = w["worst_file_recall"]["any"]
        row["comm total MB"] = r["communication"]["total_bytes"] / 1e6
        rows.append(row)
    df = pd.DataFrame(rows).set_index("seed")
    stat = pd.DataFrame({"mean": df.mean(), "std": df.std(ddof=1), "min": df.min(), "max": df.max()})
    out = ["=" * 100, f"PER SEED ({len(df)} runs, same data, same settings; only the seed differs)", "=" * 100,
           df.T.to_string(float_format=lambda x: f"{x:.4f}"), "", "=" * 100, "MEAN / STD / MIN / MAX", "=" * 100,
           stat.to_string(float_format=lambda x: f"{x:.4f}"),
           "", f"n = {len(df)} seeds: report mean and std together with the range; a std from 3 runs is itself noisy."]
    txt = "\n".join(out)
    (rdir / "seeds_summary.txt").write_text(txt)
    stat.to_csv(rdir / "seeds_summary.csv")
    print(txt)
    print(f"\nSaved {rdir / 'seeds_summary.txt'} and seeds_summary.csv")


if __name__ == "__main__":
    main()
