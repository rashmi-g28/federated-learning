"""
STEP 5 -- one comparison table for the report: Federated vs Centralized vs Local-only, plus the efficiency metrics.

    python report.py

Reads results/fl_run.json, baseline_centralized.json, baseline_local.json (whichever exist) and writes
results/fl_comparison.csv and results/fl_comparison.txt.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from fl_iot_ae import task as T


def cfg_value(key):
    txt = (Path(__file__).resolve().parent / "pyproject.toml").read_text()
    m = re.search(rf'^{key}\s*=\s*"(.*)"\s*$', txt, flags=re.M)
    return m.group(1) if m else ""


def load(rdir, name):
    p = rdir / name
    return json.loads(p.read_text()) if p.exists() else None


def main():
    rdir = Path(cfg_value("results-dir") or Path(__file__).resolve().parent / "results")
    fl, cen, loc = load(rdir, "fl_run.json"), load(rdir, "baseline_centralized.json"), load(rdir, "baseline_local.json")
    if fl is None:
        raise SystemExit(f"{rdir / 'fl_run.json'} not found. Run:  flwr run . --federation-config \"num-supernodes=N\"")
    lines, rows = [], []

    def add(budget, method, op, auc, extra="", n=None):
        acc = bacc = float("nan")
        if n:   # n = (normal records, attack records) of the evaluation set
            acc = ((1 - op["fpr"]) * n[0] + op["recall"] * n[1]) / (n[0] + n[1])
            bacc = ((1 - op["fpr"]) + op["recall"]) / 2
        rows.append(dict(budget=budget, method=method, fpr=op["fpr"], recall=op["recall"], precision=op["precision"],
                         f1=op["f1"], accuracy=acc, balanced_acc=bacc, worst_file_recall=op["worst_file_recall"],
                         auc=auc, note=extra))

    for b in T.BUDGETS:
        tag = f"{b * 100:g}%"
        n_fl = (fl["evaluation"]["n_normal"], fl["evaluation"]["n_attack"])
        add(tag, "Federated (FedAvg, pooled thr)", fl["evaluation"]["operating_points"][f"FL pooled {tag}"], fl["evaluation"]["auc_all"], n=n_fl)
        add(tag, "Federated (strict 'max' thr)", fl["evaluation"]["operating_points"][f"FL max {tag}"], fl["evaluation"]["auc_all"], n=n_fl)
        if cen:
            add(tag, "Centralized (pooled data)", cen["evaluation"]["operating_points"][f"Central {tag}"], cen["evaluation"]["auc_all"],
                n=(cen["evaluation"]["n_normal"], cen["evaluation"]["n_attack"]))
        if loc:
            ops = [c["evaluation"]["operating_points"][f"Local {tag}"] for c in loc["clients"]]
            mean_op = {k: float(np.mean([o[k] for o in ops])) for k in ("fpr", "recall", "precision", "f1")}
            mean_op["worst_file_recall"] = float(np.mean([o["worst_file_recall"] for o in ops]))
            add(tag, "Local-only (mean of devices)", mean_op, float(np.mean([c["evaluation"]["auc_all"] for c in loc["clients"]])),
                f"best device F1 {max(o['f1'] for o in ops):.4f}, worst {min(o['f1'] for o in ops):.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(rdir / "fl_comparison.csv", index=False)

    out = ["=" * 100, "DETECTION: federated vs centralized vs local-only (same data, same scaler, same hold-out)", "=" * 100]
    for tag, g in df.groupby("budget", sort=False):
        out.append(f"\nFalse-alarm budget {tag}")
        out.append(g.drop(columns="budget").to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    out.append("\nNote: the test set is ~88% attack records, so plain accuracy is flattered by recall; use balanced_acc / F1 / worst_file_recall.")

    if fl.get("window"):
        out += ["", "=" * 100, "WINDOW-LEVEL RULE (federated, strict 'max' cutoffs; unit = window of W consecutive records)", "=" * 100,
                f"{'W':>5}{'repeat-rule FPR':>16}{'RE-rule FPR':>13}{'any FPR':>10}{'recall any':>12}{'worst file':>12}{'  tau_rep':>10}{'tau_re':>11}"]
        for W, w in fl["window"].items():
            out.append(f"{W:>5}{w['window_fpr']['rep']:>16.4f}{w['window_fpr']['re']:>13.4f}{w['window_fpr']['any']:>10.4f}"
                       f"{w['window_recall']['any']:>12.4f}{w['worst_file_recall']['any']:>12.4f}{w['tau_rep']:>10.3f}{w['tau_re']:>11.2e}")
        out.append("Per-record recall (above) and window recall are different units; report both and label them.")
    c, p = fl["communication"], fl["compute"]
    out += ["", "=" * 100, "EFFICIENCY (Objective 3: communication / computation)", "=" * 100,
            f"Model                     : {fl['model']}  |  {fl['params']:,} parameters  |  {fl['macs_per_record']:,} MACs / record",
            f"Model payload             : {c['model_payload_bytes'] / 1024:.1f} KB  (one upload or one download)",
            f"Per client per round      : {c['per_client_per_round_down_bytes'] / 1024:.1f} KB down + {c['per_client_per_round_up_bytes'] / 1024:.1f} KB up",
            f"Whole FL run              : {c['total_bytes'] / 1e6:.2f} MB  ({fl['n_clients']} clients x {fl['config']['rounds']} rounds, "
            f"scaler round {c['scaler_round_bytes'] / 1024:.1f} KB, calibration round {c['calibration_round_bytes'] / 1024:.1f} KB)",
            f"Raw data sent to server   : 0 bytes (only weights, min/max, error histograms)",
            f"Client train time / round : mean {p['mean_client_train_seconds_per_round']:.2f} s, max {p['max_client_train_seconds_per_round']:.2f} s"
            if p["mean_client_train_seconds_per_round"] is not None else "Client train time / round : n/a",
            f"Training FLOPs / record   : ~{p['training_flops_per_record_approx']:,}  (6 x MACs, forward + backward)",
            f"Inference latency (CPU)   : {p['inference']['us_per_record_batch4096']:.2f} us/record (batch 4096), "
            f"{p['inference']['us_per_record_batch1']:.1f} us/record (batch 1, includes PyTorch call overhead)"]
    if cen:
        out.append(f"Centralized training time : {cen['train_seconds']:.1f} s on pooled data "
                   f"(FL wall-clock {p['fedavg_wall_clock_seconds']:.1f} s incl. simulation overhead)")
    txt = "\n".join(out)
    (rdir / "fl_comparison.txt").write_text(txt)
    print(txt)
    print(f"\nSaved {rdir / 'fl_comparison.csv'} and fl_comparison.txt")


if __name__ == "__main__":
    main()