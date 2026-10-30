"""
WINDOW RULE CHECK (offline, read-only; takes about a minute, no retraining).

    python window_eval.py

Question it answers: can a window-level rule catch the Gafgyt tcp/udp floods that the per-record reconstruction-error
rule cannot (because the flood is the same record repeated, and that record also exists in normal traffic)?

A WINDOW is W consecutive records. Two statistics per window, both computed from the saved federated model's errors:
    repeat rate  s = 1 - (distinct records in window) / W     (records equal after scaling, rounded to 1e-6)
    median RE    m = median reconstruction error in the window
Window alert  =  s > tau_rep   OR   m > tau_re
tau_rep and tau_re are the (1 - b) percentiles over NORMAL windows of the clients' own threshold-sets (the same
normal-only, label-free data the per-record threshold uses). No attack data is used to set anything.

Evaluation: non-overlapping windows of the server's held-out normal test (cut inside each device's own chunk) -> window
false-alarm rate; non-overlapping windows of every attack file -> window recall per file.
Prints a table for W in (50, 100, 200) and three calibrations (1%, 0.5% percentile, and strict 'max' = the highest
value any normal calibration window showed); saves results/window_eval.csv.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fl_iot_ae import task as T

WINDOWS = (50, 100, 200)
BUDGETS = (0.01, 0.005)


def cfg_value(key):
    txt = (Path(__file__).resolve().parent / "pyproject.toml").read_text()
    m = re.search(rf'^{key}\s*=\s*"(.*)"\s*$', txt, flags=re.M)
    return m.group(1) if m else ""


def record_ids(Xs: np.ndarray) -> np.ndarray:
    """Integer id per record; equal ids == equal records after scaling (rounded to 1e-6)."""
    R = np.ascontiguousarray(np.round(np.clip(Xs.astype(np.float64), -1e9, 1e9) * 1e6).astype(np.int64))
    v = R.view(np.dtype((np.void, R.dtype.itemsize * R.shape[1]))).ravel()
    return np.unique(v, return_inverse=True)[1].ravel()


def window_stats(ids: np.ndarray, re: np.ndarray, W: int, stride: int):
    """(repeat rate, median RE) for windows of W consecutive records, moving by `stride`."""
    s, m = [], []
    for a in range(0, len(ids) - W + 1, stride):
        s.append(1.0 - len(np.unique(ids[a:a + W])) / W)
        m.append(float(np.median(re[a:a + W])))
    return np.asarray(s), np.asarray(m)


def main():
    ddir = Path(cfg_value("data-dir"))
    rdir = Path(cfg_value("results-dir") or Path(__file__).resolve().parent / "results")
    ck = torch.load(rdir / "fl_global_3C.pt", weights_only=False)
    smin, smax = np.asarray(ck["smin"]), np.asarray(ck["smax"])
    meta = T.load_meta(ddir)
    model = T.build_model(meta["n_features"])
    model.load_state_dict(ck["state_dict"])
    dev = T.pick_device()
    test = T.load_server_test(ddir)

    # normal calibration data: each client's own chronological threshold-set (never mixed across clients)
    cal = []
    for i in range(int(meta["n_clients"])):
        X = T.scale(T.load_client(ddir, i, "thr"), smin, smax)
        cal.append((record_ids(X), T.reconstruction_error(model, X, dev)))
    # normal test: the server's held-out chunks, one per device, in order
    Xn = T.scale(test["normal"], smin, smax)
    re_n = T.reconstruction_error(model, Xn, dev)
    id_n = record_ids(Xn)
    bounds = np.cumsum([0] + [int(c["n_test"]) for c in meta["clients"]])
    att = {}
    for k, v in test["attacks"].items():
        X = T.scale(v, smin, smax)
        att[k] = (record_ids(X), T.reconstruction_error(model, X, dev))

    rows = []
    print("=" * 100)
    print("REPEAT RATE s per window  (how much of the window is the same record repeated)")
    print("=" * 100)
    W0 = 100
    s_cal = np.concatenate([window_stats(i, r, W0, W0 // 2)[0] for i, r in cal])
    s_test = np.concatenate([window_stats(id_n[a:b], re_n[a:b], W0, W0)[0] for a, b in zip(bounds[:-1], bounds[1:])])
    q = lambda x: " ".join(f"{np.percentile(x, p):6.3f}" for p in (50, 90, 99, 100))
    print(f"W={W0}      {'median':>6} {'p90':>6} {'p99':>6} {'max':>6}")
    print(f"normal (calibration windows, {len(s_cal):>5}) {q(s_cal)}")
    print(f"normal (test windows,        {len(s_test):>5}) {q(s_test)}")
    for k, (i, r) in att.items():
        s, _ = window_stats(i, r, W0, W0)
        print(f"{k:<24}({len(s):>5} windows)  {q(s)}")

    for W in WINDOWS:
        s_c, m_c = map(np.concatenate, zip(*[window_stats(i, r, W, max(W // 2, 1)) for i, r in cal]))
        parts = [window_stats(id_n[a:b], re_n[a:b], W, W) for a, b in zip(bounds[:-1], bounds[1:])]
        s_t, m_t = np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
        calibs = [(f"pct {b * 100:g}%", float(np.quantile(s_c, 1 - b)), float(np.quantile(m_c, 1 - b))) for b in BUDGETS]
        calibs.append(("max (strict)", float(s_c.max()), float(m_c.max())))   # highest value any normal window showed
        for cname, tau_rep, tau_re in calibs:
            fpr = dict(rep=float((s_t > tau_rep).mean()), re=float((m_t > tau_re).mean()),
                       any=float(((s_t > tau_rep) | (m_t > tau_re)).mean()))
            per = {}
            for k, (i, r) in att.items():
                s, m = window_stats(i, r, W, W)
                per[k] = dict(rep=float((s > tau_rep).mean()), re=float((m > tau_re).mean()),
                              any=float(((s > tau_rep) | (m > tau_re)).mean()), n=len(s))
            for rule in ("re", "rep", "any"):
                tot_w = sum(v["n"] for v in per.values())
                rec = sum(v[rule] * v["n"] for v in per.values()) / tot_w
                worst_k = min(per, key=lambda k: per[k][rule])
                rows.append(dict(W=W, calib=cname, rule=rule, tau_rep=tau_rep, tau_re=tau_re, window_fpr=fpr[rule],
                                 window_recall=rec, worst_file=worst_k, worst_file_recall=per[worst_k][rule],
                                 gafgyt_tcp=per.get("Gafgyt__tcp", {}).get(rule, np.nan),
                                 gafgyt_udp=per.get("Gafgyt__udp", {}).get(rule, np.nan)))
            if W == 100 and cname == "max (strict)":
                full = pd.DataFrame({k: {"RE rule": v["re"], "repeat rule": v["rep"], "either": v["any"]}
                                     for k, v in per.items()}).T
                print("\n" + "=" * 100)
                print(f"PER-FILE WINDOW RECALL at W=100, strict 'max' calibration   (tau_rep={tau_rep:.3f}, tau_re={tau_re:.2e})")
                print("=" * 100)
                print(full.to_string(float_format=lambda x: f"{x:.3f}"))

    df = pd.DataFrame(rows)
    df.to_csv(rdir / "window_eval.csv", index=False)
    print("\n" + "=" * 100)
    print("SUMMARY  (window-level: FPR on normal test windows, recall over attack windows)")
    print("=" * 100)
    print(df.drop(columns=["tau_rep", "tau_re"]).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved {rdir / 'window_eval.csv'}")


if __name__ == "__main__":
    main()