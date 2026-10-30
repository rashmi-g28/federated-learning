"""
STEP 4 -- baselines that make the federated result meaningful (same data, same scaler, same evaluation).

    python baselines.py --mode centralized     # all clients' data pooled, paper recipe (batch 2048, lr 1e-4, 100 epochs)
    python baselines.py --mode local           # every device trains ALONE (no federation), tested on everyone's traffic

  centralized = the upper bound ("what if we were allowed to pool the data?")
  local-only  = the lower bound ("what if devices never cooperate?")
Federated learning should land close to centralized and clearly above local-only.
Results go to <results-dir>/baseline_centralized.json and baseline_local.json.
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch

from fl_iot_ae import task as T


def cfg_value(key: str) -> str:
    txt = (Path(__file__).resolve().parent / "pyproject.toml").read_text()
    m = re.search(rf'^{key}\s*=\s*"(.*)"\s*$', txt, flags=re.M)
    return m.group(1) if m else ""


def thresholds_from(re_thr: np.ndarray, tag: str) -> dict:
    return {f"{tag} {b * 100:g}%": float(np.quantile(re_thr, 1.0 - b)) for b in T.BUDGETS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["centralized", "local"], required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-dir", default=cfg_value("data-dir"))
    ap.add_argument("--results-dir", default=cfg_value("results-dir"))
    a = ap.parse_args()
    if not a.data_dir:
        raise SystemExit("data-dir is empty: run prepare_partitions.py first or pass --data-dir")
    ddir = Path(a.data_dir)
    rdir = Path(a.results_dir or Path(__file__).resolve().parent / "results")
    rdir.mkdir(parents=True, exist_ok=True)

    device = T.pick_device()
    meta = T.load_meta(ddir)
    n_clients, n_feat = meta["n_clients"], meta["n_features"]
    train_raw = [T.load_client(ddir, i, "train") for i in range(n_clients)]
    thr_raw = [T.load_client(ddir, i, "thr") for i in range(n_clients)]
    # global min / max == the federated one (min of mins, max of maxes), so scaling is identical to the FL run
    smin = np.min([x.min(axis=0) for x in train_raw], axis=0).astype(np.float64)
    smax = np.max([x.max(axis=0) for x in train_raw], axis=0).astype(np.float64)
    test = T.load_server_test(ddir)
    print(f"mode={a.mode}  clients={n_clients}  features={n_feat}  epochs={a.epochs}  batch={a.batch_size}  lr={a.lr}")

    if a.mode == "centralized":
        X = T.scale(np.concatenate(train_raw), smin, smax)
        model = T.build_model(n_feat)
        t0 = time.time()
        loss, _ = T.train_local(model, X, a.epochs, a.batch_size, a.lr, device, seed=a.seed)
        secs = time.time() - t0
        re_thr = T.reconstruction_error(model, T.scale(np.concatenate(thr_raw), smin, smax), device)
        ev = T.evaluate_global(model, smin, smax, test, thresholds_from(re_thr, "Central"), device)
        out = dict(mode="centralized", params=T.count_params(model), train_seconds=secs, final_train_loss=loss,
                   epochs=a.epochs, batch_size=a.batch_size, lr=a.lr, evaluation=ev)
        (rdir / "baseline_centralized.json").write_text(json.dumps(T.jsonable(out), indent=2))
        print(f"AUC {ev['auc_all']:.4f} | worst-file AUC {ev['auc_worst_file']:.4f}")
        for k, op in ev["operating_points"].items():
            print(f"  {k:<14} FPR {op['fpr']:.4f}  recall {op['recall']:.4f}  F1 {op['f1']:.4f}  "
                  f"worst-file recall {op['worst_file_recall']:.4f}")
        print(f"Saved {rdir / 'baseline_centralized.json'}")
    else:
        per_client = []
        for i in range(n_clients):
            model = T.build_model(n_feat)
            X = T.scale(train_raw[i], smin, smax)
            t0 = time.time()
            loss, _ = T.train_local(model, X, a.epochs, a.batch_size, a.lr, device, seed=a.seed + i)
            re_thr = T.reconstruction_error(model, T.scale(thr_raw[i], smin, smax), device)
            ev = T.evaluate_global(model, smin, smax, test, thresholds_from(re_thr, "Local"), device)
            per_client.append(dict(client=i, n_train=int(len(X)), train_seconds=time.time() - t0, evaluation=ev))
            op = ev["operating_points"]["Local 1%"]
            print(f"  client {i}: AUC {ev['auc_all']:.4f} | at 1% budget: FPR {op['fpr']:.4f} recall {op['recall']:.4f} "
                  f"F1 {op['f1']:.4f} worst-file recall {op['worst_file_recall']:.4f}")
        out = dict(mode="local", epochs=a.epochs, batch_size=a.batch_size, lr=a.lr, clients=per_client)
        (rdir / "baseline_local.json").write_text(json.dumps(T.jsonable(out), indent=2))
        print(f"Saved {rdir / 'baseline_local.json'}")


if __name__ == "__main__":
    main()
