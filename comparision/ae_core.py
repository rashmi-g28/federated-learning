"""
ae_core.py -- shared engine for the 3 / 4 / 5 hidden-layer autoencoder study.

Everything that must be IDENTICAL across variants lives here, so the three runner scripts
(run_3_layer.py, run_4_layer.py, run_5_layer.py) cannot drift apart:
    * data loading + 80/20 split + threshold set + attack sampling   (deterministic, DATA_SEED)
    * feature scaling (fitted on AE-training normal data only)
    * training recipe from the paper: RMSprop, lr 1e-4, batch 2048, 100 epochs, MSE loss
    * threshold selection: 10% threshold set + Isolation Forest (paper Sec. 3.3)
    * metrics: Recall, Precision, FPR, F1 (+ AUC, threshold-free)

Only the hidden-layer sizes differ between runs. Each run stores a fingerprint of the exact data
it saw, and compare_all.py refuses to compare runs whose data/settings differ.
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

BASE_PATH = Path(__file__).resolve().parent
RESULTS_DIR = BASE_PATH / "results"

# ============================================================
# SETTINGS  (identical for every variant of every family)
# ============================================================
SETTINGS = dict(
    # data split (as requested): 80% normal -> train, 20% normal + attack folders -> test
    train_frac=0.80,
    threshold_frac=0.10,        # paper: 10% of the training set is the disjoint "threshold set"
    shuffle_split=False,        # paper keeps record order (temporal windows)
    val_frac=0.05,              # slice of AE-training data used only to monitor validation MSE
    max_attack_per_file=100_000,  # None = every attack record; cap only for RAM/speed
    # paper's training recipe (Sec. 4.2.2)
    epochs=100,
    batch_size=2048,
    lr=1e-4,
    # choices where the paper is silent (same for all variants)
    scaler="minmax",
    if_contamination=0.01,
    # reproducibility
    data_seed=42,               # fixes split/sampling -> same data for every variant
    seeds=(42, 43, 44),         # model-init/batch-order seeds; results are mean +- std over these
)

ATTACK_SUBDIRS = {"Mirai": "attacks/mirai_attacks", "Gafgyt": "attacks/gafgyt_attacks"}
LABEL_LIKE = {"label", "class", "attack", "target", "y", "unnamed: 0"}

# ============================================================
# THE VARIANTS  (hidden-layer widths; N = 115 input/output is implicit)
# Every variant exists to answer ONE question for the project:
#   "Lightweight FL anomaly detector": small model (cheap FL rounds + edge inference) that still
#   detects Mirai/Gafgyt with a low false-alarm rate.
# ============================================================
VARIANTS = {
    3: {
        "3A": dict(hidden=(48, 6, 48),
                   why="Paper Config A. Baseline. Paper: great per-device, but collapses all-in-one "
                       "(recall 0.35-0.79) -> check whether it also collapses on your merged data."),
        "3B": dict(hidden=(64, 6, 64),
                   why="Paper Config B. Baseline. Same 6-neuron bottleneck, wider -> isolates the "
                       "effect of width."),
        "3C": dict(hidden=(32, 6, 32),
                   why="LIGHTWEIGHT. Half the width of 3B, same bottleneck -> how much accuracy do "
                       "we pay for a smaller FL payload?"),
        "3D": dict(hidden=(64, 12, 64),
                   why="BOTTLENECK WIDTH. 3B with a 12-neuron code instead of 6 -> is the 6-neuron "
                       "bottleneck too tight for merged multi-device traffic?"),
    },
    4: {
        "4A": dict(hidden=(64, 12, 12, 64),
                   why="Symmetric, narrow middle (12-12). The 'even' layout: no single bottleneck "
                       "layer, a 2-layer code."),
        "4B": dict(hidden=(64, 24, 24, 64),
                   why="Symmetric, wide middle (24-24) -> tests whether a wider code helps when "
                       "the net has no single bottleneck."),
        "4C": dict(hidden=(64, 24, 6, 24),
                   why="ASYMMETRIC, keeps the paper's 6-neuron bottleneck (encoder 64-24-6, light "
                       "1-layer decoder) -> does a true bottleneck matter at 4 layers?"),
        "4D": dict(hidden=(32, 12, 12, 32),
                   why="LIGHTWEIGHT symmetric -> smallest 4-layer model."),
    },
    5: {
        "5A": dict(hidden=(64, 24, 6, 24, 64),
                   why="Paper's all-in-one design (Table 5). Baseline; paper reports recall "
                       "0.9994-0.9997, FPR 0.0-0.0071."),
        "5B": dict(hidden=(64, 24, 12, 24, 64),
                   why="BOTTLENECK WIDTH. 5A with a 12-neuron code (single change vs 5A)."),
        "5C": dict(hidden=(32, 16, 6, 16, 32),
                   why="LIGHTWEIGHT. About a quarter of 5A's parameters -> the real test of the "
                       "'lightweight' objective at depth 5."),
        "5D": dict(hidden=(128, 48, 6, 48, 128),
                   why="WIDER. Tests the paper's claim that adding width/depth helps all-in-one "
                       "training (still tiny vs. a CNN/LSTM)."),
    },
}
for _n, _vs in VARIANTS.items():           # guard: a family can only contain its own depth
    for _k, _v in _vs.items():
        assert len(_v["hidden"]) == _n, f"{_k} has {len(_v['hidden'])} hidden layers, expected {_n}"


# ============================================================
# MODEL
# ============================================================
class AE(nn.Module):
    """N -> hidden... -> N ; ReLU hidden layers, Tanh output, Glorot init (Keras defaults)."""

    def __init__(self, n_features, hidden):
        super().__init__()
        layers, prev = [], n_features
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, n_features), nn.Tanh()]
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


def arch_string(n_features, hidden):
    return "-".join(str(x) for x in (n_features, *hidden, n_features))


# ============================================================
# DATA
# ============================================================
def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    df = df[[c for c in df.columns if c.lower() not in LABEL_LIKE]]
    df = df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return df.dropna(axis=0, how="any").reset_index(drop=True)


def align_columns(df, ref_cols, name):
    missing = [c for c in ref_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"[{name}] column mismatch with the normal data ({len(missing)}/{len(ref_cols)} normal "
            f"columns missing). normal cols: {list(ref_cols)[:4]}...  this file: {list(df.columns)[:4]}...\n"
            f"Silently filling these would make every attack look normal. Fix the headers first."
        )
    return df[list(ref_cols)]


def _fingerprint(named_arrays):
    h = hashlib.sha1()
    for name, a in named_arrays:
        h.update(name.encode()); h.update(str(a.shape).encode())
        h.update(np.ascontiguousarray(a[:: max(1, len(a) // 2000)]).tobytes())
    return h.hexdigest()[:16]


def load_prepared_data(data_root: Path, s=SETTINGS):
    """Deterministic: the same files + same settings always give byte-identical arrays."""
    rng = np.random.default_rng(s["data_seed"])
    normal_files = sorted((data_root / "normal").rglob("*.csv"))
    if not normal_files:
        raise FileNotFoundError(f"No CSV files in {data_root / 'normal'}")

    print(f"Loading NORMAL data ({len(normal_files)} files) from {data_root / 'normal'}")
    ref_cols, fit_p, thr_p, test_p = None, [], [], []
    for f in normal_files:
        df = load_csv(f)
        ref_cols = ref_cols or list(df.columns)
        df = align_columns(df, ref_cols, f"normal/{f.name}")
        n = len(df)
        idx = rng.permutation(n) if s["shuffle_split"] else np.arange(n)
        n_train = int(round(n * s["train_frac"]))
        train, test = idx[:n_train], idx[n_train:]
        n_thr = int(round(len(train) * s["threshold_frac"]))
        fit_p.append(df.iloc[train[: len(train) - n_thr]])
        thr_p.append(df.iloc[train[len(train) - n_thr:]])
        test_p.append(df.iloc[test])
    X_fit = pd.concat(fit_p, ignore_index=True).to_numpy(np.float64)
    X_thr = pd.concat(thr_p, ignore_index=True).to_numpy(np.float64)
    X_nt = pd.concat(test_p, ignore_index=True).to_numpy(np.float64)

    scaler = (MinMaxScaler() if s["scaler"] == "minmax" else StandardScaler()).fit(X_fit)
    X_fit, X_thr, X_nt = (scaler.transform(a).astype(np.float32) for a in (X_fit, X_thr, X_nt))

    perm = rng.permutation(len(X_fit))
    n_val = int(len(X_fit) * s["val_frac"])
    X_val, X_train = X_fit[perm[:n_val]], X_fit[perm[n_val:]]

    attacks = {}
    for cat, sub in ATTACK_SUBDIRS.items():
        files = sorted((data_root / sub).rglob("*.csv"))
        if not files:
            raise FileNotFoundError(f"No CSV files in {data_root / sub}")
        print(f"Loading {cat} attacks ({len(files)} files)")
        for f in files:
            df = align_columns(load_csv(f), ref_cols, f"{cat}/{f.name}")
            cap = s["max_attack_per_file"]
            if cap and len(df) > cap:
                df = df.iloc[np.sort(rng.choice(len(df), cap, replace=False))]
            attacks[(cat, f.stem)] = scaler.transform(df.to_numpy(np.float64)).astype(np.float32)

    data = dict(X_train=X_train, X_val=X_val, X_thr=X_thr, X_ntest=X_nt, attacks=attacks,
                n_features=X_fit.shape[1], columns=ref_cols)
    named = [("train", X_train), ("val", X_val), ("thr", X_thr), ("ntest", X_nt)] + \
            [(f"{c}/{n}", a) for (c, n), a in attacks.items()]
    data["fingerprint"] = _fingerprint(named)
    print(f"N={data['n_features']} | AE-train {len(X_train)} | val {len(X_val)} | threshold-set {len(X_thr)} | "
          f"normal-test {len(X_nt)} | attack files {len(attacks)} ({sum(len(a) for a in attacks.values())} records)")
    print(f"Data fingerprint: {data['fingerprint']}")
    return data


# ============================================================
# SCORING, THRESHOLD, METRICS
# ============================================================
@torch.no_grad()
def reconstruction_error(model, X, device, batch=65536):
    model.eval()
    out = np.empty(len(X), dtype=np.float64)
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch]).to(device)
        out[i:i + batch] = ((model(xb) - xb) ** 2).mean(dim=1).double().cpu().numpy()
    return out


def select_threshold(re_in, re_out):
    """Paper Sec. 3.3: start at max(RE_OUT), lower until inliers-below / outliers-above agree best."""
    if len(re_out) == 0:
        return float(re_in.max())
    s_in, s_out = np.sort(re_in), np.sort(re_out)
    cand = np.unique(np.concatenate([s_in, s_out]))
    cand = cand[cand <= s_out[-1]][::-1]
    f_in = np.searchsorted(s_in, cand, side="right")
    f_out = len(s_out) - np.searchsorted(s_out, cand, side="right")
    return float(cand[int(np.argmax(f_in + f_out))])


def metrics(re_normal, re_attack, thr):
    tp = int((re_attack > thr).sum()); fn = len(re_attack) - tp
    fp = int((re_normal > thr).sum()); tn = len(re_normal) - fp
    R = tp / (tp + fn) if tp + fn else 0.0
    P = tp / (tp + fp) if tp + fp else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    y = np.r_[np.zeros(len(re_normal)), np.ones(len(re_attack))]
    auc = roc_auc_score(y, np.r_[re_normal, re_attack])
    return dict(recall=R, precision=P, fpr=fp / (fp + tn), f1=F, auc=auc)


# ============================================================
# ONE TRAINING RUN
# ============================================================
def run_one(vname, hidden, seed, data, s, device):
    n = data["n_features"]
    torch.manual_seed(seed); np.random.seed(seed)
    model = AE(n, hidden).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.RMSprop(model.parameters(), lr=s["lr"], alpha=0.9, eps=1e-7)  # Keras defaults
    loss_fn = nn.MSELoss()
    Xt = torch.from_numpy(data["X_train"]).to(device)
    Xv = torch.from_numpy(data["X_val"]).to(device)

    t0 = time.time()
    for _ in range(s["epochs"]):
        model.train()
        order = torch.randperm(len(Xt), device=device)
        tot = 0.0
        for i in range(0, len(Xt), s["batch_size"]):
            xb = Xt[order[i:i + s["batch_size"]]]
            loss = loss_fn(model(xb), xb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(xb)
    train_s = time.time() - t0
    model.eval()
    with torch.no_grad():
        val_mse = loss_fn(model(Xv), Xv).item() if len(Xv) else float("nan")
    train_mse = tot / len(Xt)

    # threshold from the 10% threshold set (paper Sec. 3.3)
    iso = IsolationForest(contamination=s["if_contamination"], random_state=seed, n_jobs=-1).fit(data["X_thr"])
    inl = iso.predict(data["X_thr"]) == 1
    re_thr = reconstruction_error(model, data["X_thr"], device)
    thr = select_threshold(re_thr[inl], re_thr[~inl])

    # inference latency (microseconds / record, batch 4096)
    probe = data["X_ntest"][:20000]
    reconstruction_error(model, probe[:4096], device, batch=4096)
    t1 = time.time(); reconstruction_error(model, probe, device, batch=4096)
    latency_us = (time.time() - t1) / len(probe) * 1e6

    re_normal = reconstruction_error(model, data["X_ntest"], device)
    per_file, cat_err = [], {}
    for (cat, name), Xa in data["attacks"].items():
        re_a = reconstruction_error(model, Xa, device)
        cat_err.setdefault(cat, []).append(re_a)
        per_file.append(dict(variant=vname, seed=seed, category=cat, file=name, n_attack=len(re_a),
                             median_re=float(np.median(re_a)), **metrics(re_normal, re_a, thr)))

    summary = dict(variant=vname, seed=seed, arch=arch_string(n, hidden), n_hidden=len(hidden),
                   params=n_params, size_kb=n_params * 4 / 1024, train_s=train_s, latency_us=latency_us,
                   train_mse=train_mse, val_mse=val_mse, threshold=thr,
                   if_outliers=int((~inl).sum()), normal_median_re=float(np.median(re_normal)))
    all_att = []
    for cat, errs in cat_err.items():
        e = np.concatenate(errs); all_att.append(e)
        m = metrics(re_normal, e, thr)
        summary.update({f"{cat.lower()}_{k}": v for k, v in m.items() if k != "fpr"})
    m = metrics(re_normal, np.concatenate(all_att), thr)
    summary.update({f"all_{k}": v for k, v in m.items()})
    summary["normal_fpr"] = m["fpr"]
    pf = pd.DataFrame(per_file)
    summary["worst_file_recall"] = float(pf["recall"].min())
    summary["worst_file"] = pf.loc[pf["recall"].idxmin(), "category"] + "/" + pf.loc[pf["recall"].idxmin(), "file"]
    return summary, per_file


# ============================================================
# ONE FAMILY (3, 4 or 5 hidden layers) -- never mixes families
# ============================================================
def run_family(n_layers, data_root, epochs=None, n_seeds=None):
    s = dict(SETTINGS)
    if epochs is not None:
        s["epochs"] = int(epochs)
    if n_seeds is not None:
        s["seeds"] = tuple(SETTINGS["seeds"][0] + i for i in range(int(n_seeds)))
    variants = VARIANTS[n_layers]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'=' * 78}\nFAMILY: {n_layers} HIDDEN LAYERS  |  variants: {', '.join(variants)}  |  device: {device}")
    print(f"epochs={s['epochs']} batch={s['batch_size']} lr={s['lr']} seeds={list(s['seeds'])}\n{'=' * 78}")

    data = load_prepared_data(Path(data_root), s)
    runs, files = [], []
    for vname, spec in variants.items():
        print(f"\n--- {vname}: {arch_string(data['n_features'], spec['hidden'])} ---\n    {spec['why']}")
        for seed in s["seeds"]:
            summ, pf = run_one(vname, spec["hidden"], seed, data, s, device)
            summ["family"] = n_layers
            runs.append(summ); files += [dict(family=n_layers, **r) for r in pf]
            print(f"    seed {seed}: params={summ['params']:,}  thr={summ['threshold']:.4g}  "
                  f"recall={summ['all_recall']:.4f}  FPR={summ['normal_fpr']:.4f}  F1={summ['all_f1']:.4f}  "
                  f"AUC={summ['all_auc']:.4f}  worst-file recall={summ['worst_file_recall']:.3f} "
                  f"({summ['worst_file']})  [{summ['train_s']:.0f}s]")

    RESULTS_DIR.mkdir(exist_ok=True)
    pd.DataFrame(runs).to_csv(RESULTS_DIR / f"layers_{n_layers}_runs.csv", index=False)
    pd.DataFrame(files).to_csv(RESULTS_DIR / f"layers_{n_layers}_per_file.csv", index=False)
    meta = dict(family=n_layers, fingerprint=data["fingerprint"], n_features=data["n_features"],
                settings={k: (list(v) if isinstance(v, tuple) else v) for k, v in s.items()},
                variants={k: dict(hidden=list(v["hidden"]), why=v["why"]) for k, v in variants.items()},
                n_normal_test=len(data["X_ntest"]), attack_files=[f"{c}/{n}" for c, n in data["attacks"]],
                device=str(device))
    (RESULTS_DIR / f"layers_{n_layers}_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nSaved results/layers_{n_layers}_runs.csv, _per_file.csv, _meta.json")
    print("Next: run the other families, then  python compare_all.py")