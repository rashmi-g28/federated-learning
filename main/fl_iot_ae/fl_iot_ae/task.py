"""
fl_iot_ae.task -- everything shared by the Flower clients, the Flower server and the baselines:
model (variant 3C), data access, local training, scoring, evaluation, efficiency helpers.

Design rules that keep the project honest:
  * Clients only ever read THEIR OWN partition (client_<id>/train.npy, thr.npy).
  * The server only reads server/ (labelled hold-out normal + attack files) and ONLY to measure results;
    it never trains on them and never sets thresholds from them.
  * Feature scaling parameters (global min / max) are built from per-client min/max, not from pooled raw data.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

HIDDEN = (32, 6, 32)                 # variant 3C from the layer study  ->  115-32-6-32-115
BUDGETS = (0.01, 0.005, 0.002, 0.001)  # per-record false-alarm budgets used to set thresholds
WINDOWS = (50, 100, 200)             # window sizes (consecutive records) for the window-level rule
PROJECT_DIR = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------------- paths / config
def data_dir(run_config) -> Path:
    p = os.environ.get("FL_DATA_DIR") or (run_config.get("data-dir") if run_config else "")
    if not p:
        raise RuntimeError("data-dir is empty. Run  python prepare_partitions.py  first (it writes data-dir into "
                           "pyproject.toml), or set the FL_DATA_DIR environment variable.")
    return Path(p)


def results_dir(run_config) -> Path:
    p = os.environ.get("FL_RESULTS_DIR") or (run_config.get("results-dir") if run_config else "") \
        or str(PROJECT_DIR / "results")
    d = Path(p)
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_meta(ddir: Path) -> dict:
    return json.loads((ddir / "meta.json").read_text())


def load_client(ddir: Path, pid: int, name: str) -> np.ndarray:
    return np.load(ddir / f"client_{pid}" / f"{name}.npy")


_TEST_CACHE: dict = {}


def load_server_test(ddir: Path) -> dict:
    key = str(ddir)
    if key not in _TEST_CACHE:
        sdir = ddir / "server"
        attacks = {p.stem[len("attack__"):]: np.load(p) for p in sorted(sdir.glob("attack__*.npy"))}
        _TEST_CACHE[key] = dict(normal=np.load(sdir / "test_normal.npy"), attacks=attacks)
    return _TEST_CACHE[key]


# ----------------------------------------------------------------------------- model
class AE(nn.Module):
    """N -> hidden... -> N ; ReLU hidden layers, Tanh output, Glorot init (as in the paper / layer study)."""

    def __init__(self, n_features: int, hidden=HIDDEN):
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


def build_model(n_features: int, hidden=HIDDEN) -> AE:
    return AE(n_features, hidden)


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def count_macs(model) -> int:
    return sum(m.in_features * m.out_features for m in model.modules() if isinstance(m, nn.Linear))


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------- scaling
def scale(X: np.ndarray, smin, smax) -> np.ndarray:
    """Min-max scaling with the GLOBAL (federated) min / max. Values outside [0, 1] are kept on purpose."""
    smin = np.asarray(smin, dtype=np.float64)
    rng = np.asarray(smax, dtype=np.float64) - smin
    rng[rng == 0] = 1.0
    return ((X.astype(np.float64) - smin) / rng).astype(np.float32)


# ----------------------------------------------------------------------------- training / scoring
def train_local(model, X: np.ndarray, epochs: int, batch_size: int, lr: float, device, seed: int = 0):
    """Local training on one client. Returns (mean loss, wall-clock seconds)."""
    torch.manual_seed(seed)
    model.to(device).train()
    opt = torch.optim.RMSprop(model.parameters(), lr=lr, alpha=0.9, eps=1e-7)  # Keras defaults, as in the paper
    lossf = nn.MSELoss()
    Xt = torch.from_numpy(X).to(device)
    total, seen, t0 = 0.0, 0, time.time()
    for _ in range(epochs):
        order = torch.randperm(len(Xt), device=device)
        for i in range(0, len(Xt), batch_size):
            xb = Xt[order[i:i + batch_size]]
            loss = lossf(model(xb), xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
            seen += len(xb)
    return total / max(seen, 1), time.time() - t0


@torch.no_grad()
def reconstruction_error(model, X: np.ndarray, device, batch: int = 65536) -> np.ndarray:
    """RE = mean_i (z_i - x_i)^2 per record (paper Eq. 3)."""
    model.to(device).eval()
    out = np.empty(len(X), dtype=np.float64)
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch]).to(device)
        out[i:i + batch] = ((model(xb) - xb) ** 2).mean(dim=1).double().cpu().numpy()
    return out


# ----------------------------------------------------------------------------- evaluation (server-side / baselines)
def _op_point(thr: float, re_n: np.ndarray, re_att: dict) -> dict:
    fp = int((re_n > thr).sum())
    per = {k: float((v > thr).mean()) for k, v in re_att.items()}
    tp = sum(int((v > thr).sum()) for v in re_att.values())
    n_att = sum(len(v) for v in re_att.values())
    R = tp / n_att
    P = tp / (tp + fp) if tp + fp else 0.0
    return dict(threshold=float(thr), fpr=fp / len(re_n), recall=R, precision=P,
                f1=2 * P * R / (P + R) if P + R else 0.0,
                worst_file_recall=min(per.values()), recall_per_file=per)


def evaluate_global(model, smin, smax, test: dict, thresholds: dict | None, device) -> dict:
    """Threshold-free AUC (pooled and per attack file) + operating points for the given thresholds."""
    re_n = reconstruction_error(model, scale(test["normal"], smin, smax), device)
    re_att = {k: reconstruction_error(model, scale(v, smin, smax), device) for k, v in test["attacks"].items()}
    y = np.r_[np.zeros(len(re_n)), np.ones(sum(len(v) for v in re_att.values()))]
    s = np.r_[re_n, *re_att.values()]
    per_auc = {k: float(roc_auc_score(np.r_[np.zeros(len(re_n)), np.ones(len(v))], np.r_[re_n, v]))
               for k, v in re_att.items()}
    out = dict(auc_all=float(roc_auc_score(y, s)), auc_per_file=per_auc, auc_worst_file=min(per_auc.values()),
               normal_median_re=float(np.median(re_n)), n_normal=int(len(re_n)),
               n_attack=int(sum(len(v) for v in re_att.values())), operating_points={})
    for name, thr in (thresholds or {}).items():
        out["operating_points"][name] = _op_point(thr, re_n, re_att)
    return out


# ----------------------------------------------------------------------------- window-level rule
def record_ids(Xs: np.ndarray) -> np.ndarray:
    """Integer id per record; equal ids == equal records after scaling (rounded to 1e-6)."""
    R = np.ascontiguousarray(np.round(np.clip(Xs.astype(np.float64), -1e9, 1e9) * 1e6).astype(np.int64))
    v = R.view(np.dtype((np.void, R.dtype.itemsize * R.shape[1]))).ravel()
    return np.unique(v, return_inverse=True)[1].ravel()


def window_stats(ids: np.ndarray, re: np.ndarray, W: int, stride: int):
    """(repeat rate, median RE) for windows of W consecutive records, moving by `stride`.
    repeat rate = 1 - distinct records / W  (0 = all different, ~1 = the same record repeated)."""
    s, m = [], []
    for a in range(0, len(ids) - W + 1, stride):
        s.append(1.0 - len(np.unique(ids[a:a + W])) / W)
        m.append(float(np.median(re[a:a + W])))
    return np.asarray(s), np.asarray(m)


def window_evaluate(model, smin, smax, test: dict, chunk_sizes, taus: dict, device) -> dict:
    """Window-level evaluation. taus = {W: (tau_rep, tau_re)} come from the clients' normal data only.
    Normal windows are cut inside each device's own held-out chunk (chunk_sizes, in order); attack windows inside each file."""
    Xn = scale(test["normal"], smin, smax)
    re_n, id_n = reconstruction_error(model, Xn, device), record_ids(Xn)
    bounds = np.cumsum([0] + [int(c) for c in chunk_sizes])
    att = {}
    for k, v in test["attacks"].items():
        X = scale(v, smin, smax)
        att[k] = (record_ids(X), reconstruction_error(model, X, device))
    out = {}
    for W, (tau_rep, tau_re) in taus.items():
        parts = [window_stats(id_n[a:b], re_n[a:b], W, W) for a, b in zip(bounds[:-1], bounds[1:])]
        s_t, m_t = np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
        fpr = dict(rep=float((s_t > tau_rep).mean()), re=float((m_t > tau_re).mean()),
                   any=float(((s_t > tau_rep) | (m_t > tau_re)).mean()))
        per = {}
        for k, (i, r) in att.items():
            s, m = window_stats(i, r, W, W)
            per[k] = dict(rep=float((s > tau_rep).mean()), re=float((m > tau_re).mean()),
                          any=float(((s > tau_rep) | (m > tau_re)).mean()), n=int(len(s)))
        rec = {rule: sum(v[rule] * v["n"] for v in per.values()) / sum(v["n"] for v in per.values())
               for rule in ("rep", "re", "any")}
        out[int(W)] = dict(tau_rep=float(tau_rep), tau_re=float(tau_re), n_normal_windows=int(len(s_t)),
                           window_fpr=fpr, window_recall=rec, recall_per_file=per,
                           worst_file_recall={r_: min(v[r_] for v in per.values()) for r_ in ("rep", "re", "any")})
    return out


# ----------------------------------------------------------------------------- efficiency helpers
def bench_inference(model, X: np.ndarray) -> dict:
    """CPU inference latency in microseconds / record (large batch and batch size 1)."""
    model = model.to("cpu").eval()
    Xb = torch.from_numpy(X[:20000])
    with torch.no_grad():
        model(Xb[:4096])
        t0 = time.time()
        for i in range(0, len(Xb), 4096):
            model(Xb[i:i + 4096])
        batch_us = (time.time() - t0) / len(Xb) * 1e6
        t0 = time.time()
        for i in range(1000):
            model(Xb[i:i + 1])
        single_us = (time.time() - t0) / 1000 * 1e6
    return dict(us_per_record_batch4096=batch_us, us_per_record_batch1=single_us)


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o
