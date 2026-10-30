"""
Replication of: Catillo, Pecchia, Villano -- "A Deep Learning Method for Lightweight and
Cross-Device IoT Botnet Detection", Appl. Sci. 2023, 13, 837  (all-in-one autoencoder, N-BaIoT).

Folder layout expected next to this script:
    normal/                       *.csv  (benign traffic, one or more devices)
    attacks/mirai_attacks/        *.csv
    attacks/gafgyt_attacks/       *.csv

What follows the paper exactly (Sections 3-5):
    * AE architecture (Table 5):  N-64-24-6-24-64-N, ReLU hidden layers, Tanh output, dense layers
    * Loss / RE:                  mean squared error over the N features   (Eq. 3)
    * Optimiser / training:       RMSprop, lr=1e-4, batch 2048, 100 epochs  (Sec. 4.2.2)
    * Semi-supervised:            AE is trained on NORMAL records only
    * Threshold selection:        10% disjoint "threshold set" of the training data ->
                                  Isolation Forest -> RE_IN / RE_OUT -> start at max(RE_OUT) and
                                  lower until inliers/outliers are balanced      (Sec. 3.3)
    * Metrics:                    Recall, Precision, FPR, F1                    (Eq. 4-5)
    * Record order is preserved when splitting (temporal windows)               (Sec. 4.1)

Adapted to YOUR request:
    * 80% of the normal data -> training (of which 10% is the threshold set, 90% trains the AE)
    * 20% of the normal data + the Mirai folder + the Gafgyt folder -> test

Things the paper does NOT state (my choices -- change them in SETTINGS if you want):
    * Feature scaling: MinMax fitted on the AE-training normal data only (fits the Tanh output)
    * Isolation Forest hyper-parameters: scikit-learn defaults, contamination=0.01
    * The exact "equilibrium" rule: threshold maximising (#inliers below + #outliers above); see THRESHOLD_RULE
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn

# ============================================================
# SETTINGS
# ============================================================
BASE_PATH = Path(__file__).resolve().parent
NORMAL_DIR = BASE_PATH / "normal"
ATTACK_DIRS = {
    "Mirai attacks": BASE_PATH / "attacks" / "mirai_attacks",
    "Gafgyt attacks": BASE_PATH / "attacks" / "gafgyt_attacks",
}

# ---- data split (your request) ----
TRAIN_FRAC = 0.80            # normal data used for training; remaining 20% = normal test set
THRESHOLD_FRAC = 0.10        # paper: 10% of the training set is the disjoint "threshold set"
SHUFFLE_SPLIT = False        # paper preserves record order (temporal windows) -> False
VAL_FRAC = 0.05              # slice of the AE-training data used only to monitor validation loss

# ---- paper's autoencoder / training settings (Table 5, Sec. 4.2.2) ----
HIDDEN_LAYERS = (64, 24, 6, 24, 64)
EPOCHS = 100
BATCH_SIZE = 2048
LEARNING_RATE = 1e-4

# ---- my choices where the paper is silent ----
SCALER = "minmax"            # "minmax" or "standard"; fitted on AE-training normal data only
IF_CONTAMINATION = 0.01      # Isolation Forest contamination (paper does not state it; it controls FPR -> tune on the
                             # threshold set / validation data, NEVER on the test set. "auto" flags far too many points)
THRESHOLD_RULE = "count"     # "count": maximise #inliers-below + #outliers-above (inliers dominate -> low FPR)
                             # "balanced": weight inlier/outlier groups equally (more aggressive, higher FPR)
MAX_ATTACK_PER_FILE = 200_000  # None = use every attack record (paper does). Cap only for RAM/speed.
ALLOW_POSITIONAL_ALIGN = False # True = if attack CSV headers differ but the column COUNT matches,
                               # assign normal-file column names by position (use with care)

RANDOM_STATE = 42
LABEL_LIKE = {"label", "class", "attack", "target", "y", "unnamed: 0"}


# ============================================================
# AUTOENCODER  (paper, Table 5)
# ============================================================
class AllInOneAE(nn.Module):
    """N-64-24-6-24-64-N, ReLU in hidden layers, Tanh at the output."""

    def __init__(self, n_features, hidden=HIDDEN_LAYERS):
        super().__init__()
        layers, prev = [], n_features
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, n_features), nn.Tanh()]
        self.net = nn.Sequential(*layers)
        # Keras defaults used by the authors: glorot-uniform weights, zero biases
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


# ============================================================
# DATA HELPERS
# ============================================================
def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    df = df[[c for c in df.columns if c.lower() not in LABEL_LIKE]]
    df = df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    before = len(df)
    df = df.dropna(axis=0, how="any").reset_index(drop=True)   # drop bad rows; never impute
    if len(df) < before:
        print(f"   [{path.name}] dropped {before - len(df)} rows with NaN/inf")
    return df


def align_columns(df: pd.DataFrame, ref_cols, name: str) -> pd.DataFrame:
    """Return df with exactly the reference columns. Raises instead of silently filling."""
    missing = [c for c in ref_cols if c not in df.columns]
    if missing:
        if ALLOW_POSITIONAL_ALIGN and df.shape[1] == len(ref_cols):
            print(f"   WARNING [{name}]: headers differ from normal data -> assigning by position")
            df = df.copy()
            df.columns = list(ref_cols)
        else:
            raise ValueError(
                f"\n[{name}] column mismatch with the normal data: {len(missing)}/{len(ref_cols)} "
                f"normal-data columns are missing.\n"
                f"  normal columns (first 5): {list(ref_cols)[:5]}\n"
                f"  this file's columns (first 5): {list(df.columns)[:5]}\n"
                f"This is exactly what makes every attack look 'normal' if it is silently filled with "
                f"medians. Fix the headers, or set ALLOW_POSITIONAL_ALIGN = True if the column ORDER "
                f"is identical."
            )
    return df[list(ref_cols)]


def split_indices(n, rng):
    """-> (fit_idx, thr_idx, test_idx): 80% train (90% fit / 10% threshold set), 20% test."""
    idx = rng.permutation(n) if SHUFFLE_SPLIT else np.arange(n)
    n_train = int(round(n * TRAIN_FRAC))
    train, test = idx[:n_train], idx[n_train:]
    n_thr = int(round(len(train) * THRESHOLD_FRAC))
    return train[: len(train) - n_thr], train[len(train) - n_thr:], test


def make_scaler():
    return MinMaxScaler() if SCALER == "minmax" else StandardScaler()


@torch.no_grad()
def reconstruction_error(model, X, device, batch=65536):
    """RE = mean_i (z_i - x_i)^2 per record  (paper Eq. 3)."""
    model.eval()
    out = np.empty(len(X), dtype=np.float64)
    for s in range(0, len(X), batch):
        xb = torch.from_numpy(X[s:s + batch]).to(device)
        out[s:s + batch] = ((model(xb) - xb) ** 2).mean(dim=1).double().cpu().numpy()
    return out


# ============================================================
# THRESHOLD SELECTION  (paper, Sec. 3.3 / Fig. 2)
# ============================================================
def select_threshold(re_in, re_out):
    """Start at max(RE_OUT) and lower it until inliers (below) and outliers (above) are in
    equilibrium, i.e. maximise 0.5*frac(inliers<=t) + 0.5*frac(outliers>t)."""
    if len(re_out) == 0:
        print("   WARNING: Isolation Forest found no outliers -> using max(RE_IN) as threshold")
        return float(re_in.max())
    s_in, s_out = np.sort(re_in), np.sort(re_out)
    cand = np.unique(np.concatenate([s_in, s_out]))
    cand = cand[cand <= s_out[-1]][::-1]                     # descending, starting at max(RE_OUT)
    f_in = np.searchsorted(s_in, cand, side="right") / len(s_in)
    f_out = 1.0 - np.searchsorted(s_out, cand, side="right") / len(s_out)
    if THRESHOLD_RULE == "balanced":
        score = 0.5 * f_in + 0.5 * f_out
    else:                                                     # "count": plain agreement with IF labels
        score = (len(s_in) * f_in + len(s_out) * f_out) / (len(s_in) + len(s_out))
    return float(cand[int(np.argmax(score))])                # first max = highest threshold on ties


# ============================================================
# METRICS  (paper, Eq. 4-5)
# ============================================================
def evaluate(re_normal, re_attack, thr):
    tp = int((re_attack > thr).sum()); fn = len(re_attack) - tp
    fp = int((re_normal > thr).sum()); tn = len(re_normal) - fp
    R = tp / (tp + fn) if tp + fn else 0.0
    P = tp / (tp + fp) if tp + fp else 0.0
    FPR = fp / (fp + tn) if fp + tn else 0.0
    F1 = 2 * P * R / (P + R) if P + R else 0.0
    y = np.r_[np.zeros(len(re_normal)), np.ones(len(re_attack))]
    auc = roc_auc_score(y, np.r_[re_normal, re_attack])
    return dict(Recall=R, Precision=P, FPR=FPR, F1=F1, AUC=auc, TP=tp, FN=fn, FP=fp, TN=tn)


# ============================================================
# MAIN
# ============================================================
def main():
    torch.manual_seed(RANDOM_STATE)
    rng = np.random.default_rng(RANDOM_STATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---------------- load NORMAL data, split per file (per device) ----------------
    normal_files = sorted(NORMAL_DIR.rglob("*.csv"))
    if not normal_files:
        raise FileNotFoundError(f"No CSV files in {NORMAL_DIR}")
    print(f"\nLoading NORMAL data ({len(normal_files)} files)...")
    ref_cols, fit_parts, thr_parts, test_parts = None, [], [], []
    for f in normal_files:
        df = load_csv(f)
        if ref_cols is None:
            ref_cols = list(df.columns)
        df = align_columns(df, ref_cols, f"normal/{f.name}")
        fit_i, thr_i, test_i = split_indices(len(df), rng)
        fit_parts.append(df.iloc[fit_i]); thr_parts.append(df.iloc[thr_i]); test_parts.append(df.iloc[test_i])
        print(f"   {f.relative_to(NORMAL_DIR)}: {len(df)} rows -> fit {len(fit_i)} | threshold-set {len(thr_i)} | test {len(test_i)}")

    X_fit = pd.concat(fit_parts, ignore_index=True).to_numpy(dtype=np.float64)
    X_thr = pd.concat(thr_parts, ignore_index=True).to_numpy(dtype=np.float64)
    X_ntest = pd.concat(test_parts, ignore_index=True).to_numpy(dtype=np.float64)
    n_features = X_fit.shape[1]
    print(f"Features (N) = {n_features}   |  AE-fit {len(X_fit)}  threshold-set {len(X_thr)}  normal-test {len(X_ntest)}")

    scaler = make_scaler().fit(X_fit)                          # fitted on AE-training normal data ONLY
    X_fit = scaler.transform(X_fit).astype(np.float32)
    X_thr = scaler.transform(X_thr).astype(np.float32)
    X_ntest = scaler.transform(X_ntest).astype(np.float32)

    # small validation slice (monitoring only)
    perm = rng.permutation(len(X_fit))
    n_val = int(len(X_fit) * VAL_FRAC)
    X_val, X_train = X_fit[perm[:n_val]], X_fit[perm[n_val:]]

    # ---------------- train the AE on NORMAL data only ----------------
    model = AllInOneAE(n_features).to(device)
    print("\n" + str(model))
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    # Keras RMSprop defaults: rho=0.9, epsilon=1e-7
    opt = torch.optim.RMSprop(model.parameters(), lr=LEARNING_RATE, alpha=0.9, eps=1e-7)
    loss_fn = nn.MSELoss()
    Xt = torch.from_numpy(X_train).to(device)
    Xv = torch.from_numpy(X_val).to(device)

    print(f"\nTraining on NORMAL data only: {EPOCHS} epochs, batch {BATCH_SIZE}, RMSprop lr={LEARNING_RATE}")
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = torch.randperm(len(Xt), device=device)
        run = 0.0
        for s in range(0, len(Xt), BATCH_SIZE):
            xb = Xt[order[s:s + BATCH_SIZE]]
            loss = loss_fn(model(xb), xb)
            opt.zero_grad(); loss.backward(); opt.step()
            run += loss.item() * len(xb)
        if epoch == 1 or epoch % 10 == 0 or epoch == EPOCHS:
            model.eval()
            with torch.no_grad():
                vl = loss_fn(model(Xv), Xv).item() if n_val else float("nan")
            print(f"   epoch {epoch:3d}/{EPOCHS}  train MSE {run / len(Xt):.6f}  val MSE {vl:.6f}")
    print(f"Training time: {time.time() - t0:.1f}s")

    # ---------------- threshold selection (Isolation Forest on the threshold set) ----------------
    print("\nSelecting threshold (10% threshold set + Isolation Forest)...")
    iso = IsolationForest(contamination=IF_CONTAMINATION, random_state=RANDOM_STATE, n_jobs=-1).fit(X_thr)
    is_inlier = iso.predict(X_thr) == 1
    re_thr = reconstruction_error(model, X_thr, device)
    re_in, re_out = re_thr[is_inlier], re_thr[~is_inlier]
    threshold = select_threshold(re_in, re_out)
    print(f"   inliers {len(re_in)} | outliers {len(re_out)}")
    print(f"   max(RE_OUT) = {re_out.max() if len(re_out) else float('nan'):.6g}   ->   THRESHOLD = {threshold:.6g}")

    # ---------------- test: normal 20% ----------------
    re_normal = reconstruction_error(model, X_ntest, device)
    print(f"\nNormal test set: {len(re_normal)} records | median RE {np.median(re_normal):.3g} | "
          f"FPR {np.mean(re_normal > threshold):.4f}")

    # ---------------- test: attack folders (each file scored independently) ----------------
    rows, cat_errors, file_errors, file_medians = [], {}, {}, {}
    for category, directory in ATTACK_DIRS.items():
        files = sorted(directory.rglob("*.csv"))
        if not files:
            raise FileNotFoundError(f"No CSV files for {category}: {directory}")
        print(f"\nLoading {category} ({len(files)} files)...")
        errs = []
        for f in files:
            df = load_csv(f)
            df = align_columns(df, ref_cols, f"{category}/{f.name}")       # raises on mismatch
            if MAX_ATTACK_PER_FILE and len(df) > MAX_ATTACK_PER_FILE:
                keep = np.sort(rng.choice(len(df), MAX_ATTACK_PER_FILE, replace=False))
                df = df.iloc[keep]
            Xa = scaler.transform(df.to_numpy(dtype=np.float64)).astype(np.float32)
            re_a = reconstruction_error(model, Xa, device)
            errs.append(re_a)
            file_errors[(category, f.stem)] = re_a
            rows.append(dict(Set=f"{category}: {f.stem}", Attack_records=len(re_a),
                             median_RE=np.median(re_a), **evaluate(re_normal, re_a, threshold)))
        cat_errors[category] = np.concatenate(errs)
        rows.append(dict(Set=f"{category}: ALL", Attack_records=len(cat_errors[category]),
                         median_RE=np.median(cat_errors[category]),
                         **evaluate(re_normal, cat_errors[category], threshold)))
    all_att = np.concatenate(list(cat_errors.values()))
    rows.append(dict(Set="ALL ATTACKS", Attack_records=len(all_att), median_RE=np.median(all_att),
                     **evaluate(re_normal, all_att, threshold)))

    # ---------------- report (paper Table 7 style) ----------------
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 200, "display.max_columns", 20, "display.max_colwidth", 40)
    show = res[["Set", "Attack_records", "Recall", "Precision", "FPR", "F1", "AUC", "median_RE"]].copy()
    for c in ["Recall", "Precision", "FPR", "F1", "AUC"]:
        show[c] = show[c].map(lambda v: f"{v:.4f}")
    show["median_RE"] = show["median_RE"].map(lambda v: f"{v:.3g}")
    print("\n" + "=" * 100)
    print(f"RESULTS  (threshold = {threshold:.6g}; normal-test median RE = {np.median(re_normal):.3g}, "
          f"normal-test records = {len(re_normal)})")
    print("=" * 100)
    print(show.to_string(index=False))
    print("\nPaper reference (all-in-one, Table 7): Recall 0.9994-0.9997, Precision 0.9999-1.0, "
          "FPR 0.0-0.0071, F1 0.9996-0.9998")

    if all(np.median(e) <= threshold for e in file_errors.values()):
        print("\n!! SANITY WARNING: the median attack RE is below the threshold for EVERY attack file. "
              "Check that the attack CSVs are really attack traffic with the same 115 features/units.")

    res.to_csv(BASE_PATH / "results_all_in_one_ae.csv", index=False)

    # ---------------- Fig. 3-style plot: RE of every test point ----------------
    pts = [("Normal", re_normal)] + [(c, e) for c, e in cat_errors.items()]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    start = 0
    colors = {"Normal": "tab:blue", "Mirai attacks": "tab:red", "Gafgyt attacks": "tab:gray"}
    for name, e in pts:
        step = max(1, len(e) // 60000)                    # thin the plot, not the data
        x = np.arange(start, start + len(e))[::step]
        ax.scatter(x, np.clip(e[::step], 1e-12, None), s=2, c=colors.get(name, "k"),
                   marker="o" if name == "Normal" else "v", alpha=0.6, label=name, rasterized=True)
        start += len(e)
    ax.axhline(threshold, color="red", ls="--", lw=1.5, label=f"Threshold ({threshold:.3g})")
    ax.set_yscale("log"); ax.set_xlabel("data points (test set)"); ax.set_ylabel("RE")
    ax.set_title("Reconstruction error of the test set (all-in-one autoencoder)")
    ax.legend(markerscale=4)
    out = BASE_PATH / "ae_reconstruction_error.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out}\nSaved: {BASE_PATH / 'results_all_in_one_ae.csv'}")

    torch.save(dict(state_dict=model.state_dict(), threshold=threshold, columns=ref_cols,
                    scaler=SCALER, scaler_params=[a.tolist() for a in
                    (([scaler.data_min_, scaler.data_max_] if SCALER == "minmax" else [scaler.mean_, scaler.scale_]))]),
               BASE_PATH / "ae_all_in_one.pt")


if __name__ == "__main__":
    main()