"""
STEP 1 -- turn your CSV folders into federated partitions (one client per IoT device).

    python prepare_partitions.py --data-root C:\\Users\\you\\Desktop\\major-1\\comparision --out ..\\fl_data

Expected under --data-root (same layout as the layer study):
    normal/*.csv                 one file per device  -> ONE FEDERATED CLIENT PER FILE
    attacks/mirai_attacks/*.csv
    attacks/gafgyt_attacks/*.csv

Writes under --out (default ../fl_data, i.e. OUTSIDE the Flower project so it is never packaged into the app):
    client_<i>/train.npy         raw (unscaled) normal records the client trains on          (72% of its device data)
    client_<i>/thr.npy           raw normal 'threshold set' used only for threshold calibration (8%)
    server/test_normal.npy       held-out normal records, 20% of every device   -> evaluation only
    server/attack__<Cat>__<file>.npy   attack records (capped per file)           -> evaluation only
    meta.json                    columns, client list, sizes
and writes the absolute data-dir / results-dir into pyproject.toml.

Split rule = the layer study's rule: chronological per device; first 80% = train side (last 10% of that = threshold
set), final 20% = test. Attack records are NEVER used for training or for choosing thresholds.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

LABEL_LIKE = {"label", "class", "attack", "target", "y", "unnamed: 0"}
ATTACK_SUBDIRS = {"Mirai": "attacks/mirai_attacks", "Gafgyt": "attacks/gafgyt_attacks"}


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    df = df[[c for c in df.columns if c.lower() not in LABEL_LIKE]]
    df = df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return df.dropna(axis=0, how="any").reset_index(drop=True)


def align(df: pd.DataFrame, ref_cols, name: str) -> pd.DataFrame:
    missing = [c for c in ref_cols if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] {len(missing)}/{len(ref_cols)} normal-data columns are missing "
                         f"(e.g. {missing[:3]}). Fix the headers first; never fill attack data with medians.")
    return df[list(ref_cols)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "fl_data"))
    ap.add_argument("--max-attack-per-file", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-pyproject", action="store_true", help="do not edit pyproject.toml")
    a = ap.parse_args()

    root, out = Path(a.data_root), Path(a.out).resolve()
    rng = np.random.default_rng(a.seed)
    files = sorted((root / "normal").rglob("*.csv"))
    if not files:
        raise SystemExit(f"No CSV files in {root / 'normal'}")
    (out / "server").mkdir(parents=True, exist_ok=True)

    ref_cols, test_parts, clients = None, [], []
    print(f"Partitioning {len(files)} normal files -> {len(files)} federated clients")
    for i, f in enumerate(files):
        df = load_csv(f)
        ref_cols = ref_cols or list(df.columns)
        X = align(df, ref_cols, f"normal/{f.name}").to_numpy(np.float32)
        n = len(X)
        n80 = int(round(n * 0.8))
        n_thr = int(round(n80 * 0.1))
        cdir = out / f"client_{i}"
        cdir.mkdir(exist_ok=True)
        np.save(cdir / "train.npy", X[: n80 - n_thr])
        np.save(cdir / "thr.npy", X[n80 - n_thr: n80])
        test_parts.append(X[n80:])
        clients.append(dict(id=i, file=str(f.relative_to(root)), n_train=n80 - n_thr, n_thr=n_thr, n_test=n - n80))
        print(f"   client {i}: {f.name}: train {n80 - n_thr} | threshold-set {n_thr} | test(server) {n - n80}")
    np.save(out / "server" / "test_normal.npy", np.concatenate(test_parts))

    n_att = 0
    for cat, sub in ATTACK_SUBDIRS.items():
        afiles = sorted((root / sub).rglob("*.csv"))
        if not afiles:
            raise SystemExit(f"No CSV files in {root / sub}")
        for f in afiles:
            df = align(load_csv(f), ref_cols, f"{cat}/{f.name}")
            if len(df) > a.max_attack_per_file:
                df = df.iloc[np.sort(rng.choice(len(df), a.max_attack_per_file, replace=False))]
            np.save(out / "server" / f"attack__{cat}__{f.stem}.npy", df.to_numpy(np.float32))
            n_att += len(df)
            print(f"   attack {cat}/{f.stem}: {len(df)} records")

    meta = dict(n_features=len(ref_cols), columns=ref_cols, n_clients=len(files), clients=clients,
                n_test_normal=int(sum(len(t) for t in test_parts)), n_attack=int(n_att))
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    if not a.no_pyproject:
        pp = Path(__file__).resolve().parent / "pyproject.toml"
        txt = pp.read_text()
        res = (Path(__file__).resolve().parent / "results").as_posix()
        txt = re.sub(r'^data-dir\s*=.*$', f'data-dir = "{out.as_posix()}"', txt, flags=re.M)
        txt = re.sub(r'^results-dir\s*=.*$', f'results-dir = "{res}"', txt, flags=re.M)
        pp.write_text(txt)
        print(f"\nWrote data-dir and results-dir into {pp}")

    print(f"\nDONE. {len(files)} clients, {meta['n_features']} features.")
    print(f"Use exactly this number of simulated nodes:  --federation-config \"num-supernodes={len(files)}\"")


if __name__ == "__main__":
    main()
