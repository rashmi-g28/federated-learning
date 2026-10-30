"""
DIAGNOSE -- why do some attack files fall BELOW the normal reconstruction-error range?

    python diagnose.py

Reads results/fl_global_3C.pt (saved by the federated run) and the partitions in data-dir. Read-only: changes nothing.
Prints (1) reconstruction-error percentiles for normal traffic and for every attack file,
       (2) how extreme the feature ranges used for scaling are,
       (3) for the failing attack files, which features differ most from normal traffic (scaled units).
"""
import re
from pathlib import Path

import numpy as np
import torch

from fl_iot_ae import task as T


def cfg_value(key):
    txt = (Path(__file__).resolve().parent / "pyproject.toml").read_text()
    m = re.search(rf'^{key}\s*=\s*"(.*)"\s*$', txt, flags=re.M)
    return m.group(1) if m else ""


def pct(x):
    return np.percentile(x, [1, 50, 90, 99, 99.9])


def main():
    ddir = Path(cfg_value("data-dir"))
    rdir = Path(cfg_value("results-dir") or Path(__file__).resolve().parent / "results")
    ck = torch.load(rdir / "fl_global_3C.pt", weights_only=False)
    smin, smax, cols = np.asarray(ck["smin"]), np.asarray(ck["smax"]), ck.get("columns")
    meta = T.load_meta(ddir)
    model = T.build_model(meta["n_features"])
    model.load_state_dict(ck["state_dict"])
    dev = T.pick_device()
    test = T.load_server_test(ddir)

    Xn = T.scale(test["normal"], smin, smax)
    re_n = T.reconstruction_error(model, Xn, dev)
    print("=" * 96)
    print("1) RECONSTRUCTION ERROR PERCENTILES  (attack files with a median BELOW the normal 99th pct are missed)")
    print("=" * 96)
    print(f"{'set':<22}{'n':>8}{'p1':>12}{'median':>12}{'p90':>12}{'p99':>12}{'p99.9':>12}")
    p = pct(re_n)
    print(f"{'NORMAL (all devices)':<22}{len(re_n):>8}" + "".join(f"{v:>12.2e}" for v in p))
    ATT = {}
    for k, v in test["attacks"].items():
        X = T.scale(v, smin, smax)
        r = T.reconstruction_error(model, X, dev)
        ATT[k] = (X, r)
        p = pct(r)
        flag = "   <-- below normal p99" if np.median(r) < np.percentile(re_n, 99) else ""
        print(f"{k:<22}{len(r):>8}" + "".join(f"{v_:>12.2e}" for v_ in p) + flag)

    print("\n" + "=" * 96)
    print("2) SCALING RANGES  (max - min per feature, from the federated min/max)")
    print("=" * 96)
    rng = smax - smin
    print(f"features: {len(rng)} | range > 1e3: {(rng > 1e3).sum()} | > 1e6: {(rng > 1e6).sum()} | > 1e12: {(rng > 1e12).sum()}")
    big = np.argsort(-rng)[:8]
    for i in big:
        nm = cols[i] if cols else str(i)
        print(f"   {nm:<34} min {smin[i]:>14.4g}  max {smax[i]:>14.4g}")

    print("\n" + "=" * 96)
    print("3) FAILING FILES: features that differ most from normal (scaled mean, attack vs normal)")
    print("=" * 96)
    mu_n = Xn.mean(axis=0)
    worst = sorted(ATT, key=lambda k: np.median(ATT[k][1]) - np.percentile(re_n, 99))[:2]
    for k in [w for w in worst if np.median(ATT[w][1]) < np.percentile(re_n, 99)] or worst[:1]:
        X, r = ATT[k]
        d = np.abs(X.mean(axis=0) - mu_n)
        print(f"\n{k}: RE median {np.median(r):.2e} vs normal median {np.median(re_n):.2e}; "
              f"share of RE variance among records: std/mean = {r.std() / max(r.mean(), 1e-12):.2f}")
        for i in np.argsort(-d)[:8]:
            nm = cols[i] if cols else str(i)
            print(f"   {nm:<34} attack mean {X[:, i].mean():>10.4f}   normal mean {mu_n[i]:>10.4f}   range {rng[i]:.3g}")
        print(f"   records with ALL scaled features within [0,1]: {((X >= 0) & (X <= 1)).all(axis=1).mean():.1%}  "
              f"(normal: {((Xn >= 0) & (Xn <= 1)).all(axis=1).mean():.1%})")

    print("\n" + "=" * 96)
    print("4) IS THE ATTACK VECTOR ALSO PRESENT IN NORMAL TRAFFIC?  (squared distance in scaled space, mean over features)")
    print("=" * 96)
    flagged = [k for k in ATT if np.median(ATT[k][1]) < np.percentile(re_n, 99)]
    n_cl = int(meta["n_clients"])
    normals = {f"client {i} train": T.scale(T.load_client(ddir, i, "train"), smin, smax) for i in range(n_cl)}
    normals["server normal test"] = Xn
    for k in flagged:
        X, r = ATT[k]
        v = np.median(X, axis=0)
        print(f"\n{k}: distinct vectors among attack records (rounded 1e-6): "
              f"{len(np.unique(np.round(X, 6), axis=0))} of {len(X)}")
        print(f"   {'normal set':<22}{'records':>9}{'dist<1e-6':>12}{'dist<1e-4':>12}{'dist<1e-3':>12}")
        tot = [0, 0, 0, 0]
        for nm, Xs in normals.items():
            d = ((Xs - v) ** 2).mean(axis=1)
            c = [(d < 1e-6).sum(), (d < 1e-4).sum(), (d < 1e-3).sum()]
            print(f"   {nm:<22}{len(Xs):>9}{c[0]:>12}{c[1]:>12}{c[2]:>12}")
            if nm != "server normal test":
                tot = [tot[0] + len(Xs)] + [tot[i + 1] + c[i] for i in range(3)]
        print(f"   {'ALL client train':<22}{tot[0]:>9}{tot[1]:>12}{tot[2]:>12}{tot[3]:>12}")


if __name__ == "__main__":
    main()