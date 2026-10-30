"""Runs ONLY the 3-hidden-layer variants (VARIANTS[3] in ae_core.py). Nothing from other depths is touched.

    python run_3_layer.py                      # full run: 100 epochs, 3 seeds
    python run_3_layer.py --quick              # smoke test: 10 epochs, 1 seed (NOT for decisions)
    python run_3_layer.py --data-root /path    # folder containing normal/ and attacks/
"""
import argparse
from pathlib import Path

import ae_core

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--quick", action="store_true", help="10 epochs, 1 seed")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--n-seeds", type=int, default=None)
    a = ap.parse_args()
    if a.quick:
        a.epochs, a.n_seeds = a.epochs or 10, a.n_seeds or 1
    ae_core.run_family(3, a.data_root, epochs=a.epochs, n_seeds=a.n_seeds)