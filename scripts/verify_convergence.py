#!/usr/bin/env python3
"""
Quadrature convergence of the MACE tables (uniform weight).

Compares tables computed with different MGIntegrationConfig quadrature orders.
Variants live in output/convergence/{label}/ (written by compute_matrix.py
with --out-root); "ep16" falls back to the production tables in
output/tables/ when no separate run exists. Each comparison isolates one
quadrature:

  e_panel  ep16, ep32, ep64  vs  ep128
  xi       ep64              vs  ep64_xi96   (xi_order, xi_tail_order 48 -> 96)
  E'       ep64              vs  ep64_epp48  (ep_edge_order, ep_interior_order 24 -> 48)
           ep64_epp48        vs  ep64_epp96  (48 -> 96)
           ep16 / ep16_epp48 vs  ep16_epp48 / ep16_epp96 (same, at the default e_panel order)

sigma errors are relative to |sigma|; dsigma/dT errors are in units of
|sigma|/T. Statistics are over entries above 1e-2 and 1e-6 of their incoming
group's maximum sigma, plus the worst angle- and outgoing-integrated row sum.

Usage:
  python scripts/verify_convergence.py [--temps 0 10 20] [--quantity sigma]
"""

from __future__ import annotations

import argparse

import numpy as np

from compute_matrix import OUTPUT_DIR, ROOT, TEMPERATURES_K, output_path

CONV_DIR = ROOT / "output" / "convergence"
THRESHOLDS = (1e-2, 1e-6)
COMPARISONS = [
    ("e_panel", "ep16", "ep64"),
    ("e_panel", "ep16", "ep128"),
    ("e_panel", "ep32", "ep128"),
    ("e_panel", "ep64", "ep128"),
    ("xi", "ep64", "ep64_xi96"),
    ("E'", "ep64", "ep64_epp48"),
    ("E'", "ep64_epp48", "ep64_epp96"),
    ("E'", "ep16", "ep16_epp48"),
    ("E'", "ep16_epp48", "ep16_epp96"),
]


def load(label: str, quantity: str, tidx: int) -> np.ndarray | None:
    T = TEMPERATURES_K[tidx]
    key = "sigma_matrix" if quantity == "sigma" else "dsigma_dT_matrix"
    paths = [output_path(CONV_DIR / label, "uniform", quantity, tidx, T)]
    if label == "ep16":
        paths.append(output_path(OUTPUT_DIR, "uniform", quantity, tidx, T))
    for path in paths:
        if path.exists():
            return np.load(path)[key]
    return None


def compare(x: np.ndarray, ref: np.ndarray, sigma: np.ndarray, T: float | None) -> str:
    row_max = sigma.max(axis=(1, 2), keepdims=True)
    scale = np.abs(ref) if T is None else np.abs(sigma) / T
    parts = []
    for thr in THRESHOLDS:
        mask = sigma > thr * row_max
        e = np.abs(x - ref)[mask] / scale[mask]
        parts.append(f"{np.median(e):8.1e} {np.percentile(e, 90):8.1e} {np.percentile(e, 99):8.1e} {e.max():8.1e}")
    rows = np.abs(ref.sum(axis=(1, 2)))
    row_scale = rows if T is None else sigma.sum(axis=(1, 2)) / T
    ok = row_scale > 0
    row_err = np.abs(x.sum(axis=(1, 2)) - ref.sum(axis=(1, 2)))[ok] / row_scale[ok]
    return " | ".join(parts) + f" | {row_err.max():8.1e}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--temps", type=int, nargs="+", default=[0, 10, 20])
    parser.add_argument("--quantity", choices=("sigma", "dsigma_dT"), default="sigma")
    args = parser.parse_args()

    header = " | ".join(f"{'p50':>8} {'p90':>8} {'p99':>8} {'max':>8}" for _ in THRESHOLDS)
    print(f"uniform {args.quantity}; entry stats over sigma > "
          + " and > ".join(f"{t:.0e}" for t in THRESHOLDS) + " x row max; last column: row sums")
    for tidx in args.temps:
        T = TEMPERATURES_K[tidx]
        sigma_ref = load("ep128", "sigma", tidx)
        if sigma_ref is None:
            sigma_ref = load("ep64", "sigma", tidx)
        print(f"\nT{tidx:03d} (T = {T:.3e} K)")
        print(f"  {'quadrature':10s} {'variant':10s} {'reference':12s} {header} | {'row sum':>8}")
        for quad, label, ref_label in COMPARISONS:
            x, ref = load(label, args.quantity, tidx), load(ref_label, args.quantity, tidx)
            if x is None or ref is None or sigma_ref is None:
                continue
            scale_T = None if args.quantity == "sigma" else T
            print(f"  {quad:10s} {label:10s} {ref_label:12s} {compare(x, ref, sigma_ref, scale_T)}")


if __name__ == "__main__":
    main()
