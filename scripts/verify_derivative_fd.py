#!/usr/bin/env python3
"""
Check a stored dsigma/dT table against a central difference of sigma.

Computes sigma at T(1 - h) and T(1 + h) with the production sigma
configuration and compares (sigma+ - sigma-) / (2 h T) with the stored
output/tables/{weight}/dsigma_dT table. Errors are measured in units of the
natural derivative scale sigma/T, on entries above 1e-6 of their incoming
group's maximum; row sums (angle- and outgoing-integrated) are checked too.
sigma(T(1 +- h)) uses the production quadrature orders. Results are printed
and written to output/verify/fd_{weight}_T{idx}.json; the exit status is 1
if the worst row-sum error exceeds --max-row-sum-err (a pass/fail gate).

Usage:
  python scripts/verify_derivative_fd.py --weight planck --temperature-index 32
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from compute_matrix import (
    DEFAULT_CAP_X, N_ANGLE_BINS, OUTPUT_DIR, QUADRATURE_ORDERS, ROOT, TEMPERATURES_K, WEIGHTS,
    compute_table, make_boundaries_kev, output_path, production_ep_k_cut, production_orders,
)

VERIFY_DIR = ROOT / "output" / "verify"
SIGNIFICANT = 1e-6


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weight", choices=WEIGHTS, required=True)
    parser.add_argument("--temperature-index", type=int, required=True)
    parser.add_argument("--rel-step", type=float, default=1e-3)
    parser.add_argument("--max-row-sum-err", type=float, default=1e-2,
                        help="Fail if any row-sum error exceeds this (units of sigma/T)")
    args = parser.parse_args()

    tidx, h = args.temperature_index, args.rel_step
    T = TEMPERATURES_K[tidx]
    sigma_path = output_path(OUTPUT_DIR, args.weight, "sigma", tidx, T)
    deriv_path = output_path(OUTPUT_DIR, args.weight, "dsigma_dT", tidx, T)
    sigma = np.load(sigma_path)["sigma_matrix"]
    with np.load(deriv_path) as d:
        deriv = d["dsigma_dT_matrix"]
        deriv_orders = {k: int(d[k]) for k in QUADRATURE_ORDERS if k in d.files}

    orders = production_orders(tidx)
    ep_k_cut = production_ep_k_cut(tidx)
    b = make_boundaries_kev()
    t0 = time.time()
    s_plus, meta = compute_table(args.weight, "sigma", b, N_ANGLE_BINS, T * (1 + h),
                                 cap_x=DEFAULT_CAP_X, orders=orders, ep_k_cut=ep_k_cut)
    s_minus, _ = compute_table(args.weight, "sigma", b, N_ANGLE_BINS, T * (1 - h),
                               cap_x=DEFAULT_CAP_X, orders=orders, ep_k_cut=ep_k_cut)
    elapsed = time.time() - t0
    fd = (s_plus - s_minus) / (2 * h * T)

    row_max = sigma.max(axis=(1, 2), keepdims=True)
    scale = np.maximum(np.abs(sigma), SIGNIFICANT * row_max) / T
    mask = sigma > SIGNIFICANT * row_max
    err = np.abs(fd - deriv) / scale
    rows = sigma.sum(axis=(1, 2)) > 0
    row_err = (np.abs(fd.sum(axis=(1, 2)) - deriv.sum(axis=(1, 2)))[rows]
               / (sigma.sum(axis=(1, 2))[rows] / T))

    worst = np.unravel_index(np.argmax(np.where(mask, err, 0)), err.shape)
    result = {
        "weight": args.weight,
        "temperature_index": tidx,
        "temperature_K": T,
        "rel_step": h,
        "n_significant": int(mask.sum()),
        "max_err": float(err[mask].max()),
        "p99_err": float(np.percentile(err[mask], 99)),
        "median_err": float(np.median(err[mask])),
        "worst_entry": [int(i) for i in worst],
        "max_row_sum_err": float(row_err.max()),
        "median_row_sum_err": float(np.median(row_err)),
        "fd_wall_time_s": elapsed,
        "fd_orders": {k: int(meta[k]) for k in QUADRATURE_ORDERS},
        "dsigma_dT_orders": deriv_orders,
        "max_row_sum_err_threshold": args.max_row_sum_err,
        "pass": bool(row_err.max() <= args.max_row_sum_err),
    }
    VERIFY_DIR.mkdir(parents=True, exist_ok=True)
    out = VERIFY_DIR / f"fd_{args.weight}_T{tidx:03d}.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"Saved: {out}")
    if not result["pass"]:
        print(f"FAIL: row-sum error {row_err.max():.3e} > {args.max_row_sum_err:.1e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
