#!/usr/bin/env python3
"""
Validate the website collapse (docs/py/collapse.py) against MACE directly.

For one weight function and temperature index, computes sigma and dsigma/dT
with MACE directly on coarse grids and compares them with the collapse of the
production 128-group tables in output/tables/, both as raw float64 tables and
in the float32 website format produced by scripts/prepare_website.py.

  aligned      every 8th fine edge (16 groups), 10 angle bins   -> gate
  non-aligned  11 geometric edges over the full range, 10 bins  -> reported
  interp       log-T interpolation between T_i and T_{i+1} at
               their geometric mean on the aligned grid         -> reported

The direct coarse runs use twice the production E-panel Gauss-Legendre order
(override with --direct-e-panel-order) because each coarse group spans eight
fine groups; at cold temperatures the recoil-dominated high-energy groups
converge slowly in that order.

Errors are relative to |sigma| (or |sigma|/T for the derivative) on entries
above 1e-6 of their incoming group's maximum. Results are printed and written
to output/verify/collapse_{weight}_T{idx}.json.

Usage:
  python scripts/verify_collapse_direct.py --weight wien --temperature-index 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from compute_matrix import (
    DEFAULT_CAP_X, OUTPUT_DIR, ROOT, TEMPERATURES_K, WEIGHTS,
    compute_table, make_boundaries_kev, output_path, production_orders,
)
from prepare_website import npz_bytes, site_dsigma_arrays, site_sigma_arrays

sys.path.insert(0, str(ROOT / "docs" / "py"))
from collapse import _collapse_dsigma_dT_to_array, _collapse_to_array  # noqa: E402

VERIFY_DIR = ROOT / "output" / "verify"
SIGNIFICANT = 1e-6
N_COARSE_ANGLES = 10
ALIGNED_GATE = 1e-4


def errors(got: np.ndarray, ref: np.ndarray, sigma_ref: np.ndarray, T: float | None) -> dict:
    """Error stats on significant entries; derivative errors in units of sigma/T."""
    row_max = sigma_ref.max(axis=(1, 2), keepdims=True)
    mask = sigma_ref > SIGNIFICANT * row_max
    scale = np.abs(ref) if T is None else np.maximum(np.abs(sigma_ref), SIGNIFICANT * row_max) / T
    err = np.where(mask, np.abs(got - ref) / np.where(mask, scale, 1.0), 0.0)
    worst = np.unravel_index(np.argmax(err), err.shape)
    vals = err[mask]
    return {
        "max": float(vals.max()), "p99": float(np.percentile(vals, 99)),
        "median": float(np.median(vals)), "n": int(mask.sum()),
        "worst_entry": [int(i) for i in worst],
        "worst_rel_to_row_max": float(sigma_ref[worst] / row_max[worst[0], 0, 0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weight", choices=WEIGHTS, required=True)
    parser.add_argument("--temperature-index", type=int, required=True)
    parser.add_argument("--direct-e-panel-order", type=int, default=None)
    args = parser.parse_args()
    w, tidx = args.weight, args.temperature_index
    if args.direct_e_panel_order is None:
        args.direct_e_panel_order = 2 * production_orders(tidx)["e_panel_order"]
    T = TEMPERATURES_K[tidx]

    raw_s = dict(np.load(output_path(OUTPUT_DIR, w, "sigma", tidx, T)))
    raw_d = dict(np.load(output_path(OUTPUT_DIR, w, "dsigma_dT", tidx, T)))
    formats = {
        "raw": (npz_bytes(raw_s), npz_bytes(raw_d)),
        "site": (npz_bytes(site_sigma_arrays(raw_s)),
                 npz_bytes(site_dsigma_arrays(raw_d, raw_s["sigma_matrix"]))),
    }

    fine = make_boundaries_kev()
    grids = {"aligned": fine[::8], "non-aligned": np.geomspace(fine[0], fine[-1], 11)}

    def direct(bounds, temp):
        kw = dict(cap_x=DEFAULT_CAP_X, e_panel_order=args.direct_e_panel_order)
        s, _ = compute_table(w, "sigma", bounds, N_COARSE_ANGLES, temp, **kw)
        d, _ = compute_table(w, "dsigma_dT", bounds, N_COARSE_ANGLES, temp, **kw)
        return s, d

    result = {"weight": w, "temperature_index": tidx, "temperature_K": T,
              "direct_e_panel_order": args.direct_e_panel_order}
    t0 = time.time()
    for gname, bounds in grids.items():
        ref_s, ref_d = direct(bounds, T)
        for fname, (sb, db) in formats.items():
            got_s = _collapse_to_array(sb, bounds.tolist(), N_COARSE_ANGLES)
            got_d = _collapse_dsigma_dT_to_array(sb, db, bounds.tolist(), N_COARSE_ANGLES)
            result[f"{gname}/{fname}"] = {
                "sigma": errors(got_s, ref_s, ref_s, None),
                "dsigma_dT": errors(got_d, ref_d, ref_s, T),
            }

    if tidx + 1 < len(TEMPERATURES_K):
        T_hi = TEMPERATURES_K[tidx + 1]
        T_mid = np.sqrt(T * T_hi)
        s_lo, d_lo = direct(grids["aligned"], T)
        s_hi, d_hi = direct(grids["aligned"], T_hi)
        s_mid, d_mid = direct(grids["aligned"], T_mid)
        alpha = (np.log(T_mid) - np.log(T)) / (np.log(T_hi) - np.log(T))
        result["interp/aligned"] = {
            "T_target": T_mid,
            "sigma": errors((1 - alpha) * s_lo + alpha * s_hi, s_mid, s_mid, None),
            "dsigma_dT": errors(((1 - alpha) * T * d_lo + alpha * T_hi * d_hi) / T_mid,
                                d_mid, s_mid, T_mid),
        }
    result["direct_wall_time_s"] = time.time() - t0

    gate = max(result["aligned/raw"][q]["max"] for q in ("sigma", "dsigma_dT"))
    result["aligned_gate"] = {"threshold": ALIGNED_GATE, "max_err": gate, "pass": gate < ALIGNED_GATE}

    VERIFY_DIR.mkdir(parents=True, exist_ok=True)
    out = VERIFY_DIR / f"collapse_{w}_T{tidx:03d}.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
