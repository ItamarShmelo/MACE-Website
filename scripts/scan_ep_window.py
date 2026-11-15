#!/usr/bin/env python3
"""
Map where the published tables are limited by MACE's E' window.

Compares the production tables in output/tables (library default E' window,
ep_k_cut 5 with E' orders 24) against full-table variants computed by
scripts/submit_ep_scan.sh into output/ep_scan/{label}/, one row (incoming
group, summed over outgoing groups and angle bins) at a time:

  row    sum(published) / sum(variant) - 1
  entry  max |published - variant| / max(variant), within the row

A second variant, when present, checks that the first is itself converged.
For dsigma/dT both errors are scaled by sigma instead (T * delta of the
derivative over the sigma row sum or row maximum), since derivative rows can
cross zero.

Usage:
  python scripts/scan_ep_window.py [--variant k10ep48] [--check k15ep96] [--tol 1e-4]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "output" / "tables"
SCAN = ROOT / "output" / "ep_scan"
KEYS = {"sigma": "sigma_matrix", "dsigma_dT": "dsigma_dT_matrix"}


def table_files(root: Path, weight: str, quantity: str) -> dict[int, Path]:
    files = {}
    for path in (root / weight / quantity).glob("T*.npz"):
        if path.name.endswith(".tmp.npz"):
            continue
        files[int(re.match(r"T(\d+)_", path.name).group(1))] = path
    return files


def load(path: Path, quantity: str) -> tuple[np.ndarray, float, np.ndarray]:
    with np.load(path) as d:
        return d[KEYS[quantity]], float(d["temperature_K"]), d["boundaries_keV"]


def row_errors(ref: np.ndarray, test: np.ndarray, scale: np.ndarray | None = None
               ) -> tuple[np.ndarray, np.ndarray]:
    """Per incoming group: relative row-total and max-entry differences of test vs ref."""
    if scale is None:
        scale = ref
    row_scale = np.abs(scale.reshape(scale.shape[0], -1).sum(axis=1))
    max_scale = np.abs(scale).reshape(scale.shape[0], -1).max(axis=1)
    diff = (test - ref).reshape(ref.shape[0], -1)
    with np.errstate(divide="ignore", invalid="ignore"):
        row = np.where(row_scale > 0, diff.sum(axis=1) / row_scale, 0.0)
        entry = np.where(max_scale > 0, np.abs(diff).max(axis=1) / max_scale, 0.0)
    return row, entry


def summarise(row: np.ndarray, entry: np.ndarray, e_lo: np.ndarray, tol: float) -> dict:
    worst = int(np.argmax(np.abs(row)))
    over_row = np.flatnonzero(np.abs(row) > tol)
    over_entry = np.flatnonzero(entry > tol)
    return dict(
        max_row=float(row[worst]),
        max_row_E_keV=float(e_lo[worst]),
        max_entry=float(entry.max()),
        max_entry_E_keV=float(e_lo[int(np.argmax(entry))]),
        rows_over_row_tol=int(over_row.size),
        rows_over_entry_tol=int(over_entry.size),
        min_E_over_row_tol_keV=float(e_lo[over_row].min()) if over_row.size else None,
        min_E_over_entry_tol_keV=float(e_lo[over_entry].min()) if over_entry.size else None,
    )


def fmt(x: float | None) -> str:
    return "-" if x is None else f"{x:+.1e}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", default="k10ep48")
    parser.add_argument("--check", default="k15ep96")
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--json", type=Path, default=SCAN / "summary.json")
    args = parser.parse_args()

    results = []
    for weight in ("uniform", "wien", "planck"):
        for quantity in ("sigma", "dsigma_dT"):
            published = table_files(TABLES, weight, quantity)
            variant = table_files(SCAN / args.variant, weight, quantity)
            check = table_files(SCAN / args.check, weight, quantity)
            sigma_variant = table_files(SCAN / args.variant, weight, "sigma")
            for tidx in sorted(set(published) & set(variant)):
                pub, T, b = load(published[tidx], quantity)
                var, _, b_var = load(variant[tidx], quantity)
                if not np.array_equal(b, b_var):
                    raise RuntimeError(f"Grid mismatch for {weight}/{quantity} T{tidx:03d}")
                scale = None
                if quantity == "dsigma_dT":
                    if tidx not in sigma_variant:
                        continue
                    scale = load(sigma_variant[tidx], "sigma")[0] / T
                rec = dict(weight=weight, quantity=quantity, tidx=tidx, T_K=T,
                           published=summarise(*row_errors(var, pub, scale), b[:-1], args.tol))
                if tidx in check:
                    chk = load(check[tidx], quantity)[0]
                    rec["check"] = summarise(*row_errors(chk, var, scale), b[:-1], args.tol)
                results.append(rec)

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(dict(variant=args.variant, check=args.check,
                                         tol=args.tol, results=results), indent=1))

    print(f"published vs {args.variant} (and {args.variant} vs {args.check}); "
          f"rows = incoming groups with |error| > {args.tol:g}")
    print(f"{'weight':8s} {'qty':9s} {'T':>4s} {'T [K]':>8s} | {'max row':>8s} {'at keV':>7s} "
          f"{'rows':>4s} {'from keV':>8s} | {'max entry':>9s} {'at keV':>7s} {'rows':>4s} | "
          f"{'check row':>9s} {'check ent':>9s}")
    for r in results:
        p = r["published"]
        c = r.get("check")
        from_kev = p["min_E_over_row_tol_keV"]
        from_str = "-" if from_kev is None else f"{from_kev:.3g}"
        check_row = fmt(c["max_row"]) if c else ""
        check_entry = f"{c['max_entry']:.1e}" if c else ""
        print(f"{r['weight']:8s} {r['quantity']:9s} {r['tidx']:4d} {r['T_K']:8.2e} | "
              f"{p['max_row']:+8.1e} {p['max_row_E_keV']:7.3g} {p['rows_over_row_tol']:4d} "
              f"{from_str:>8s} | "
              f"{p['max_entry']:9.1e} {p['max_entry_E_keV']:7.3g} {p['rows_over_entry_tol']:4d} | "
              f"{check_row:>9s} {check_entry:>9s}")
    print(f"Wrote {args.json}")


if __name__ == "__main__":
    main()
