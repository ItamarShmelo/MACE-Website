#!/usr/bin/env python3
"""
Build the website data under docs/data/ from the raw tables in output/tables/.

For each weight function (uniform, wien, planck) this reads the float64
tables written by scripts/compute_matrix.py, validates them for consistency
(shape, required keys, matching boundaries and temperatures), and writes

  docs/data/{weight}/sigma/T{idx}_{temp}K.npz      sigma_matrix (float32) [cm^2]
  docs/data/{weight}/dsigma_dT/T{idx}_{temp}K.npz  T_dsigma_dT (float32)  [cm^2]
  docs/data/{weight}/manifest.json

T_dsigma_dT = T * dsigma/dT is stored instead of dsigma/dT, which would
underflow float32 at high temperature; entries below 1e-10 of the incoming
group's maximum sigma are set to zero. The sigma files also carry the
per-group weight normalisation (log_norm, dlog_norm_dT) used by the collapse.

Usage:
  python scripts/prepare_website.py                 # all weights, all 64 temperatures
  python scripts/prepare_website.py --weights planck --allow-partial
"""

from __future__ import annotations

import argparse
import io
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "output" / "tables"
DATA_DIR = ROOT / "docs" / "data"

WEIGHTS = ("uniform", "wien", "planck")
N_TEMPS = 64
DERIV_ZERO_RATIO = 1e-10

FILENAME_RE = re.compile(r"^T(\d{3})_(.+)K\.npz$")
SIGMA_KEYS = {"sigma_matrix", "boundaries_keV", "temperature_K", "weight_function",
              "log_norm", "dlog_norm_dT", "cap_x"}
DERIV_KEYS = {"dsigma_dT_matrix", "boundaries_keV", "temperature_K", "weight_function"}


def site_sigma_arrays(raw: dict) -> dict:
    """Arrays for a site sigma file from a raw sigma table."""
    return dict(
        sigma_matrix=raw["sigma_matrix"].astype(np.float32),
        boundaries_keV=raw["boundaries_keV"],
        temperature_K=raw["temperature_K"],
        weight_function=raw["weight_function"],
        cap_x=raw["cap_x"],
        log_norm=raw["log_norm"],
        dlog_norm_dT=raw["dlog_norm_dT"],
    )


def site_dsigma_arrays(raw_deriv: dict, sigma: np.ndarray) -> dict:
    """Arrays for a site dsigma/dT file; sigma sets the per-row zero threshold."""
    T = float(raw_deriv["temperature_K"])
    t_ds = T * raw_deriv["dsigma_dT_matrix"]
    row_max = np.abs(sigma).max(axis=(1, 2), keepdims=True)
    t_ds = np.where(np.abs(t_ds) < DERIV_ZERO_RATIO * row_max, 0.0, t_ds)
    return dict(
        T_dsigma_dT=t_ds.astype(np.float32),
        boundaries_keV=raw_deriv["boundaries_keV"],
        temperature_K=raw_deriv["temperature_K"],
        weight_function=raw_deriv["weight_function"],
    )


def npz_bytes(arrays: dict) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


def load_raw(path: Path, required: set[str]) -> dict:
    with np.load(path) as data:
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"{path}: missing keys {sorted(missing)}")
        return {k: data[k] for k in data.files}


def collect(raw_dir: Path, weight: str) -> tuple[dict[int, Path], dict[int, Path]]:
    found = {}
    for quantity in ("sigma", "dsigma_dT"):
        found[quantity] = {}
        for path in sorted((raw_dir / weight / quantity).glob("T*_*K.npz")):
            m = FILENAME_RE.match(path.name)
            if m:
                found[quantity][int(m.group(1))] = path
    return found["sigma"], found["dsigma_dT"]


def prepare_weight(weight: str, allow_partial: bool, raw_dir: Path, data_dir: Path) -> bool:
    sigma_paths, deriv_paths = collect(raw_dir, weight)
    indices = sorted(set(sigma_paths) & set(deriv_paths))
    missing = sorted(set(range(N_TEMPS)) - set(indices))
    if missing and not allow_partial:
        print(f"ERROR: {weight}: missing sigma and/or dsigma_dT for T indices {missing}",
              file=sys.stderr)
        return False
    if not indices:
        print(f"ERROR: {weight}: no complete temperatures in {raw_dir / weight}", file=sys.stderr)
        return False

    # Build into a scratch directory and swap it in only if everything validates.
    out_dir = data_dir / weight
    new_dir = data_dir / f".{weight}.new"
    if new_dir.exists():
        shutil.rmtree(new_dir)
    for sub in ("sigma", "dsigma_dT"):
        (new_dir / sub).mkdir(parents=True)

    ref_shape = ref_boundaries = None
    cap_x = None
    temperatures = []
    sizes = {"sigma": 0, "dsigma_dT": 0}
    for tidx in indices:
        raw_s = load_raw(sigma_paths[tidx], SIGMA_KEYS)
        raw_d = load_raw(deriv_paths[tidx], DERIV_KEYS)
        sigma, deriv = raw_s["sigma_matrix"], raw_d["dsigma_dT_matrix"]
        name = sigma_paths[tidx].name

        problems = []
        if sigma.ndim != 3 or deriv.shape != sigma.shape:
            problems.append(f"shapes sigma {sigma.shape}, dsigma_dT {deriv.shape}")
        if ref_shape is None:
            ref_shape, ref_boundaries = sigma.shape, raw_s["boundaries_keV"]
            cap_x = float(raw_s["cap_x"])
        elif sigma.shape != ref_shape or not np.array_equal(raw_s["boundaries_keV"], ref_boundaries):
            problems.append("shape or boundaries differ from reference")
        if not np.array_equal(raw_d["boundaries_keV"], raw_s["boundaries_keV"]) or \
                float(raw_d["temperature_K"]) != float(raw_s["temperature_K"]):
            problems.append("sigma and dsigma_dT tables do not match")
        if str(raw_s["weight_function"]) != weight or str(raw_d["weight_function"]) != weight:
            problems.append("weight_function mismatch")
        if not (np.all(np.isfinite(sigma)) and np.all(np.isfinite(deriv))):
            problems.append("non-finite values")
        if np.any(sigma < 0):
            problems.append(f"negative sigma (min {sigma.min():.3e})")
        if problems:
            print(f"ERROR: {weight}/{name}: {'; '.join(problems)}", file=sys.stderr)
            shutil.rmtree(new_dir)
            return False

        site_s = site_sigma_arrays(raw_s)
        site_d = site_dsigma_arrays(raw_d, sigma)
        for sub, arrays in (("sigma", site_s), ("dsigma_dT", site_d)):
            path = new_dir / sub / name
            path.write_bytes(npz_bytes(arrays))
            sizes[sub] += path.stat().st_size

        temperatures.append({
            "index": tidx,
            "temperature_K": float(raw_s["temperature_K"]),
            "file": f"sigma/{name}",
            "dsigma_dT_file": f"dsigma_dT/{name}",
        })

    G, _, M = ref_shape
    manifest = {
        "description": f"Multigroup Compton scattering matrices ({weight} weight function)",
        "boundaries_keV": ref_boundaries.tolist(),
        "n_groups": G,
        "n_angle_bins": M,
        "xi_min": -1.0,
        "xi_max": 1.0,
        "weight_function": weight,
        "cap_x": cap_x if weight == "planck" else None,
        "quantities": {
            "sigma": {"key": "sigma_matrix", "units": "cm^2"},
            "dsigma_dT": {"key": "T_dsigma_dT", "units": "cm^2",
                          "note": "stored as T * dsigma/dT; divide by temperature_K for cm^2/K"},
        },
        "temperatures": temperatures,
    }
    with open(new_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    new_dir.rename(out_dir)

    print(f"{weight}: {len(temperatures)} temperatures, {G} groups, {M} angle bins; "
          f"sigma {sizes['sigma'] / 1e6:.1f} MB, dsigma_dT {sizes['dsigma_dT'] / 1e6:.1f} MB"
          + (f"; MISSING {missing}" if missing else ""))
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", nargs="+", choices=WEIGHTS, default=list(WEIGHTS))
    parser.add_argument("--allow-partial", action="store_true",
                        help="Publish whatever temperatures are complete")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR,
                        help="Raw tables (default output/tables)")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR,
                        help="Website data directory (default docs/data)")
    args = parser.parse_args()

    ok = all([prepare_weight(w, args.allow_partial, args.raw_dir, args.data_dir)
              for w in args.weights])
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
