#!/usr/bin/env python3
"""
Compute a deterministic multigroup Compton scattering table (angle-resolved).

Headless, SLURM-friendly script. Computes either the 128x128x100 sigma matrix
or its full temperature derivative dsigma/dT on a 128-group geometric energy
grid with 100 angle bins, for one of three weight functions:

  uniform  w = 1
  wien     w = x^3 exp(-(x - x_lo))            (shifted per group)
  planck   w = x^3 / (exp(x) - 1) for x < cap_x, shifted Wien above

Raw float64 output goes to output/tables/{weight}/{quantity}/; run
scripts/prepare_website.py afterwards to produce the files under docs/data/.

Usage:
  python scripts/compute_matrix.py --weight planck --quantity sigma --temperature-index 0
  python scripts/compute_matrix.py --weight wien --quantity dsigma_dT --temperature-index 32

Quadrature flags override production_orders() one field at a time, and
--ep-k-cut overrides production_ep_k_cut().
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

import numpy as np

import compton_matrix._compton_differential_cross_section as cds
import compton_matrix._compton_multigroup as cm
from compton_matrix import k_boltz, kev

ROOT = Path(__file__).resolve().parent.parent
MACE_DIR = ROOT / "external" / "ComptonMatrixExact"

N_GROUPS = 128
N_ANGLE_BINS = 100
E_MIN_KEV = 1e-5  # 0.01 eV
E_MAX_KEV = 300.0

N_TEMPS = 64
T_MIN_K = 1000.0
T_MAX_K = 1e9

TEMPERATURES_K = np.geomspace(T_MIN_K, T_MAX_K, N_TEMPS)

WEIGHTS = ("uniform", "wien", "planck")
QUANTITIES = ("sigma", "dsigma_dT")
SIGMA_CUTOFF_RATIO = 1e-10
DEFAULT_CAP_X = 25.0
QUADRATURE_ORDERS = ("e_panel_order", "xi_order", "xi_tail_order", "ep_edge_order", "ep_interior_order")
COLD_T_INDEX = 20  # T000-T019 use the higher cold-temperature E-panel order
WIDE_EP_T_INDEX = 31  # T000-T030 use the wider E' window (production_ep_k_cut)
WIDE_EP_K_CUT = 15.0
WIDE_EP_ORDER = 96

OUTPUT_DIR = ROOT / "output" / "tables"


def make_boundaries_kev(n_groups: int = N_GROUPS) -> np.ndarray:
    return np.geomspace(E_MIN_KEV, E_MAX_KEV, n_groups + 1)


def make_weight_function(weight: str, boundaries_erg: list[float], cap_x: float):
    if weight == "uniform":
        return cm.UniformWeightFunction()
    if weight == "wien":
        return cm.WienWeightFunction(group_boundaries=boundaries_erg)
    if weight == "planck":
        return cm.PlanckWeightFunction(cap_x=cap_x, group_boundaries=boundaries_erg)
    raise ValueError(f"Unknown weight function: {weight!r}")


def is_shifted(weight: str, x_lo: np.ndarray, cap_x: float) -> np.ndarray:
    """Groups whose weight carries the exp(x_lo) shift (see weight_function.cpp)."""
    if weight == "wien":
        return np.ones_like(x_lo, dtype=bool)
    if weight == "planck":
        return x_lo >= cap_x
    return np.zeros_like(x_lo, dtype=bool)


def weight_norms(
    wf, weight: str, boundaries_erg: list[float], T: float, cap_x: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-group log of the *unshifted* weight integral and its T-derivative.

    log_norm[g]     = ln D_g - x_lo,g * [shifted]
    dlog_norm_dT[g] = D'_g / D_g + (x_lo,g / T) * [shifted]

    where D_g = wf.compute_denominator(E_g, E_{g+1}, T). These are the
    weights needed to collapse fine groups into coarse ones exactly.
    """
    b = np.asarray(boundaries_erg)
    D = np.array([wf.compute_denominator(lo, hi, T) for lo, hi in zip(b[:-1], b[1:])])
    dD = np.array([wf.d_denominator_dT(lo, hi, T) for lo, hi in zip(b[:-1], b[1:])])
    if np.any(~np.isfinite(D)) or np.any(D <= 0):
        raise RuntimeError(f"Non-positive or non-finite weight denominator at T={T:.6e} K")
    x_lo = b[:-1] / (k_boltz * T)
    shifted = is_shifted(weight, x_lo, cap_x)
    log_norm = np.log(D) - np.where(shifted, x_lo, 0.0)
    dlog_norm_dT = dD / D + np.where(shifted, x_lo / T, 0.0)
    return log_norm, dlog_norm_dT


def compute_table(
    weight: str,
    quantity: str,
    boundaries_kev: np.ndarray,
    n_angle_bins: int,
    T: float,
    *,
    cap_x: float = DEFAULT_CAP_X,
    e_panel_order: int | None = None,
    orders: dict[str, int] | None = None,
    ep_k_cut: float | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Compute sigma or dsigma/dT (cm^2 or cm^2/K), shape (G, G, A), plus metadata.

    ``orders`` overrides MGIntegrationConfig quadrature orders by field name
    (any of QUADRATURE_ORDERS); ``e_panel_order`` is shorthand for one of them.
    ``ep_k_cut`` sets the half-width of the E' window in thermal widths.
    """
    boundaries_erg = (np.asarray(boundaries_kev) * kev).tolist()
    wf = make_weight_function(weight, boundaries_erg, cap_x)
    config = make_config(quantity, e_panel_order=e_panel_order, orders=orders, ep_k_cut=ep_k_cut)
    cutoff = config.cutoff_ratio

    mg = cm.ComptonMultigroupKernel(
        energy_group_boundaries=boundaries_erg,
        weight_function=wf,
        config=config,
    )
    kernel = cds.ComptonKernelSolver()

    if quantity == "sigma":
        matrix = mg.compute_sigma_matrix(kernel, n_angle_bins, T=T)
    elif weight == "uniform":
        matrix = mg.compute_kernel_derivative_contribution(kernel, n_angle_bins, T=T)
    else:
        matrix = mg.compute_dsigma_dT_matrix(kernel, n_angle_bins, T=T)

    log_norm, dlog_norm_dT = weight_norms(wf, weight, boundaries_erg, T, cap_x)
    meta = dict(
        log_norm=log_norm,
        dlog_norm_dT=dlog_norm_dT,
        cutoff_ratio=np.nan if cutoff is None else cutoff,
        **effective_settings(config),
    )
    return np.asarray(matrix, dtype=np.float64), meta


def production_orders(tidx: int) -> dict[str, int]:
    """
    Quadrature orders used for the published tables at temperature index tidx.

    The E-panel order is raised from the library default (16): Wien/Planck
    weights decay on the scale kT inside high-energy groups and need 64 at all
    temperatures, and the cold, recoil-dominated regime needs 128 below T020
    (see scripts/verify_convergence.py). The xi orders are kept at their
    defaults.

    Below T031 the E' orders are raised to 96 together with the wider E'
    window of production_ep_k_cut(). In the most forward angle bin MACE ends
    the E' window at E' = E once the recoil shift across the bin exceeds
    ep_k_cut thermal widths, and misses the narrow near-forward upscatter
    tail; with the default window (5) this costs up to 3e-4 in row sums at
    every temperature up to T029 (scripts/scan_ep_window.py). A window of 15
    needs E' orders 96 to keep narrow features resolved.
    """
    orders = {"e_panel_order": 128 if tidx < COLD_T_INDEX else 64}
    if tidx < WIDE_EP_T_INDEX:
        orders.update(ep_edge_order=WIDE_EP_ORDER, ep_interior_order=WIDE_EP_ORDER)
    return orders


def production_ep_k_cut(tidx: int) -> float | None:
    """E' window half-width (thermal widths) for the published tables; None is the library default."""
    return WIDE_EP_K_CUT if tidx < WIDE_EP_T_INDEX else None


def make_config(quantity: str, *, e_panel_order: int | None = None,
                orders: dict[str, int] | None = None, ep_k_cut: float | None = None):
    # dsigma/dT is evaluated over all G^2 pairs (the library disables the
    # cutoff for compute_dsigma_dT_matrix); for uniform weighting the
    # kernel-only derivative is the full derivative, so one pass suffices.
    cutoff = SIGMA_CUTOFF_RATIO if quantity == "sigma" else None
    config_kwargs: dict = dict(cutoff_ratio=cutoff)
    if e_panel_order is not None:
        config_kwargs["e_panel_order"] = e_panel_order
    for name, value in (orders or {}).items():
        if name not in QUADRATURE_ORDERS:
            raise ValueError(f"Unknown quadrature order {name!r}")
        if value is not None:
            config_kwargs[name] = value
    if ep_k_cut is not None:
        config_kwargs["ep_k_cut"] = ep_k_cut
    return cm.MGIntegrationConfig(**config_kwargs)


def effective_orders(config) -> dict[str, int]:
    return {name: getattr(config, f"effective_{name}")() for name in QUADRATURE_ORDERS}


def effective_settings(config) -> dict[str, float]:
    """Effective quadrature orders plus the E' window half-width."""
    return {**effective_orders(config), "ep_k_cut": float(config.ep_k_cut)}


def stored_settings(npz) -> dict[str, float]:
    """Settings recorded in an output file; files without ep_k_cut used the default."""
    have = {name: int(npz[name]) for name in QUADRATURE_ORDERS if name in npz.files}
    have["ep_k_cut"] = (float(npz["ep_k_cut"]) if "ep_k_cut" in npz.files
                        else float(cm.MGIntegrationConfig().ep_k_cut))
    return have


def mace_version() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(MACE_DIR), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def output_path(out_root: Path, weight: str, quantity: str, tidx: int, T: float) -> Path:
    return out_root / weight / quantity / f"T{tidx:03d}_{T:.6e}K.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weight", choices=WEIGHTS, required=True)
    parser.add_argument("--quantity", choices=QUANTITIES, required=True)
    parser.add_argument(
        "--temperature-index", type=int, required=True,
        help=f"Temperature index (0..{N_TEMPS - 1})")
    parser.add_argument(
        "--cap-x", type=float, default=DEFAULT_CAP_X,
        help=f"Planck cap x = E/kT above which shifted Wien is used (default {DEFAULT_CAP_X})")
    parser.add_argument(
        "--e-panel-order", type=int, default=None,
        help="Override e_panel_order (default: production_orders(), 128 below T020 else 64)")
    parser.add_argument(
        "--xi-order", type=int, default=None,
        help="Override xi_order (default: library default of 48)")
    parser.add_argument(
        "--xi-tail-order", type=int, default=None,
        help="Override xi_tail_order (default: library default of 48)")
    parser.add_argument(
        "--ep-edge-order", type=int, default=None,
        help="Override ep_edge_order (default: production_orders(), 96 below T031 else 24)")
    parser.add_argument(
        "--ep-interior-order", type=int, default=None,
        help="Override ep_interior_order (default: production_orders(), 96 below T031 else 24)")
    parser.add_argument(
        "--ep-k-cut", type=float, default=None,
        help="Half-width of the E' window in thermal widths (default: production_ep_k_cut(), 15 below T031 else 5)")
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip if the output exists with the same quadrature settings")
    parser.add_argument(
        "--groups", type=int, default=N_GROUPS,
        help="Debug: number of energy groups (default %(default)s)")
    parser.add_argument(
        "--angles", type=int, default=N_ANGLE_BINS,
        help="Debug: number of angle bins (default %(default)s)")
    parser.add_argument(
        "--out-root", type=Path, default=OUTPUT_DIR,
        help="Output root directory (default output/tables)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tidx = args.temperature_index
    if not 0 <= tidx < N_TEMPS:
        raise ValueError(f"temperature-index must be in [0, {N_TEMPS - 1}], got {tidx}")

    T = TEMPERATURES_K[tidx]
    overrides = {name: getattr(args, name) for name in QUADRATURE_ORDERS
                 if getattr(args, name) is not None}
    orders = {**production_orders(tidx), **overrides}
    ep_k_cut = args.ep_k_cut if args.ep_k_cut is not None else production_ep_k_cut(tidx)
    wanted = effective_settings(make_config(args.quantity, orders=orders, ep_k_cut=ep_k_cut))

    out_path = output_path(args.out_root, args.weight, args.quantity, tidx, T)
    if args.skip_existing and out_path.exists():
        with np.load(out_path) as existing:
            have = stored_settings(existing)
        if have == wanted:
            print(f"Exists with the same quadrature settings, skipping: {out_path}")
            return
        print(f"Exists with settings {have}, recomputing with {wanted}: {out_path}")

    boundaries_kev = make_boundaries_kev(args.groups)
    print(f"[{args.weight}/{args.quantity} T{tidx:03d}] Computing "
          f"{args.groups}x{args.groups}x{args.angles} matrix at T = {T:.6e} K "
          f"with settings {wanted}, OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'unset')} ...",
          flush=True)

    t0 = time.time()
    matrix, meta = compute_table(
        args.weight, args.quantity, boundaries_kev, args.angles, T,
        cap_x=args.cap_x, orders=orders, ep_k_cut=ep_k_cut,
    )
    elapsed = time.time() - t0
    if not np.all(np.isfinite(matrix)):
        raise RuntimeError(f"Non-finite values in {args.quantity} matrix")
    print(f"  {args.quantity} done ({elapsed:.1f}s)", flush=True)

    key = "sigma_matrix" if args.quantity == "sigma" else "dsigma_dT_matrix"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp_path,
        **{key: matrix},
        boundaries_keV=boundaries_kev,
        temperature_K=T,
        weight_function=args.weight,
        cap_x=args.cap_x if args.weight == "planck" else np.nan,
        mace_commit=mace_version(),
        wall_time_s=elapsed,
        **meta,
    )
    tmp_path.replace(out_path)
    print(f"Saved: {out_path} ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
