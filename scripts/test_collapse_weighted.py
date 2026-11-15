#!/usr/bin/env python3
"""
Synthetic exactness tests for the weight-aware collapse in docs/py/collapse.py.

Builds group-averaged tables of a known smooth function f(E, T) on the
production 128-group grid and on coarser grids, using MACE's own weight
denominators (shifted Wien/Planck) and the production ``weight_norms``.
On grids aligned with the fine edges the collapsed fine table must reproduce
the directly computed coarse table, for sigma and for dsigma/dT (obtained by
central differences in T). Non-aligned grids are reported, not asserted.

Needs the MACE venv:
  external/ComptonMatrixExact/.venv/bin/python3 scripts/test_collapse_weighted.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "docs" / "py"))
sys.path.insert(0, str(ROOT / "scripts"))

from collapse import _collapse_dsigma_dT_to_array, _collapse_to_array  # noqa: E402
from compute_matrix import (  # noqa: E402
    DEFAULT_CAP_X, is_shifted, make_boundaries_kev, make_weight_function, weight_norms,
)
from compton_matrix import k_boltz, kev  # noqa: E402

CAP_X = DEFAULT_CAP_X
H_REL = 1e-4
GL_X, GL_W = np.polynomial.legendre.leggauss(16)
N_PANELS = 32
N_OUT, N_ANG = 128, 4
# The reference derivative is a central difference of MACE's denominators.
# For Planck, planck_integral is only accurate to ~3e-5 for x_lo in [1.5, 2.5],
# so its finite difference disagrees with the analytic d_denominator_dT used
# by the collapse at the ~1e-5 level; that is not a collapse error.
DERIV_TOL = {"uniform": 1e-7, "wien": 1e-7, "planck": 1e-4}

RNG = np.random.default_rng(7)
PROFILE = RNG.uniform(0.5, 2.0, size=(N_OUT, N_ANG))  # T-independent outgoing/angle shape


def f_test(E_kev: np.ndarray, T: float) -> np.ndarray:
    """Smooth, T-dependent test function of incoming energy."""
    kT = k_boltz * T / kev
    return 1.0 + 0.5 * np.tanh(np.log(E_kev / (3.0 * kT))) + 0.2 * np.sin(np.log(E_kev))


def log_w_unshifted(weight: str, x: np.ndarray) -> np.ndarray:
    if weight == "uniform":
        return np.zeros_like(x)
    if weight == "wien":
        return 3 * np.log(x) - x
    return np.where(x < CAP_X, 3 * np.log(x) - np.log(np.expm1(np.minimum(x, CAP_X))), 3 * np.log(x) - x)


def log_quad(weight: str, xa: float, xb: float, T: float) -> float:
    """ln of integral_xa^xb w_unshifted(x) f(x) dx (x = E/kT), composite GL."""
    span = xb - xa if weight == "uniform" else min(xb - xa, 80.0)
    edges = np.linspace(0.0, span, N_PANELS + 1)
    mid, half = 0.5 * (edges[1:] + edges[:-1]), 0.5 * np.diff(edges)
    t = (mid[:, None] + half[:, None] * GL_X).ravel()
    x = xa + t
    lw = log_w_unshifted(weight, x)
    ref = lw.max()
    kT = k_boltz * T / kev
    vals = np.exp(lw - ref) * f_test(x * kT, T)
    return float(np.log((vals.reshape(N_PANELS, -1) @ GL_W) @ half) + ref)


def log_numerators(weight: str, bounds_kev: np.ndarray, T: float) -> np.ndarray:
    """ln integral_g w_unshifted f dx per group, summing fine sub-pieces."""
    fine = make_boundaries_kev()
    kT = k_boltz * T / kev
    out = []
    for lo, hi in zip(bounds_kev[:-1], bounds_kev[1:]):
        edges = np.concatenate([[lo], fine[(fine > lo * (1 + 1e-12)) & (fine < hi * (1 - 1e-12))], [hi]])
        pieces = [log_quad(weight, a / kT, b / kT, T) for a, b in zip(edges[:-1], edges[1:])]
        out.append(np.logaddexp.reduce(pieces))
    return np.array(out)


def group_average(weight: str, bounds_kev: np.ndarray, T: float) -> np.ndarray:
    """<f>_g with MACE's (shifted) weight: N_g / D_g, D_g from the library."""
    b_erg = (bounds_kev * kev).tolist()
    wf = make_weight_function(weight, b_erg, CAP_X)
    D = np.array([wf.compute_denominator(lo, hi, T) for lo, hi in zip(b_erg[:-1], b_erg[1:])])
    kT_erg = k_boltz * T
    x_lo = bounds_kev[:-1] * kev / kT_erg
    shift = np.where(is_shifted(weight, x_lo, CAP_X), x_lo, 0.0)
    # Library D is in erg and shifted by exp(x_lo); numerators are unshifted in x units.
    return np.exp(log_numerators(weight, bounds_kev, T) + shift - (np.log(D) - np.log(kT_erg)))


def fine_npz(weight: str, T: float) -> tuple[bytes, bytes]:
    """Synthetic fine sigma and dsigma/dT tables in the raw output format."""
    b = make_boundaries_kev()
    b_erg = (b * kev).tolist()
    wf = make_weight_function(weight, b_erg, CAP_X)
    avg = group_average(weight, b, T)
    davg = (group_average(weight, b, T * (1 + H_REL)) - group_average(weight, b, T * (1 - H_REL))) / (2 * H_REL * T)
    log_norm, dlog_norm = weight_norms(wf, weight, b_erg, T, CAP_X)
    common = dict(boundaries_keV=b, temperature_K=T, weight_function=weight,
                  cap_x=CAP_X, log_norm=log_norm, dlog_norm_dT=dlog_norm)
    sig, dsig = io.BytesIO(), io.BytesIO()
    np.savez(sig, sigma_matrix=avg[:, None, None] * PROFILE[None], **common)
    np.savez(dsig, dsigma_dT_matrix=davg[:, None, None] * PROFILE[None], **common)
    return sig.getvalue(), dsig.getvalue()


def coarse_direct(weight: str, coarse_kev: np.ndarray, T: float) -> tuple[np.ndarray, np.ndarray]:
    """Directly computed coarse <f>_I and d<f>_I/dT (outgoing/angle profile collapsed)."""
    fine = make_boundaries_kev()
    sys.path.insert(0, str(ROOT / "docs" / "py"))
    from collapse import _overlap_matrix
    f_e = _overlap_matrix(fine, coarse_kev)
    f_xi = _overlap_matrix(np.linspace(-1, 1, N_ANG + 1), np.linspace(-1, 1, 3))
    H = f_e.T @ PROFILE @ f_xi
    avg = group_average(weight, coarse_kev, T)
    davg = (group_average(weight, coarse_kev, T * (1 + H_REL))
            - group_average(weight, coarse_kev, T * (1 - H_REL))) / (2 * H_REL * T)
    return avg[:, None, None] * H[None], davg[:, None, None] * H[None]


def rel_err(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def main() -> None:
    fine = make_boundaries_kev()
    grids = {
        "aligned/8": fine[::8],
        "aligned-sub": fine[40:121:5],
        "non-aligned": np.geomspace(fine[0], fine[-1], 11),
    }
    temps = [1.0e3, 3.0e5, 1.0e8, 1.0e9]
    failures = []
    for weight in ("uniform", "wien", "planck"):
        for T in temps:
            sig_b, dsig_b = fine_npz(weight, T)
            for name, coarse in grids.items():
                ref_s, ref_d = coarse_direct(weight, coarse, T)
                got_s = _collapse_to_array(sig_b, coarse.tolist(), 2)
                got_d = _collapse_dsigma_dT_to_array(sig_b, dsig_b, coarse.tolist(), 2)
                es, ed = rel_err(got_s, ref_s), rel_err(got_d, ref_d)
                aligned = name.startswith("aligned")
                ok = es < 1e-9 and ed < DERIV_TOL[weight]
                if aligned and not ok:
                    failures.append((weight, T, name))
                label = ("PASS" if ok else "FAIL") if aligned else "INFO"
                print(f"  {label}: {weight:7s} "
                      f"T={T:.1e} K {name:12s} sigma {es:.2e}  dsigma/dT {ed:.2e}")
    if failures:
        raise SystemExit(f"\n{len(failures)} aligned-grid failures: {failures}")
    print("\nAll aligned-grid checks passed.")


if __name__ == "__main__":
    main()
