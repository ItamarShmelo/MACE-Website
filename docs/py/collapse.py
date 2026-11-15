"""
Collapse a fine-grid Compton scattering matrix to a coarser energy/angle grid.

Runs in Pyodide (browser) or standalone CPython. Supports arbitrary coarse
energy boundaries and any number of uniform or custom angle bins, for tables
computed with the uniform, Wien, or Planck weight function, and for both the
cross section sigma and its temperature derivative dsigma/dT.

Incoming groups are combined with the weight function's own group integrals:
a coarse group I gets

    sigma_I = sum_g P[g, I] s_g / sum_g P[g, I],
    P[g, I] = D_g * phi[g, I],

where D_g is the (unshifted) integral of the weight over fine group g, stored
in the table as ``log_norm``, and phi[g, I] is the fraction of that integral
lying inside coarse group I. On grids whose edges coincide with fine-group
edges phi is 0 or 1 and the collapse is exact. Partial overlaps use the weight
function's analytic integral over the sub-range. Outgoing energies and angles
use width-proportional overlap fractions. Fully vectorized with numpy einsum.

The derivative of the collapsed matrix follows from the quotient rule:

    dsigma_I/dT = sum_g P[g, I] (s'_g + r[g, I] (s_g - sigma_I)) / sum_g P[g, I],
    r[g, I]     = d ln P[g, I] / dT = dlog_norm_dT[g] + d ln phi[g, I] / dT.
"""

from __future__ import annotations

import io

import numpy as np

K_BOLTZ_KEV = 8.617333262e-8  # keV / K (k_boltz / kev in MACE units)
_SNAP_RTOL = 1e-9
_DLNT_STEP = 1e-4
_GL_X, _GL_W = np.polynomial.legendre.leggauss(16)


def _overlap_matrix(fine_bounds: np.ndarray, coarse_bounds: np.ndarray) -> np.ndarray:
    """
    Build a dense overlap-fraction matrix F[g, I].

    F[g, I] = (overlap length of fine bin g with coarse bin I) / (fine bin g width).
    Rows sum to <= 1 (exactly 1 when coarse grid covers the fine bin).
    """
    G = len(fine_bounds) - 1
    N = len(coarse_bounds) - 1
    F = np.zeros((G, N))
    j_start = 0
    for g in range(G):
        f_lo, f_hi = fine_bounds[g], fine_bounds[g + 1]
        f_width = f_hi - f_lo
        if f_width <= 0:
            continue
        for j in range(j_start, N):
            c_lo, c_hi = coarse_bounds[j], coarse_bounds[j + 1]
            if c_lo >= f_hi:
                break
            if c_hi <= f_lo:
                j_start = j + 1
                continue
            overlap = min(f_hi, c_hi) - max(f_lo, c_lo)
            if overlap > 0:
                F[g, j] = overlap / f_width
    return F


def _snap_to_fine(coarse: np.ndarray, fine: np.ndarray) -> np.ndarray:
    """Snap coarse edges lying within a relative 1e-9 of a fine edge onto it."""
    idx = np.clip(np.searchsorted(fine, coarse), 1, len(fine) - 1)
    nearest = np.where(
        np.abs(fine[idx - 1] - coarse) <= np.abs(fine[idx] - coarse),
        fine[idx - 1], fine[idx],
    )
    close = np.abs(nearest - coarse) <= _SNAP_RTOL * np.abs(nearest)
    return np.where(close, nearest, coarse)


def _validate_inputs(energy_boundaries_keV, n_angle_bins, angle_boundaries):
    """Validate and parse energy/angle boundary arguments. Returns (B, coarse_xi)."""
    if (n_angle_bins is None) == (angle_boundaries is None):
        raise ValueError("Provide exactly one of n_angle_bins or angle_boundaries")

    B = np.asarray(energy_boundaries_keV, dtype=np.float64)
    if B.ndim != 1 or len(B) < 2:
        raise ValueError("energy_boundaries_keV must have at least 2 values")
    if np.any(np.isnan(B)):
        raise ValueError("energy_boundaries_keV contains NaN")
    if not np.all(np.diff(B) > 0):
        raise ValueError("energy_boundaries_keV must be strictly increasing")

    if angle_boundaries is not None:
        coarse_xi = np.asarray(angle_boundaries, dtype=np.float64)
        if coarse_xi.ndim != 1 or len(coarse_xi) < 2:
            raise ValueError("angle_boundaries must have at least 2 values")
        if not np.all(np.diff(coarse_xi) > 0):
            raise ValueError("angle_boundaries must be strictly increasing")
        if coarse_xi[0] < -1 - 1e-12 or coarse_xi[-1] > 1 + 1e-12:
            raise ValueError(
                f"angle_boundaries [{coarse_xi[0]:.6e}, {coarse_xi[-1]:.6e}] "
                f"exceed [-1, 1]"
            )
    else:
        if not isinstance(n_angle_bins, (int, np.integer)) or n_angle_bins < 1:
            raise ValueError(f"n_angle_bins must be a positive integer, got {n_angle_bins}")
        coarse_xi = np.linspace(-1.0, 1.0, n_angle_bins + 1)

    return B, coarse_xi


# ── Weight-function integrals ─────────────────────────────


def _lower_gamma_int(k: int, d: np.ndarray) -> np.ndarray:
    """I_k(d) = integral_0^d t^k e^{-t} dt for d >= 0, accurate for small d."""
    d = np.asarray(d, dtype=np.float64)
    out = np.empty_like(d)
    small = d < 1.0
    if np.any(small):
        ds = d[small][:, None]
        j = np.arange(30)
        fact = np.concatenate([[1.0], np.cumprod(np.arange(1, 30, dtype=np.float64))])
        out[small] = ds[:, 0] ** (k + 1) * np.sum((-ds) ** j / (fact * (k + 1 + j)), axis=1)
    if np.any(~small):
        dl = d[~small]
        I = -np.expm1(-dl)
        for m in range(1, k + 1):
            I = m * I - dl**m * np.exp(-dl)
        out[~small] = I
    return out


def _log_wien_integral(xa, xb, xs):
    """ln integral_xa^xb x^3 exp(-(x - xs)) dx, stable for any xa < xb."""
    d = xb - xa
    poly = (xa**3 * _lower_gamma_int(0, d) + 3 * xa**2 * _lower_gamma_int(1, d)
            + 3 * xa * _lower_gamma_int(2, d) + _lower_gamma_int(3, d))
    return np.log(poly) - (xa - xs)


def _log_planck_integral(xa, xb, xs):
    """ln( exp(xs) integral_xa^xb x^3 / (e^x - 1) dx ) by 16-point Gauss-Legendre."""
    half = 0.5 * (xb - xa)
    x = (0.5 * (xa + xb))[:, None] + half[:, None] * _GL_X
    vals = x**3 / np.expm1(x)
    return np.log(half * (vals @ _GL_W)) + xs


def _log_weight_integral(weight: str, xa, xb, xs, cap_x: float) -> np.ndarray:
    """
    ln of the weight integral over [xa, xb] (x = E/kT), up to the factor
    exp(xs) and a constant per group; only differences within one fine group
    (same xs) are meaningful.
    """
    xa = np.asarray(xa, dtype=np.float64)
    xb = np.asarray(xb, dtype=np.float64)
    xs = np.asarray(xs, dtype=np.float64)
    if weight == "uniform":
        return np.log(xb - xa)
    if weight == "wien":
        return _log_wien_integral(xa, xb, xs)
    if weight != "planck":
        raise ValueError(f"Unknown weight function {weight!r}")

    # Planck below cap_x, (unshifted-equivalent) Wien above, as in MACE.
    out = np.full(xa.shape, -np.inf)
    lo = xa < cap_x
    if np.any(lo):
        out[lo] = _log_planck_integral(xa[lo], np.minimum(xb[lo], cap_x), xs[lo])
    hi = xb > cap_x
    if np.any(hi):
        wien = _log_wien_integral(np.maximum(xa[hi], cap_x), xb[hi], xs[hi])
        out[hi] = np.logaddexp(out[hi], wien)
    return out


def _log_phi(weight, fine_bounds, coarse_bounds, F, T, cap_x):
    """
    ln phi[g, I] (fraction of fine group g's weight integral inside coarse I)
    and its T-derivative. -inf where there is no overlap.
    """
    log_phi = np.where(F > 0, 0.0, -np.inf)
    dlog_phi = np.zeros_like(F)
    partial = (F > 0) & (F < 1 - 1e-12)
    if weight == "uniform" or not np.any(partial):
        if np.any(partial):
            log_phi[partial] = np.log(F[partial])
        return log_phi, dlog_phi

    g, I = np.nonzero(partial)
    f_lo, f_hi = fine_bounds[g], fine_bounds[g + 1]
    a = np.maximum(f_lo, coarse_bounds[I])
    b = np.minimum(f_hi, coarse_bounds[I + 1])

    def at(temp):
        kT = K_BOLTZ_KEV * temp
        xs = f_lo / kT
        return (_log_weight_integral(weight, a / kT, b / kT, xs, cap_x)
                - _log_weight_integral(weight, f_lo / kT, f_hi / kT, xs, cap_x))

    log_phi[g, I] = at(T)
    dlog_phi[g, I] = (at(T * (1 + _DLNT_STEP)) - at(T * (1 - _DLNT_STEP))) / (2 * _DLNT_STEP * T)
    return log_phi, dlog_phi


# ── Table loading ─────────────────────────────────────────


def _load_table(npz_bytes: bytes) -> dict:
    """Load a sigma table plus the weight data needed to collapse it."""
    data = np.load(io.BytesIO(npz_bytes))
    sigma = np.asarray(data["sigma_matrix"], dtype=np.float64)
    bounds = np.asarray(data["boundaries_keV"], dtype=np.float64)
    if sigma.ndim != 3:
        raise ValueError(f"Expected 3D sigma_matrix, got shape {sigma.shape}")
    if "log_norm" in data.files:
        weight = str(data["weight_function"])
        log_norm = np.asarray(data["log_norm"], dtype=np.float64)
        dlog_norm = np.asarray(data["dlog_norm_dT"], dtype=np.float64)
        cap_x = float(data["cap_x"]) if "cap_x" in data.files else np.nan
    else:  # legacy uniform tables
        weight = "uniform"
        log_norm = np.log(np.diff(bounds))
        dlog_norm = np.zeros_like(log_norm)
        cap_x = np.nan
    return dict(sigma=sigma, bounds=bounds, T=float(data["temperature_K"]),
                weight=weight, log_norm=log_norm, dlog_norm=dlog_norm, cap_x=cap_x)


def _load_dsigma_dT(npz_bytes: bytes) -> tuple[np.ndarray, float]:
    """Load dsigma/dT [cm^2/K] from a raw (dsigma_dT_matrix) or site (T_dsigma_dT) table."""
    data = np.load(io.BytesIO(npz_bytes))
    T = float(data["temperature_K"])
    if "dsigma_dT_matrix" in data.files:
        return np.asarray(data["dsigma_dT_matrix"], dtype=np.float64), T
    return np.asarray(data["T_dsigma_dT"], dtype=np.float64) / T, T


# ── Collapse core ─────────────────────────────────────────


def _collapse_core(table, dsigma, energy_boundaries_keV, n_angle_bins, angle_boundaries):
    """Collapse sigma (and dsigma/dT if given). Returns (sigma_c, dsigma_c or None)."""
    B, coarse_xi = _validate_inputs(energy_boundaries_keV, n_angle_bins, angle_boundaries)
    sigma, fine_bounds = table["sigma"], table["bounds"]
    G, _, M = sigma.shape

    if B[0] < fine_bounds[0] * (1 - 1e-12) or B[-1] > fine_bounds[-1] * (1 + 1e-12):
        raise ValueError(
            f"Coarse boundaries [{B[0]:.6e}, {B[-1]:.6e}] exceed fine grid "
            f"[{fine_bounds[0]:.6e}, {fine_bounds[-1]:.6e}]"
        )
    B = _snap_to_fine(np.clip(B, fine_bounds[0], fine_bounds[-1]), fine_bounds)

    f_e = _overlap_matrix(fine_bounds, B)
    fine_xi = np.linspace(-1.0, 1.0, M + 1)
    f_xi = _overlap_matrix(fine_xi, coarse_xi)

    log_phi, dlog_phi = _log_phi(table["weight"], fine_bounds, B, f_e, table["T"], table["cap_x"])
    log_p = table["log_norm"][:, None] + log_phi
    col_max = np.max(log_p, axis=0)
    col_max = np.where(np.isfinite(col_max), col_max, 0.0)
    P = np.exp(log_p - col_max)
    Q = P.sum(axis=0)
    safe_Q = np.where(Q > 0, Q, 1.0)[:, None, None]

    s = np.einsum("gpa,aA,pJ->gJA", sigma, f_xi, f_e, optimize=True)
    result = np.einsum("gI,gJA->IJA", P, s) / safe_Q
    if dsigma is None:
        return result, None

    ds = np.einsum("gpa,aA,pJ->gJA", dsigma, f_xi, f_e, optimize=True)
    PR = P * np.where(P > 0, table["dlog_norm"][:, None] + dlog_phi, 0.0)
    dresult = (np.einsum("gI,gJA->IJA", P, ds) + np.einsum("gI,gJA->IJA", PR, s)
               - result * PR.sum(axis=0)[:, None, None]) / safe_Q
    return result, dresult


def _collapse_to_array(
    npz_bytes: bytes,
    energy_boundaries_keV: list[float],
    n_angle_bins: int | None = None,
    *,
    angle_boundaries: list[float] | None = None,
) -> np.ndarray:
    """
    Collapse a fine-grid sigma matrix to a coarser grid, returning the 3D array.

    Parameters
    ----------
    npz_bytes : bytes
        Raw bytes of a .npz file containing ``sigma_matrix`` (G, G, M),
        ``boundaries_keV`` (G+1,), ``temperature_K`` and, for weighted tables,
        ``weight_function``, ``log_norm``, ``dlog_norm_dT`` (and ``cap_x``).
    energy_boundaries_keV : list[float]
        Coarse energy group edges in keV.
    n_angle_bins : int, optional
        Number of uniform coarse angle bins.
    angle_boundaries : list[float], optional
        Explicit coarse angle bin edges.

    Returns
    -------
    np.ndarray
        Collapsed matrix with shape (N, N, K).
    """
    table = _load_table(npz_bytes)
    result, _ = _collapse_core(table, None, energy_boundaries_keV, n_angle_bins, angle_boundaries)
    return result


def _collapse_dsigma_dT_to_array(
    sigma_npz_bytes: bytes,
    dsigma_npz_bytes: bytes,
    energy_boundaries_keV: list[float],
    n_angle_bins: int | None = None,
    *,
    angle_boundaries: list[float] | None = None,
) -> np.ndarray:
    """
    Collapse dsigma/dT [cm^2/K] to a coarser grid, returning the 3D array.

    Needs both the sigma table and the dsigma/dT table at the same temperature,
    because the weights of the coarse group depend on temperature.
    """
    table = _load_table(sigma_npz_bytes)
    dsigma, T = _load_dsigma_dT(dsigma_npz_bytes)
    if dsigma.shape != table["sigma"].shape or not np.isclose(T, table["T"], rtol=1e-12):
        raise ValueError("sigma and dsigma/dT tables do not match")
    _, dresult = _collapse_core(table, dsigma, energy_boundaries_keV, n_angle_bins, angle_boundaries)
    return dresult


def collapse(
    npz_bytes: bytes,
    energy_boundaries_keV: list[float],
    n_angle_bins: int | None = None,
    *,
    angle_boundaries: list[float] | None = None,
) -> bytes:
    """
    Load a .npz from raw bytes, collapse to a coarser grid, return .npy bytes.

    Parameters
    ----------
    npz_bytes : bytes
        Raw bytes of a sigma .npz file (see ``_collapse_to_array``).
    energy_boundaries_keV : list[float]
        Coarse energy group edges in keV.  Length N+1, strictly increasing.
        Values must lie within the fine-grid range.
    n_angle_bins : int, optional
        Number of uniform coarse angle bins in xi = cos(theta) over [-1, 1].
        Mutually exclusive with ``angle_boundaries``.
    angle_boundaries : list[float], optional
        Explicit coarse angle bin edges in xi = cos(theta).  Must be strictly
        increasing and lie within [-1, 1].  Mutually exclusive with
        ``n_angle_bins``.

    Returns
    -------
    bytes
        Raw bytes of a .npy file containing the collapsed float64 array.
        Shape is (N, N, K), reduced for edge cases:
        - (N, N) if K == 1
        - (K,) if N == 1
        - scalar if N == 1 and K == 1
    """
    result = _collapse_to_array(
        npz_bytes, energy_boundaries_keV, n_angle_bins,
        angle_boundaries=angle_boundaries,
    )
    N, _, K = result.shape

    if N == 1 and K == 1:
        result = result.ravel()[0]
    elif N == 1:
        result = result[0, 0, :]
    elif K == 1:
        result = result[:, :, 0]

    buf = io.BytesIO()
    np.save(buf, result)
    return buf.getvalue()


def _log_t_alpha(T_lo: float, T_hi: float, T_target: float) -> float:
    return (np.log(T_target) - np.log(T_lo)) / (np.log(T_hi) - np.log(T_lo))


def _interp_sigma(sigma_lo: np.ndarray, sigma_hi: np.ndarray, alpha: float) -> np.ndarray:
    """
    Interpolate sigma between two temperatures at weight alpha in ln T.

    Linear in ln(sigma) (a power law in T) where both neighbours are
    positive, which follows the steep temperature dependence of the kernel
    tails; plain linear interpolation elsewhere (zeros from the cutoff).
    """
    linear = (1.0 - alpha) * sigma_lo + alpha * sigma_hi
    positive = (sigma_lo > 0) & (sigma_hi > 0)
    log_lo = np.log(np.where(positive, sigma_lo, 1.0))
    log_hi = np.log(np.where(positive, sigma_hi, 1.0))
    return np.where(positive, np.exp((1.0 - alpha) * log_lo + alpha * log_hi), linear)


def collapse_interp(
    npz_lo_bytes: bytes,
    npz_hi_bytes: bytes,
    T_lo: float,
    T_hi: float,
    T_target: float,
    energy_boundaries_keV: list[float],
    n_angle_bins: int | None = None,
    *,
    angle_boundaries: list[float] | None = None,
) -> np.ndarray:
    """
    Collapse two temperature grids and interpolate sigma in log-T space
    (see ``_interp_sigma``).

    Parameters
    ----------
    npz_lo_bytes : bytes
        Raw bytes of the lower-temperature .npz file.
    npz_hi_bytes : bytes
        Raw bytes of the higher-temperature .npz file.
    T_lo : float
        Temperature (K) of the lower bracket.
    T_hi : float
        Temperature (K) of the upper bracket.
    T_target : float
        Desired interpolation temperature (K). Must satisfy T_lo <= T_target <= T_hi.
    energy_boundaries_keV : list[float]
        Coarse energy group edges in keV.
    n_angle_bins : int, optional
        Number of uniform coarse angle bins.
    angle_boundaries : list[float], optional
        Explicit coarse angle bin edges.

    Returns
    -------
    np.ndarray
        Interpolated collapsed matrix with shape (N, N, K).
    """
    alpha = _log_t_alpha(T_lo, T_hi, T_target)

    sigma_lo = _collapse_to_array(
        npz_lo_bytes, energy_boundaries_keV, n_angle_bins,
        angle_boundaries=angle_boundaries,
    )
    sigma_hi = _collapse_to_array(
        npz_hi_bytes, energy_boundaries_keV, n_angle_bins,
        angle_boundaries=angle_boundaries,
    )

    return _interp_sigma(sigma_lo, sigma_hi, alpha)


def _matrix_at(quantity, sigma_bytes, dsigma_bytes, energy_boundaries_keV,
               n_angle_bins, angle_boundaries, identity):
    if identity:
        if quantity == "sigma":
            return _load_table(sigma_bytes)["sigma"]
        return _load_dsigma_dT(dsigma_bytes)[0]
    if quantity == "sigma":
        return _collapse_to_array(sigma_bytes, energy_boundaries_keV, n_angle_bins,
                                  angle_boundaries=angle_boundaries)
    return _collapse_dsigma_dT_to_array(sigma_bytes, dsigma_bytes, energy_boundaries_keV,
                                        n_angle_bins, angle_boundaries=angle_boundaries)


def matrix_at_temperature(
    quantity: str,
    lo_files: tuple[bytes, bytes | None],
    hi_files: tuple[bytes, bytes | None] | None,
    T_lo: float,
    T_hi: float,
    T_target: float,
    energy_boundaries_keV: list[float],
    n_angle_bins: int | None = None,
    *,
    angle_boundaries: list[float] | None = None,
    identity: bool = False,
) -> np.ndarray:
    """
    Entry point for the website: one (N, N, K) float64 matrix at T_target.

    ``quantity`` is "sigma" [cm^2] or "dsigma_dT" [cm^2/K]. ``lo_files`` and
    ``hi_files`` are (sigma_bytes, dsigma_bytes) at the bracketing stored
    temperatures (dsigma_bytes may be None for sigma); ``hi_files`` is None
    when T_target is a stored temperature. ``identity`` skips the collapse
    when the requested grid equals the stored one.
    """
    args = (energy_boundaries_keV, n_angle_bins, angle_boundaries, identity)
    a_lo = _matrix_at(quantity, *lo_files, *args)
    if hi_files is None:
        return np.asarray(a_lo, dtype=np.float64)
    a_hi = _matrix_at(quantity, *hi_files, *args)
    alpha = _log_t_alpha(T_lo, T_hi, T_target)
    if quantity == "sigma":
        return _interp_sigma(a_lo, a_hi, alpha)
    return ((1.0 - alpha) * T_lo * a_lo + alpha * T_hi * a_hi) / T_target


def collapse_dsigma_dT_interp(
    sigma_lo_bytes: bytes,
    dsigma_lo_bytes: bytes,
    sigma_hi_bytes: bytes,
    dsigma_hi_bytes: bytes,
    T_lo: float,
    T_hi: float,
    T_target: float,
    energy_boundaries_keV: list[float],
    n_angle_bins: int | None = None,
    *,
    angle_boundaries: list[float] | None = None,
) -> np.ndarray:
    """
    Collapse dsigma/dT at two temperatures and interpolate in log-T space.

    T * dsigma/dT (= dsigma/dlnT) is interpolated linearly in ln T and divided
    by T_target. Returns dsigma/dT [cm^2/K] with shape (N, N, K).
    """
    alpha = _log_t_alpha(T_lo, T_hi, T_target)
    kwargs = dict(angle_boundaries=angle_boundaries)

    d_lo = _collapse_dsigma_dT_to_array(
        sigma_lo_bytes, dsigma_lo_bytes, energy_boundaries_keV, n_angle_bins, **kwargs)
    d_hi = _collapse_dsigma_dT_to_array(
        sigma_hi_bytes, dsigma_hi_bytes, energy_boundaries_keV, n_angle_bins, **kwargs)

    return ((1.0 - alpha) * T_lo * d_lo + alpha * T_hi * d_hi) / T_target
