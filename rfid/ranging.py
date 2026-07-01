"""RSSI -> distance -> 2D location.

The maths here is the standard **log-distance path-loss** model plus a couple of
helpers for turning a set of per-antenna distance estimates into an approximate
(x, y) position on a 2D field.

Log-distance path-loss model
----------------------------
For a tag at distance ``d`` from an antenna::

    RSSI(d) = RSSI(d0) - 10 * n * log10(d / d0) + X

where

* ``RSSI(d0)`` is the RSSI measured at a known reference distance ``d0``
  (the calibration anchor, e.g. -45 dBm at 1 m),
* ``n`` is the environment path-loss exponent (~2 in free space, 2-4 indoors),
* ``X`` is zero-mean Gaussian shadowing noise.

Inverting for distance::

    d = d0 * 10 ** ((RSSI(d0) - RSSI) / (10 * n))

Phase-based ranging
-------------------
Zebra FX readers can also report the RF phase angle ``phi`` (0..2*pi).  Because
the signal travels to the tag and back, phase relates to distance as::

    phi = (4 * pi * d / lambda + phi0) mod 2*pi

with ``lambda = c / f`` (~0.328 m at 915 MHz).  Phase is very precise but
*ambiguous* every ``lambda / 2`` (~16 cm), so we use it for fine motion /
velocity rather than absolute range.  See :func:`phase_velocity`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .models import AntennaConfig

# Speed of light, m/s.
_C = 299_792_458.0
# US FCC UHF RFID band centre, Hz (902-928 MHz).  ETSI EU is ~866 MHz.
DEFAULT_FREQ_HZ = 915_000_000.0


def wavelength(freq_hz: float = DEFAULT_FREQ_HZ) -> float:
    """Carrier wavelength in metres."""
    return _C / freq_hz


# --------------------------------------------------------------------------- #
# RSSI  <->  distance
# --------------------------------------------------------------------------- #
def expected_rssi(distance_m: float, cfg: AntennaConfig) -> float:
    """Forward model: RSSI a tag *should* produce at ``distance_m``."""
    distance_m = max(distance_m, 1e-3)
    return cfg.rssi_ref - 10.0 * cfg.path_loss_n * math.log10(distance_m / cfg.d_ref)


def rssi_to_distance(rssi_dbm: float, cfg: AntennaConfig) -> float:
    """Invert the path-loss model to an approximate distance in metres."""
    exponent = (cfg.rssi_ref - rssi_dbm) / (10.0 * cfg.path_loss_n)
    return cfg.d_ref * (10.0 ** exponent)


def distance_uncertainty(rssi_dbm: float, cfg: AntennaConfig, rssi_sigma: float = 3.0) -> float:
    """1-sigma distance error band (metres) implied by RSSI noise ``rssi_sigma``.

    Because distance is exponential in RSSI, a fixed dB error maps to a
    *multiplicative* distance error, so the band grows with range.
    """
    d = rssi_to_distance(rssi_dbm, cfg)
    # d * ln(10) / (10 n) * sigma  (first-order propagation of error)
    return d * math.log(10.0) / (10.0 * cfg.path_loss_n) * rssi_sigma


# --------------------------------------------------------------------------- #
# Calibration:  fit (rssi_ref, n) from labelled (distance, rssi) samples
# --------------------------------------------------------------------------- #
def calibrate_path_loss(
    distances_m: Sequence[float],
    rssis_dbm: Sequence[float],
    d_ref: float = 1.0,
) -> Tuple[float, float, float]:
    """Least-squares fit of the path-loss model.

    Returns ``(rssi_ref, path_loss_n, rms_error_db)`` where ``rssi_ref`` is the
    fitted RSSI at ``d_ref``.  Model is linear in ``log10(d)``::

        RSSI = a + b * log10(d),   b = -10 n,   rssi_ref = a + b*log10(d_ref)
    """
    d = np.asarray(distances_m, dtype=float)
    r = np.asarray(rssis_dbm, dtype=float)
    mask = d > 0
    d, r = d[mask], r[mask]
    if len(d) < 2:
        raise ValueError("Need at least two distinct calibration samples.")
    x = np.log10(d)
    # r = a + b*x
    b, a = np.polyfit(x, r, 1)
    n = -b / 10.0
    rssi_ref = a + b * math.log10(d_ref)
    pred = a + b * x
    rms = float(np.sqrt(np.mean((r - pred) ** 2)))
    return float(rssi_ref), float(n), rms


# --------------------------------------------------------------------------- #
# Phase -> fine motion
# --------------------------------------------------------------------------- #
def phase_velocity(
    phases_rad: Sequence[float],
    times_s: Sequence[float],
    freq_hz: float = DEFAULT_FREQ_HZ,
) -> Optional[float]:
    """Estimate radial speed (m/s) from a short run of phase samples.

    Unwraps phase, fits a line, and converts the phase rate to a range rate via
    the round-trip relation ``phi = 4*pi*d/lambda`` -> ``dr/dt = (lambda/4pi) *
    dphi/dt``.  Returns ``None`` if too few samples.

    Sign note: the *magnitude* is the reliable output.  The sign of the reported
    phase-vs-range slope is reader-convention dependent (some readers report
    phase increasing with range, others decreasing); this uses the same
    convention as :mod:`rfid.simulator` (phase increases with range, so a
    positive result means moving away).
    """
    if len(phases_rad) < 2 or len(phases_rad) != len(times_s):
        return None
    ph = np.unwrap(np.asarray(phases_rad, dtype=float))
    t = np.asarray(times_s, dtype=float)
    if t[-1] - t[0] <= 0:
        return None
    dphi_dt = np.polyfit(t, ph, 1)[0]
    lam = wavelength(freq_hz)
    return float(dphi_dt * lam / (4.0 * math.pi))


# --------------------------------------------------------------------------- #
# 2D localization from one or more antennas
# --------------------------------------------------------------------------- #
@dataclass
class Fix:
    """A position estimate on the 2D field."""

    x: float
    y: float
    radius_m: float          # 1-sigma positional uncertainty
    n_antennas: int
    method: str              # "arc", "trilateration"


def _ring_point(cfg: AntennaConfig, distance_m: float) -> Tuple[float, float]:
    """Best-guess point for a single antenna: along its heading at ``distance``."""
    theta = math.radians(cfg.heading_deg)
    return (cfg.x + distance_m * math.cos(theta), cfg.y + distance_m * math.sin(theta))


def localize(
    ranges: Dict[int, float],
    antennas: Dict[int, AntennaConfig],
    rssi_sigma: float = 3.0,
) -> Optional[Fix]:
    """Estimate a tag's (x, y) from per-antenna distance estimates.

    * 1 antenna  -> point along the antenna heading at the estimated range
      (a single reader can only give range + coarse bearing).
    * 2+ antennas -> weighted least-squares multilateration of the circle
      intersections, weighting near antennas (lower range error) more.

    ``ranges`` maps antenna_id -> distance_m.  Returns ``None`` if empty.
    """
    used = [(aid, r) for aid, r in ranges.items() if aid in antennas]
    if not used:
        return None

    if len(used) == 1:
        aid, dist = used[0]
        cfg = antennas[aid]
        x, y = _ring_point(cfg, dist)
        band = distance_uncertainty(expected_rssi(dist, cfg), cfg, rssi_sigma)
        return Fix(x, y, max(band, 0.1), 1, "arc")

    # Multilateration: minimise sum_i w_i (||p - a_i|| - r_i)^2 by Gauss-Newton.
    pts = np.array([antennas[a].position for a, _ in used], dtype=float)
    rads = np.array([r for _, r in used], dtype=float)
    # Weight ~ 1 / range^2  (closer antennas are more trustworthy).
    weights = 1.0 / np.clip(rads, 0.2, None) ** 2

    p = pts.mean(axis=0)  # initial guess: centroid of antennas
    for _ in range(50):
        diff = p - pts
        dist = np.linalg.norm(diff, axis=1)
        dist = np.clip(dist, 1e-6, None)
        residual = dist - rads
        # Jacobian rows: unit vectors from antenna to p.
        jac = diff / dist[:, None]
        w = weights[:, None]
        # Weighted normal equations.
        h = (jac * w).T @ jac
        g = (jac * w).T @ residual
        try:
            step = np.linalg.solve(h, g)
        except np.linalg.LinAlgError:
            break
        p = p - step
        if np.linalg.norm(step) < 1e-4:
            break

    # Residual spread -> uncertainty radius.
    dist = np.linalg.norm(p - pts, axis=1)
    rms = float(np.sqrt(np.mean((dist - rads) ** 2)))
    band = max(rms, 0.1)
    return Fix(float(p[0]), float(p[1]), band, len(used), "trilateration")


def aoa_to_xy(
    azimuth_deg: float,
    elevation_deg: float,
    reader_xy: Tuple[float, float] = (0.0, 0.0),
    mount_height_m: float = 4.0,
    tag_height_m: float = 0.0,
) -> Tuple[float, float]:
    """Overhead AoA (Zebra ATR7000) azimuth/elevation -> floor (x, y).

    For an antenna mounted at ``mount_height_m`` looking down, a beam at
    ``elevation_deg`` off the downward boresight hits the floor at radius
    ``r = (H - tag_height) * tan(elevation)`` and ``azimuth_deg`` sets the
    direction (measured like a compass bearing, 0 = +y / "north").
    """
    h = max(mount_height_m - tag_height_m, 1e-3)
    r = h * math.tan(math.radians(elevation_deg))
    az = math.radians(azimuth_deg)
    x = reader_xy[0] + r * math.sin(az)
    y = reader_xy[1] + r * math.cos(az)
    return (x, y)


def rssi_heatmap(
    antennas: Iterable[AntennaConfig],
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    resolution: int = 80,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Grid of the *best* expected RSSI from any antenna over the field.

    Returns ``(xs, ys, grid)`` for a plotly heatmap/contour.
    """
    antennas = list(antennas)
    xs = np.linspace(xlim[0], xlim[1], resolution)
    ys = np.linspace(ylim[0], ylim[1], resolution)
    grid = np.full((resolution, resolution), -120.0)
    for cfg in antennas:
        gx, gy = np.meshgrid(xs, ys)
        d = np.sqrt((gx - cfg.x) ** 2 + (gy - cfg.y) ** 2)
        d = np.clip(d, 1e-3, None)
        rssi = cfg.rssi_ref - 10.0 * cfg.path_loss_n * np.log10(d / cfg.d_ref)
        grid = np.maximum(grid, rssi)
    return xs, ys, grid
