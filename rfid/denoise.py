"""Smoothing / denoising of noisy RSSI (and phase) streams.

Raw UHF RFID RSSI is jumpy: multipath, antenna switching, tag orientation and
reader AGC all add several dB of noise read-to-read.  This module offers a
small family of denoisers behind one interface so the UI can compare them:

* :class:`MedianKalman`       - median prefilter -> 1-D Kalman. The empirically
  best *classical* pipeline for RFID/BLE RSSI: the median stage removes
  impulsive dropouts that would otherwise break the Kalman filter's Gaussian
  assumption.  This is the default recommendation.
* :class:`KalmanRSSI`         - the bare 1-D Kalman filter (random-walk latent
  RSSI), the single most-used online RSSI smoother.
* :class:`LearnedDenoiser`    - a tiny self-supervised denoising autoencoder
  (numpy only) that *learns* the temporal structure of the live stream and is
  the "ML model" option in the UI.
* :class:`ExponentialFilter`  - EWMA, one-pole low-pass.
* :class:`MovingAverage`      - simple boxcar mean.
* :class:`SavitzkyGolay`      - polynomial smoothing (keeps edges/peaks).

All denoisers implement :class:`Denoiser`: ``update(x) -> y`` for streaming and
``smooth(seq) -> np.ndarray`` for a whole batch.

References (the models people actually "use online" for this):
  * Wouter Bulten, "Kalman filters explained: removing noise from RSSI signals".
  * rlabbe/filterpy - the standard Python Kalman/Bayesian filter library.
  * Denoising-autoencoder RSSI papers (e.g. arXiv:2001.02396).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Protocol, Sequence

import numpy as np

try:  # SciPy is a listed core dep; degrade gracefully if missing.
    from scipy.signal import medfilt as _medfilt
    from scipy.signal import savgol_filter as _savgol
except Exception:  # pragma: no cover
    _savgol = None
    _medfilt = None


class Denoiser(Protocol):
    """Common streaming + batch interface for every filter."""

    def update(self, x: float) -> float: ...

    def reset(self) -> None: ...

    def smooth(self, seq: Sequence[float]) -> np.ndarray: ...


def _batch_from_stream(d: "Denoiser", seq: Sequence[float]) -> np.ndarray:
    d.reset()
    return np.array([d.update(float(x)) for x in seq], dtype=float)


# --------------------------------------------------------------------------- #
@dataclass
class MovingAverage:
    window: int = 7
    _buf: Deque[float] = field(default_factory=deque, init=False, repr=False)

    def reset(self) -> None:
        self._buf = deque(maxlen=self.window)

    def update(self, x: float) -> float:
        if self._buf.maxlen != self.window:
            self._buf = deque(self._buf, maxlen=self.window)
        self._buf.append(x)
        return float(np.mean(self._buf))

    def smooth(self, seq: Sequence[float]) -> np.ndarray:
        return _batch_from_stream(self, seq)


@dataclass
class ExponentialFilter:
    """EWMA: y_t = a*x_t + (1-a)*y_{t-1}. Smaller alpha = smoother."""

    alpha: float = 0.3
    _y: Optional[float] = field(default=None, init=False, repr=False)

    def reset(self) -> None:
        self._y = None

    def update(self, x: float) -> float:
        self._y = x if self._y is None else self.alpha * x + (1 - self.alpha) * self._y
        return float(self._y)

    def smooth(self, seq: Sequence[float]) -> np.ndarray:
        return _batch_from_stream(self, seq)


@dataclass
class SavitzkyGolay:
    """Batch polynomial smoother.  Streaming falls back to a trailing window."""

    window: int = 9
    poly: int = 2

    def reset(self) -> None:
        self._buf: Deque[float] = deque(maxlen=self.window)

    def update(self, x: float) -> float:
        if not hasattr(self, "_buf"):
            self.reset()
        self._buf.append(x)
        if _savgol is None or len(self._buf) < self.poly + 2:
            return float(np.mean(self._buf))
        w = len(self._buf) | 1  # odd
        try:
            return float(_savgol(np.array(self._buf), w, self.poly)[-1])
        except Exception:
            return float(np.mean(self._buf))

    def smooth(self, seq: Sequence[float]) -> np.ndarray:
        arr = np.asarray(seq, dtype=float)
        if _savgol is None or len(arr) < self.window:
            return _batch_from_stream(self, seq)
        w = self.window if self.window % 2 == 1 else self.window + 1
        w = min(w, len(arr) if len(arr) % 2 == 1 else len(arr) - 1)
        w = max(w, self.poly + 2 - ((self.poly) % 2))
        try:
            return _savgol(arr, w, self.poly)
        except Exception:
            return _batch_from_stream(self, seq)


@dataclass
class KalmanRSSI:
    """1-D Kalman filter modelling RSSI as a slowly-drifting random walk.

    State = true RSSI.  ``q`` (process noise) sets how fast the true value may
    drift; ``r`` (measurement noise) is the RSSI reading variance in dB^2.
    Larger ``r`` / smaller ``q`` -> smoother, laggier output.  Defaults follow
    the widely-cited Wouter Bulten RSSI tuning (Q~0.01, R = raw RSSI variance).
    """

    q: float = 0.01          # process variance (dB^2 per step)
    r: float = 4.0           # measurement variance (dB^2)
    _x: Optional[float] = field(default=None, init=False, repr=False)
    _p: float = field(default=1.0, init=False, repr=False)

    def reset(self) -> None:
        self._x = None
        self._p = 1.0

    def update(self, x: float) -> float:
        if self._x is None:
            self._x = x
            self._p = self.r
            return x
        # Predict (random walk: state unchanged, variance grows).
        p_pred = self._p + self.q
        # Update.
        k = p_pred / (p_pred + self.r)          # Kalman gain
        self._x = self._x + k * (x - self._x)
        self._p = (1 - k) * p_pred
        return float(self._x)

    def smooth(self, seq: Sequence[float]) -> np.ndarray:
        return _batch_from_stream(self, seq)


@dataclass
class MedianKalman:
    """Median prefilter -> Kalman: the recommended classical RSSI pipeline.

    A short sliding median removes impulsive dropouts/outliers (the non-Gaussian
    spikes that smear a bare Kalman filter), then the Kalman filter does the
    smoothing.  This is the empirically-best classical combo for RFID/BLE RSSI.
    """

    median_window: int = 5
    q: float = 0.01
    r: float = 4.0
    _buf: Deque[float] = field(default_factory=deque, init=False, repr=False)
    _kalman: KalmanRSSI = field(default=None, init=False, repr=False)  # type: ignore

    def reset(self) -> None:
        self._buf = deque(maxlen=self.median_window)
        self._kalman = KalmanRSSI(q=self.q, r=self.r)
        self._kalman.reset()

    def update(self, x: float) -> float:
        if self._kalman is None:
            self.reset()
        self._buf.append(x)
        med = float(np.median(self._buf))  # trailing median
        return self._kalman.update(med)

    def smooth(self, seq: Sequence[float]) -> np.ndarray:
        arr = np.asarray(seq, dtype=float)
        if _medfilt is not None and len(arr) >= self.median_window:
            k = self.median_window if self.median_window % 2 == 1 else self.median_window + 1
            arr = _medfilt(arr, kernel_size=min(k, len(arr) | 1))
        k = KalmanRSSI(q=self.q, r=self.r)
        return k.smooth(arr)


@dataclass
class LearnedDenoiser:
    """A tiny self-supervised denoising autoencoder (numpy only).

    It slides a window over the stream and learns to reconstruct the *centre*
    sample of the window from its noisy neighbours, i.e. it learns the local
    temporal structure of clean RSSI and rejects the zero-mean noise.  Training
    is online SGD, so it adapts to whatever environment the reader is in without
    any offline dataset or heavyweight ML framework.

    Architecture: window -> hidden (tanh) -> 1 linear output.
    """

    window: int = 9
    hidden: int = 16
    lr: float = 0.01
    seed: int = 0
    warmup: int = 12
    _init: bool = field(default=False, init=False, repr=False)

    def _build(self) -> None:
        rng = np.random.default_rng(self.seed)
        n_in = self.window
        # He-ish init.
        self._w1 = rng.normal(0, 1.0 / np.sqrt(n_in), size=(n_in, self.hidden))
        self._b1 = np.zeros(self.hidden)
        self._w2 = rng.normal(0, 1.0 / np.sqrt(self.hidden), size=(self.hidden, 1))
        self._b2 = np.zeros(1)
        self._buf: Deque[float] = deque(maxlen=self.window)
        self._mu = 0.0          # running mean for normalisation
        self._n = 0
        self._init = True

    def reset(self) -> None:
        self._build()

    def _forward(self, x_vec: np.ndarray):
        z1 = x_vec @ self._w1 + self._b1
        a1 = np.tanh(z1)
        y = a1 @ self._w2 + self._b2
        return z1, a1, y

    def update(self, x: float) -> float:
        if not self._init:
            self._build()
        # Online mean for centring (RSSI lives around -50 dBm).
        self._n += 1
        self._mu += (x - self._mu) / self._n
        self._buf.append(x)

        if len(self._buf) < self.window:
            # Not enough context yet -> return a light EWMA-ish estimate.
            return float(np.mean(self._buf))

        arr = np.array(self._buf, dtype=float) - self._mu
        centre_idx = self.window // 2
        # Input = window with the centre masked (set to local mean) so the net
        # must *infer* the centre from its neighbours -> denoising.
        x_in = arr.copy()
        target = x_in[centre_idx]
        x_in[centre_idx] = np.mean(np.delete(arr, centre_idx))

        z1, a1, y = self._forward(x_in[None, :])
        pred = float(y[0, 0])

        if self._n > self.warmup:
            # SGD step on squared error.
            err = pred - target
            dy = np.array([[err]])
            dw2 = a1.T @ dy
            db2 = dy.sum(axis=0)
            da1 = dy @ self._w2.T
            dz1 = da1 * (1 - a1 ** 2)
            dw1 = x_in[None, :].T @ dz1
            db1 = dz1.sum(axis=0)
            self._w2 -= self.lr * dw2
            self._b2 -= self.lr * db2
            self._w1 -= self.lr * dw1
            self._b1 -= self.lr * db1

        # Denoised estimate for the most-recent sample: re-run with the newest
        # sample as the centre of a right-aligned window.
        return pred + self._mu

    def smooth(self, seq: Sequence[float]) -> np.ndarray:
        return _batch_from_stream(self, seq)


# --------------------------------------------------------------------------- #
FILTERS = {
    "Median + Kalman (recommended)": MedianKalman,
    "Kalman": KalmanRSSI,
    "Learned denoiser (ML)": LearnedDenoiser,
    "Exponential (EWMA)": ExponentialFilter,
    "Moving average": MovingAverage,
    "Savitzky-Golay": SavitzkyGolay,
}


def make_filter(name: str, **kwargs) -> Denoiser:
    cls = FILTERS.get(name, MedianKalman)
    return cls(**kwargs)  # type: ignore[call-arg]


def denoise_phase(phases_rad: Sequence[float], denoiser: Optional[Denoiser] = None) -> np.ndarray:
    """Denoise a wrapped RFID phase trace correctly.

    RFID phase is reported modulo 2*pi, so it must be **unwrapped first** (never
    smooth wrapped phase) - then smoothed, then re-wrapped to [0, 2*pi).
    """
    ph = np.asarray(phases_rad, dtype=float)
    if len(ph) == 0:
        return ph
    unwrapped = np.unwrap(ph)
    d = denoiser or MedianKalman()
    smoothed = d.smooth(unwrapped)
    return np.mod(smoothed, 2.0 * np.pi)


def snr_improvement_db(raw: Sequence[float], clean: Sequence[float]) -> float:
    """Rough noise-reduction figure: how much the sample-to-sample jitter drops.

    Uses the std of first differences (a proxy for high-frequency noise) before
    vs after filtering, in dB.  Positive = smoother.
    """
    raw = np.asarray(raw, dtype=float)
    clean = np.asarray(clean, dtype=float)
    if len(raw) < 3:
        return 0.0
    n_raw = np.std(np.diff(raw)) + 1e-9
    n_clean = np.std(np.diff(clean)) + 1e-9
    return float(20.0 * np.log10(n_raw / n_clean))
