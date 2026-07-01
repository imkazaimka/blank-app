"""Tool 4 - Obstruction / line-of-sight blockage detection.

When something (a person, metal, liquid) moves between a reader antenna and a
tag, the direct ray is attenuated and the link starts living off weaker,
fluctuating reflections.  That shows up as four correlated symptoms:

1. **Mean RSSI drops** vs. the calibrated clear-LOS value for that geometry
   (a human body is ~3-8 dB at 900 MHz; metal/liquid can kill reads entirely).
2. **RSSI variance rises** - the dominant ray is gone, so reflections dominate
   (variance/kurtosis is the single most discriminative LOS-vs-NLOS feature).
3. **Read rate falls** - fewer decodes get through; a fully blocked path yields
   *no reads at all*, which the detector must treat as strong evidence.
4. **Phase jumps** that the tag's own motion/Doppler does not explain.

Because absolute RSSI is dominated by *distance* (see :mod:`rfid.ranging`), we
threshold the **delta from a per-(antenna, tag) baseline**, not raw RSSI.  The
detector is a fast, explainable threshold vote; an optional numpy logistic
regression (:class:`ObstructionClassifier`) can be trained on simulator-labelled
windows for a learned second opinion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .models import AntennaConfig, TagRead
from .ranging import expected_rssi, localize, rssi_to_distance

try:
    from scipy.stats import kurtosis as _kurtosis
except Exception:  # pragma: no cover
    _kurtosis = None


# Starting thresholds (all tunable in the UI).  Defaults are matched to the
# simulator: clear-LOS std ~= noise_sigma (2.5 dB), a 6 dB blocker pushes the
# mean down >5 dB and the std above ~4.5 dB.
CLEAR_STD_DB = 3.0        # a clean LOS link sits below this RSSI std
RSSI_DROP_DB = 5.0        # obstruction suspected when mean drops >= this
OBST_STD_DB = 4.5         # ...and the std climbs to at least this
READ_RATE_LOW = 0.5       # observed/expected read-rate ratio below this = blocked
KURTOSIS_HI = 1.0         # heavy-tailed RSSI (reflections) supports NLOS


@dataclass
class Baseline:
    """Clear-LOS reference statistics for one (antenna, EPC) pair."""

    antenna: int
    epc: str
    mean_rssi: float
    std_rssi: float
    expected_read_rate: float  # reads per second under clear LOS

    @property
    def key(self) -> Tuple[int, str]:
        return (self.antenna, self.epc)


@dataclass
class WindowFeatures:
    """Features extracted from one sliding window of reads for a pair."""

    antenna: int
    epc: str
    n_reads: int
    duration_s: float
    rssi_mean: float = float("nan")
    rssi_std: float = 0.0
    rssi_range: float = 0.0
    rssi_entropy: float = 0.0
    rssi_kurtosis: float = 0.0
    read_rate: float = 0.0
    read_rate_ratio: float = 1.0
    rssi_delta: float = 0.0       # rssi_mean - baseline.mean_rssi
    phase_jump: float = 0.0
    doppler_std: float = 0.0

    def vector(self) -> np.ndarray:
        """Ordered feature vector for the ML classifier."""
        return np.array(
            [
                self.rssi_delta,
                self.rssi_std,
                self.rssi_range,
                self.rssi_entropy,
                self.rssi_kurtosis,
                self.read_rate_ratio,
                self.phase_jump,
                self.doppler_std,
            ],
            dtype=float,
        )


@dataclass
class Verdict:
    """Per-pair obstruction decision with human-readable reasons."""

    antenna: int
    epc: str
    obstructed: bool
    confidence: float
    reasons: List[str]
    features: WindowFeatures

    @property
    def status(self) -> str:
        return "OBSTRUCTED" if self.obstructed else "CLEAR"


FEATURE_NAMES = [
    "rssi_delta", "rssi_std", "rssi_range", "rssi_entropy",
    "rssi_kurtosis", "read_rate_ratio", "phase_jump", "doppler_std",
]


# --------------------------------------------------------------------------- #
# Feature helpers
# --------------------------------------------------------------------------- #
def _entropy(values: np.ndarray, bins: int = 12) -> float:
    if len(values) < 2:
        return 0.0
    hist, _ = np.histogram(values, bins=bins)
    p = hist[hist > 0] / hist.sum()
    return float(-(p * np.log2(p)).sum())


def _kurt(values: np.ndarray) -> float:
    if len(values) < 4:
        return 0.0
    if _kurtosis is not None:
        return float(_kurtosis(values, fisher=True, bias=False))
    m = values.mean()
    s = values.std()
    if s < 1e-9:
        return 0.0
    return float(np.mean(((values - m) / s) ** 4) - 3.0)


def group_reads(reads: Sequence[TagRead]) -> Dict[Tuple[int, str], List[TagRead]]:
    groups: Dict[Tuple[int, str], List[TagRead]] = {}
    for r in reads:
        groups.setdefault((r.antenna, r.epc), []).append(r)
    for g in groups.values():
        g.sort(key=lambda r: r.timestamp)
    return groups


def extract_features(
    reads: Sequence[TagRead],
    antenna: int,
    epc: str,
    baseline: Optional[Baseline] = None,
    duration_s: Optional[float] = None,
) -> WindowFeatures:
    """Compute window features for one (antenna, epc) pair."""
    rssis = np.array([r.rssi for r in reads], dtype=float)
    if duration_s is None:
        if len(reads) >= 2:
            duration_s = max(reads[-1].timestamp - reads[0].timestamp, 1e-3)
        else:
            duration_s = 1.0
    feats = WindowFeatures(
        antenna=antenna, epc=epc, n_reads=len(reads), duration_s=duration_s,
    )
    if len(rssis) == 0:
        # No reads at all -> potential full blockage; leave a zero read rate.
        feats.read_rate = 0.0
        feats.read_rate_ratio = 0.0
        if baseline is not None:
            feats.rssi_delta = -RSSI_DROP_DB * 2  # push strongly negative
        return feats

    feats.rssi_mean = float(rssis.mean())
    feats.rssi_std = float(rssis.std())
    feats.rssi_range = float(rssis.max() - rssis.min())
    feats.rssi_entropy = _entropy(rssis)
    feats.rssi_kurtosis = _kurt(rssis)
    feats.read_rate = len(reads) / duration_s
    if baseline is not None:
        feats.rssi_delta = feats.rssi_mean - baseline.mean_rssi
        if baseline.expected_read_rate > 0:
            feats.read_rate_ratio = feats.read_rate / baseline.expected_read_rate

    phases = [r.phase for r in reads if r.phase is not None]
    if len(phases) >= 2:
        unwrapped = np.unwrap(np.array(phases, dtype=float))
        # Jump = largest step not explained by a smooth trend.
        steps = np.abs(np.diff(unwrapped))
        feats.phase_jump = float(np.max(steps) - np.median(steps)) if len(steps) else 0.0
    dopplers = [r.doppler for r in reads if r.doppler is not None]
    if len(dopplers) >= 2:
        feats.doppler_std = float(np.std(dopplers))
    return feats


# --------------------------------------------------------------------------- #
# Baseline calibration
# --------------------------------------------------------------------------- #
def build_baselines(
    reads: Sequence[TagRead], duration_s: Optional[float] = None
) -> Dict[Tuple[int, str], Baseline]:
    """Fit clear-LOS baselines from a short 'known clear' capture."""
    groups = group_reads(reads)
    out: Dict[Tuple[int, str], Baseline] = {}
    for (ant, epc), g in groups.items():
        rssis = np.array([r.rssi for r in g], dtype=float)
        if duration_s is None:
            dur = max(g[-1].timestamp - g[0].timestamp, 1e-3) if len(g) >= 2 else 1.0
        else:
            dur = duration_s
        out[(ant, epc)] = Baseline(
            antenna=ant,
            epc=epc,
            mean_rssi=float(rssis.mean()),
            std_rssi=max(float(rssis.std()), 0.5),
            expected_read_rate=len(g) / dur,
        )
    return out


# --------------------------------------------------------------------------- #
# Heuristic classifier (the real-time detector)
# --------------------------------------------------------------------------- #
def classify(
    feats: WindowFeatures,
    baseline: Optional[Baseline],
    rssi_drop_db: float = RSSI_DROP_DB,
    obst_std_db: float = OBST_STD_DB,
    read_rate_low: float = READ_RATE_LOW,
) -> Verdict:
    """Explainable threshold vote -> :class:`Verdict`."""
    reasons: List[str] = []
    score = 0.0

    # 1. Full blockage: baseline expects reads, we got (almost) none.
    if baseline is not None and baseline.expected_read_rate > 0 and feats.read_rate_ratio <= 0.05:
        reasons.append("no reads where the link normally decodes (path fully blocked)")
        return Verdict(feats.antenna, feats.epc, True, 0.97, reasons, feats)

    # 2. RSSI dropped and got noisy together -> classic body/NLOS signature.
    if feats.rssi_delta <= -rssi_drop_db and feats.rssi_std >= obst_std_db:
        reasons.append(
            f"RSSI down {abs(feats.rssi_delta):.1f} dB with std {feats.rssi_std:.1f} dB"
        )
        score += 0.55

    # 3. Read rate materially below baseline.
    if baseline is not None and feats.read_rate_ratio < read_rate_low:
        reasons.append(f"read rate at {feats.read_rate_ratio*100:.0f}% of baseline")
        score += 0.3

    # 4. Corroborating cues.
    if feats.rssi_kurtosis >= KURTOSIS_HI:
        reasons.append(f"heavy-tailed RSSI (kurtosis {feats.rssi_kurtosis:.1f})")
        score += 0.1
    if feats.phase_jump > 1.2:
        reasons.append(f"unexplained phase jump {feats.phase_jump:.2f} rad")
        score += 0.1
    if feats.rssi_delta <= -rssi_drop_db and feats.rssi_std < obst_std_db:
        # A steady, large drop (e.g. metal) even without extra variance.
        reasons.append(f"steady RSSI drop of {abs(feats.rssi_delta):.1f} dB")
        score += 0.3

    obstructed = score >= 0.5
    if not obstructed and not reasons:
        reasons.append("stable RSSI, normal read rate — clear line of sight")
    return Verdict(feats.antenna, feats.epc, obstructed, min(score, 0.95), reasons, feats)


def detect(
    reads: Sequence[TagRead],
    baselines: Dict[Tuple[int, str], Baseline],
    window_s: Optional[float] = None,
    classifier: Optional["ObstructionClassifier"] = None,
    rssi_drop_db: float = RSSI_DROP_DB,
    obst_std_db: float = OBST_STD_DB,
    read_rate_low: float = READ_RATE_LOW,
) -> List[Verdict]:
    """Run detection over every *baseline* pair (so missing reads are caught)."""
    groups = group_reads(reads)
    verdicts: List[Verdict] = []
    keys = set(baselines.keys()) | set(groups.keys())
    for key in sorted(keys):
        ant, epc = key
        pair_reads = groups.get(key, [])
        base = baselines.get(key)
        feats = extract_features(pair_reads, ant, epc, base, window_s)
        v = classify(feats, base, rssi_drop_db, obst_std_db, read_rate_low)
        if classifier is not None and classifier.trained and base is not None:
            p = classifier.predict_proba(feats.vector())
            # Blend the learned opinion in.
            v.confidence = float(0.5 * v.confidence + 0.5 * p)
            v.obstructed = v.confidence >= 0.5
            v.reasons.append(f"ML classifier p(obstructed)={p:.2f}")
        verdicts.append(v)
    return verdicts


# --------------------------------------------------------------------------- #
# Motion-invariant detection (position residuals + Doppler)
# --------------------------------------------------------------------------- #
# The fixed-baseline detector above assumes the tag stays put: a tag that simply
# moves away also drops RSSI and would look "obstructed".  The detector below
# removes that confound.  Each window it RE-ESTIMATES the tag's position from the
# antennas that agree, predicts what RSSI each antenna *should* see at that
# position (from the path-loss model), and flags an antenna only when its
# measured RSSI is far BELOW that prediction.  Tag motion is absorbed into the
# position estimate (all residuals stay ~0), so only an *unexplained* per-path
# drop counts as an obstruction.  Requires >= 3 antennas with known positions.

# Mean |Doppler| (Hz) above this means the tag is moving (~1 m/s at 915 MHz is
# ~6 Hz; a static tag sits near the reader's Doppler noise floor).
DOPPLER_MOVING_HZ = 3.0
# An antenna reading this many dB below its predicted RSSI is a blocked path.
RESID_DROP_DB = 6.0


@dataclass
class PathObservation:
    """Windowed stats for one (tag, antenna) path."""

    antenna: int
    mean_rssi: float
    rssi_std: float
    doppler_mean: Optional[float]
    n_reads: int


@dataclass
class PathVerdict:
    """Per-antenna obstruction decision for a tag at its estimated position."""

    antenna: int
    obstructed: bool
    measured_rssi: float
    predicted_rssi: float
    residual: float          # measured - predicted (<= 0 means weaker than expected)
    rssi_std: float


@dataclass
class TagReport:
    """Motion-aware obstruction report for one tag across all antennas."""

    epc: str
    position: Optional[Tuple[float, float]]
    position_uncertainty: float
    moving: bool
    paths: List[PathVerdict]
    reasons: List[str]

    @property
    def blocked_antennas(self) -> List[int]:
        return [p.antenna for p in self.paths if p.obstructed]

    @property
    def obstructed(self) -> bool:
        return len(self.blocked_antennas) > 0

    @property
    def status(self) -> str:
        return "OBSTRUCTED" if self.obstructed else "CLEAR"


def _window_observations(epc_reads: Sequence[TagRead]) -> Dict[int, PathObservation]:
    """Collapse a tag's reads into one robust observation per antenna."""
    by_ant: Dict[int, List[TagRead]] = {}
    for r in epc_reads:
        by_ant.setdefault(r.antenna, []).append(r)
    obs: Dict[int, PathObservation] = {}
    for ant, rs in by_ant.items():
        rssis = np.array([r.rssi for r in rs], dtype=float)
        # Median is robust to the impulsive dropouts UHF RSSI is prone to.
        mean_rssi = float(np.median(rssis))
        dopplers = [r.doppler for r in rs if r.doppler is not None]
        obs[ant] = PathObservation(
            antenna=ant,
            mean_rssi=mean_rssi,
            rssi_std=float(np.std(rssis)),
            doppler_mean=float(np.mean(dopplers)) if dopplers else None,
            n_reads=len(rs),
        )
    return obs


def _robust_position(
    obs: Dict[int, PathObservation],
    antennas: Dict[int, AntennaConfig],
    resid_drop_db: float,
    max_iter: int = 4,
):
    """Localize the tag using only the antennas that mutually agree.

    Iteratively drops the antenna whose measured RSSI is furthest *below* what
    its distance implies (a blocked path reads too weak -> its range is too
    long -> it is the outlier), refitting until the kept antennas are
    consistent or only two remain.
    """
    active = set(obs.keys())
    ranges = {a: rssi_to_distance(obs[a].mean_rssi, antennas[a]) for a in active}
    fix = localize({a: ranges[a] for a in active}, antennas)
    for _ in range(max_iter):
        if fix is None or len(active) <= 2:
            break
        residuals = {}
        for a in active:
            d = math.hypot(fix.x - antennas[a].x, fix.y - antennas[a].y)
            residuals[a] = obs[a].mean_rssi - expected_rssi(d, antennas[a])
        worst = min(active, key=lambda a: residuals[a])
        if residuals[worst] > -resid_drop_db:
            break  # everyone consistent
        active.discard(worst)
        fix = localize({a: ranges[a] for a in active}, antennas)
    return fix


def analyze(
    reads: Sequence[TagRead],
    antennas: Dict[int, AntennaConfig],
    resid_drop_db: float = RESID_DROP_DB,
    min_antennas: int = 3,
    doppler_moving_hz: float = DOPPLER_MOVING_HZ,
) -> List[TagReport]:
    """Motion-invariant obstruction detection over a window of reads.

    For each tag seen by ``>= min_antennas`` antennas (with known positions),
    estimate its position from the consistent antennas and flag any antenna
    whose RSSI sits ``resid_drop_db`` below the path-loss prediction at that
    position.  Because the position is re-estimated every call, a tag that
    simply moved keeps all residuals near zero and is reported CLEAR.
    """
    groups = group_reads_by_epc(reads)
    reports: List[TagReport] = []
    for epc, epc_reads in sorted(groups.items()):
        obs = _window_observations(epc_reads)
        present = [a for a in obs if a in antennas]
        if len(present) < min_antennas:
            continue  # cannot separate motion from blockage; see analyze_or_baseline
        fix = _robust_position({a: obs[a] for a in present}, antennas, resid_drop_db)
        # Motion from Doppler magnitude: use mean |Doppler| so a tag changing
        # direction within the window doesn't average itself back to zero.
        dopplers = [abs(r.doppler) for r in epc_reads if r.doppler is not None]
        moving = bool(dopplers) and float(np.mean(dopplers)) > doppler_moving_hz
        paths: List[PathVerdict] = []
        reasons: List[str] = []
        if fix is None:
            continue
        for a in sorted(present):
            d = math.hypot(fix.x - antennas[a].x, fix.y - antennas[a].y)
            predicted = expected_rssi(d, antennas[a])
            resid = obs[a].mean_rssi - predicted
            blocked = resid <= -resid_drop_db
            paths.append(PathVerdict(a, blocked, obs[a].mean_rssi, predicted, resid, obs[a].rssi_std))
            if blocked:
                reasons.append(
                    f"Ant {a}: {obs[a].mean_rssi:.1f} dBm is {abs(resid):.1f} dB below the "
                    f"{predicted:.1f} dBm expected at the tag's position — path blocked"
                )
        if not reasons:
            move_txt = "moving" if moving else "static"
            reasons.append(
                f"all antennas match the tag's estimated position ({move_txt}); "
                f"any RSSI change is explained by geometry — clear line of sight"
            )
        reports.append(TagReport(
            epc=epc,
            position=(fix.x, fix.y),
            position_uncertainty=fix.radius_m,
            moving=moving,
            paths=paths,
            reasons=reasons,
        ))
    return reports


def group_reads_by_epc(reads: Sequence[TagRead]) -> Dict[str, List[TagRead]]:
    out: Dict[str, List[TagRead]] = {}
    for r in reads:
        out.setdefault(r.epc, []).append(r)
    return out


# --------------------------------------------------------------------------- #
# Optional learned classifier (numpy logistic regression)
# --------------------------------------------------------------------------- #
@dataclass
class ObstructionClassifier:
    """Tiny logistic-regression obstruction classifier (numpy only).

    Trained on feature vectors from simulator-labelled clear/obstructed windows
    (see :func:`make_training_set`).  Standardises inputs, then fits weights by
    gradient descent.  Purely a learned *second opinion* — the heuristic in
    :func:`classify` is the primary detector.
    """

    lr: float = 0.1
    epochs: int = 400
    trained: bool = field(default=False, init=False)

    def _standardise(self, X: np.ndarray) -> np.ndarray:
        return (X - self._mu) / self._sd

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ObstructionClassifier":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self._mu = X.mean(axis=0)
        self._sd = X.std(axis=0) + 1e-9
        Xs = self._standardise(X)
        n, d = Xs.shape
        self._w = np.zeros(d)
        self._b = 0.0
        for _ in range(self.epochs):
            z = Xs @ self._w + self._b
            p = 1.0 / (1.0 + np.exp(-z))
            grad_w = Xs.T @ (p - y) / n
            grad_b = float(np.mean(p - y))
            self._w -= self.lr * grad_w
            self._b -= self.lr * grad_b
        self.trained = True
        return self

    def predict_proba(self, x: np.ndarray) -> float:
        if not self.trained:
            return 0.0
        xs = (np.asarray(x, dtype=float) - self._mu) / self._sd
        z = float(xs @ self._w + self._b)
        return 1.0 / (1.0 + math.exp(-z))


def make_training_set(n_windows: int = 240, seed: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Generate labelled feature vectors by toggling an obstruction in the sim."""
    from .simulator import Obstruction, Scene, SimTag
    from .models import AntennaConfig

    rng = np.random.default_rng(seed)
    X: List[np.ndarray] = []
    y: List[int] = []
    for i in range(n_windows):
        obstruct = i % 2 == 0
        ant = AntennaConfig(antenna_id=1, x=0.0, y=0.0)
        tx = float(rng.uniform(1.5, 4.0))
        ty = float(rng.uniform(-1.0, 1.0))
        tag = SimTag("E280-TRAIN", x=tx, y=ty)
        obs = []
        if obstruct:
            # Put a blocker on the segment between antenna (0,0) and the tag.
            f = float(rng.uniform(0.3, 0.7))
            obs = [Obstruction(x=tx * f, y=ty * f, radius=0.4,
                               attenuation_db=float(rng.uniform(4.0, 12.0)))]
        scene = Scene(antennas={1: ant}, tags=[tag], obstructions=obs,
                      bounds=(-2, -3, 6, 3), seed=int(rng.integers(0, 1e6)))
        # First a clear baseline (no obstruction), then the labelled window.
        clear = Scene(antennas={1: ant}, tags=[tag], obstructions=[],
                      bounds=(-2, -3, 6, 3), seed=int(rng.integers(0, 1e6)))
        base_reads: List[TagRead] = []
        for _ in range(25):
            base_reads.extend(clear.read_once())
        baselines = build_baselines(base_reads, duration_s=25 / 6.0)
        win_reads: List[TagRead] = []
        for _ in range(20):
            win_reads.extend(scene.read_once())
        base = baselines.get((1, "E280-TRAIN"))
        feats = extract_features(win_reads, 1, "E280-TRAIN", base, duration_s=20 / 6.0)
        X.append(feats.vector())
        y.append(1 if obstruct else 0)
    return np.array(X), np.array(y)
