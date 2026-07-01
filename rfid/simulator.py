"""Physically-motivated tag-read simulator.

When no Zebra reader is on the network (e.g. running this in the cloud, or at a
desk), the whole tool suite still needs live-looking data.  :class:`Scene`
models antennas, moving tags and blocking objects on a 2D field and emits
:class:`~rfid.models.TagRead` objects that obey the same log-distance path-loss
physics the ranging tool inverts — so distance/localization, denoising and
obstruction detection all behave the way they would on real hardware.

The simulation is intentionally deterministic given a seed so demos are
repeatable, but adds realistic per-read RSSI noise, phase, multipath jitter and
signal loss from obstructions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .models import AntennaConfig, TagRead
from .ranging import DEFAULT_FREQ_HZ, expected_rssi, wavelength

# A read below this RSSI is unlikely to decode (reader sensitivity floor).
SENSITIVITY_DBM = -78.0


@dataclass
class SimTag:
    """A tag moving on the field."""

    epc: str
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    # Small per-tag antenna-gain offset (orientation/polarisation), dB.
    gain_offset_db: float = 0.0

    def advance(self, dt: float, bounds: Tuple[float, float, float, float]) -> None:
        self.x += self.vx * dt
        self.y += self.vy * dt
        x0, y0, x1, y1 = bounds
        # Bounce off the field edges so tags stay in view.
        if self.x < x0 or self.x > x1:
            self.vx = -self.vx
            self.x = min(max(self.x, x0), x1)
        if self.y < y0 or self.y > y1:
            self.vy = -self.vy
            self.y = min(max(self.y, y0), y1)


@dataclass
class Obstruction:
    """A circular blocker that attenuates any antenna<->tag path crossing it.

    ``attenuation_db`` is the extra path loss when the line of sight is fully
    blocked (a human body is ~3-8 dB at UHF, metal/water far more).
    """

    x: float
    y: float
    radius: float = 0.35
    attenuation_db: float = 6.0
    label: str = "person"

    def blockage(self, ax: float, ay: float, tx: float, ty: float) -> float:
        """Fraction (0..1) of this blocker that intersects the A->T segment."""
        # Distance from circle centre to the segment.
        dx, dy = tx - ax, ty - ay
        seg_len2 = dx * dx + dy * dy
        if seg_len2 < 1e-9:
            return 0.0
        t = ((self.x - ax) * dx + (self.y - ay) * dy) / seg_len2
        if t < 0.0 or t > 1.0:
            return 0.0  # closest point is outside the segment -> not between them
        px, py = ax + t * dx, ay + t * dy
        dist = math.hypot(self.x - px, self.y - py)
        if dist >= self.radius:
            return 0.0
        # Linear ramp: fully blocked at centre, none at the edge.
        return 1.0 - dist / self.radius


@dataclass
class Scene:
    """A field of antennas, tags and obstructions that produces reads."""

    antennas: Dict[int, AntennaConfig]
    tags: List[SimTag] = field(default_factory=list)
    obstructions: List[Obstruction] = field(default_factory=list)
    bounds: Tuple[float, float, float, float] = (0.0, 0.0, 6.0, 4.0)
    noise_sigma_db: float = 2.5
    freq_hz: float = DEFAULT_FREQ_HZ
    seed: int = 7
    _t: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    # -- dynamics ---------------------------------------------------------- #
    def step(self, dt: float = 0.2) -> None:
        self._t += dt
        for tag in self.tags:
            tag.advance(dt, self.bounds)

    # -- measurement model ------------------------------------------------- #
    def _path_attenuation(self, cfg: AntennaConfig, tag: SimTag) -> Tuple[float, float]:
        """Total obstruction loss (dB) and blockage fraction for one path."""
        total_db = 0.0
        max_frac = 0.0
        for ob in self.obstructions:
            frac = ob.blockage(cfg.x, cfg.y, tag.x, tag.y)
            if frac > 0:
                total_db += frac * ob.attenuation_db
                max_frac = max(max_frac, frac)
        return total_db, max_frac

    def read_once(self, now: Optional[float] = None) -> List[TagRead]:
        """Generate one report window of reads across all antennas/tags."""
        now = self._t if now is None else now
        lam = wavelength(self.freq_hz)
        reads: List[TagRead] = []
        for cfg in self.antennas.values():
            for tag in self.tags:
                d = math.hypot(tag.x - cfg.x, tag.y - cfg.y)
                d = max(d, 0.05)
                base = expected_rssi(d, cfg) + tag.gain_offset_db
                obst_db, frac = self._path_attenuation(cfg, tag)
                # Obstruction lowers RSSI *and* adds jitter (scattering).
                extra_sigma = self.noise_sigma_db + 3.0 * frac
                rssi = base - obst_db + self._rng.normal(0, extra_sigma)

                # Read probability drops near the sensitivity floor and when
                # the path is blocked.
                margin = rssi - SENSITIVITY_DBM
                p_read = 1.0 / (1.0 + math.exp(-(margin) / 3.0))
                p_read *= (1.0 - 0.6 * frac)
                if self._rng.random() > p_read:
                    continue  # missed read this window

                # Phase: round-trip, wrapped to [0, 2pi), with small noise.
                phase = (4.0 * math.pi * d / lam) % (2.0 * math.pi)
                phase = (phase + self._rng.normal(0, 0.15 + 0.4 * frac)) % (2.0 * math.pi)
                seen = int(np.clip(self._rng.poisson(max(p_read * 8, 0.5)), 1, 50))

                # Doppler = 2*v_radial/lambda (round trip). v_radial is the tag's
                # velocity projected onto the antenna->tag line; ~0 for a static
                # tag, non-zero (signed) for one moving toward/away.  This is the
                # signal that lets the detector tell motion from a static block.
                ux, uy = (tag.x - cfg.x) / d, (tag.y - cfg.y) / d
                v_radial = tag.vx * ux + tag.vy * uy
                doppler = 2.0 * v_radial / lam + self._rng.normal(0, 2.0)

                reads.append(
                    TagRead(
                        epc=tag.epc,
                        rssi=round(float(rssi), 1),
                        antenna=cfg.antenna_id,
                        phase=round(float(phase), 3),
                        channel=int(self._rng.integers(0, 50)),
                        timestamp=now,
                        seen_count=seen,
                        doppler=round(float(doppler), 1),
                        reader="SIMULATOR",
                    )
                )
        return reads


# --------------------------------------------------------------------------- #
# Ready-made demo scene
# --------------------------------------------------------------------------- #
def room_scene(seed: int = 9) -> Scene:
    """Four antennas on the corners of a room + one tag in the middle.

    The default tag is static; the Obstruction tool lets you add velocity and
    drop a blocker on any antenna's path.  Four antennas mean one path can be
    blocked and the other three still fix the tag's position.
    """
    antennas = {
        1: AntennaConfig(antenna_id=1, x=0.0, y=0.0, heading_deg=45.0),
        2: AntennaConfig(antenna_id=2, x=6.0, y=0.0, heading_deg=135.0),
        3: AntennaConfig(antenna_id=3, x=6.0, y=5.0, heading_deg=225.0),
        4: AntennaConfig(antenna_id=4, x=0.0, y=5.0, heading_deg=315.0),
    }
    tags = [SimTag("E280-1170-0042", x=3.0, y=2.5, vx=0.0, vy=0.0)]
    return Scene(antennas=antennas, tags=tags, bounds=(0.0, 0.0, 6.0, 5.0), seed=seed)
