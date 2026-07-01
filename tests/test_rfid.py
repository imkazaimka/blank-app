"""Unit tests for the Zebra RFID toolkit core library (no hardware needed)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from rfid.models import AntennaConfig, Device, DeviceKind, TagRead
from rfid import ranging, denoise, obstruction
from rfid.simulator import Obstruction, Scene, SimTag, demo_scene, spread_scene
from rfid.sources import LLRPTagSource, SimulatedTagSource
from rfid import discovery


# --------------------------------------------------------------------------- #
# Ranging
# --------------------------------------------------------------------------- #
def test_rssi_distance_roundtrip():
    cfg = AntennaConfig(1, rssi_ref=-45, d_ref=1.0, path_loss_n=2.5)
    for d in [0.5, 1.0, 2.0, 5.0, 8.0]:
        rssi = ranging.expected_rssi(d, cfg)
        back = ranging.rssi_to_distance(rssi, cfg)
        assert math.isclose(back, d, rel_tol=1e-6)


def test_calibration_recovers_params():
    cfg = AntennaConfig(1, rssi_ref=-42, d_ref=1.0, path_loss_n=2.8)
    ds = [1, 2, 3, 4, 5, 6]
    rssis = [ranging.expected_rssi(d, cfg) for d in ds]
    ref, n, rms = ranging.calibrate_path_loss(ds, rssis)
    assert math.isclose(ref, -42, abs_tol=0.2)
    assert math.isclose(n, 2.8, abs_tol=0.05)
    assert rms < 1e-6


def test_localize_recovers_known_position():
    ants = {1: AntennaConfig(1, x=0, y=0), 2: AntennaConfig(2, x=6, y=0),
            3: AntennaConfig(3, x=3, y=5)}
    tx, ty = 2.3, 3.1
    ranges = {a: math.hypot(tx - c.x, ty - c.y) for a, c in ants.items()}
    fix = ranging.localize(ranges, ants)
    assert fix is not None
    assert abs(fix.x - tx) < 0.05 and abs(fix.y - ty) < 0.05  # no sign bug
    assert fix.method == "trilateration"


def test_localize_single_antenna_uses_heading():
    ants = {1: AntennaConfig(1, x=0, y=0, heading_deg=0.0)}
    fix = ranging.localize({1: 3.0}, ants)
    assert fix.method == "arc"
    assert abs(fix.x - 3.0) < 1e-6 and abs(fix.y) < 1e-6


def test_aoa_overhead_projection():
    # Straight down (elevation 0) lands under the reader.
    x, y = ranging.aoa_to_xy(0, 0, (2, 2), mount_height_m=4)
    assert abs(x - 2) < 1e-9 and abs(y - 2) < 1e-9
    # 45 deg elevation from 4 m -> radius 4 m.
    x, y = ranging.aoa_to_xy(0, 45, (0, 0), mount_height_m=4)
    assert math.isclose(math.hypot(x, y), 4.0, rel_tol=1e-6)


def test_phase_velocity_sign():
    # Moving away -> positive radial velocity.
    lam = ranging.wavelength()
    times = np.linspace(0, 1, 20)
    # d increases linearly; phase = 4*pi*d/lam
    d = 1.0 + 0.5 * times
    phase = (4 * math.pi * d / lam) % (2 * math.pi)
    v = ranging.phase_velocity(phase, times)
    assert v is not None and v > 0


# --------------------------------------------------------------------------- #
# Denoising
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", list(denoise.FILTERS.keys()))
def test_every_filter_reduces_noise(name):
    rng = np.random.default_rng(0)
    true = -55 + 3 * np.sin(np.linspace(0, 6, 150))
    noisy = true + rng.normal(0, 4, 150)
    out = denoise.make_filter(name).smooth(noisy)
    assert len(out) == len(noisy)
    # Denoised signal should track truth better than raw noise.
    assert np.std(out - true) < np.std(noisy - true)


def test_kalman_reduces_jitter():
    rng = np.random.default_rng(1)
    noisy = -50 + rng.normal(0, 5, 200)
    out = denoise.KalmanRSSI().smooth(noisy)
    assert np.std(np.diff(out)) < np.std(np.diff(noisy))


def test_denoise_phase_unwraps_and_rewraps():
    ph = np.mod(np.linspace(0, 30, 100), 2 * np.pi)
    out = denoise.denoise_phase(ph)
    assert out.min() >= 0 and out.max() <= 2 * np.pi + 1e-6
    assert len(out) == len(ph)


def test_learned_denoiser_trains_online():
    rng = np.random.default_rng(2)
    noisy = -60 + rng.normal(0, 4, 120)
    d = denoise.LearnedDenoiser(seed=0)
    out = d.smooth(noisy)
    assert np.isfinite(out).all()


# --------------------------------------------------------------------------- #
# Simulator
# --------------------------------------------------------------------------- #
def test_simulator_produces_reads():
    scene = demo_scene()
    total = 0
    for _ in range(10):
        scene.step(0.2)
        total += len(scene.read_once())
    assert total > 0


def test_obstruction_lowers_rssi():
    ant = AntennaConfig(1, x=0, y=0)
    tag = SimTag("E", x=3, y=0)
    clear = Scene({1: ant}, [tag], [], bounds=(-2, -3, 6, 3), seed=1)
    blocked = Scene({1: ant}, [tag],
                    [Obstruction(x=1.5, y=0, radius=0.5, attenuation_db=10)],
                    bounds=(-2, -3, 6, 3), seed=1)
    cr = np.mean([r.rssi for _ in range(40) for r in clear.read_once()])
    br = np.mean([r.rssi for _ in range(40) for r in blocked.read_once()])
    assert br < cr - 3  # blocked path is materially weaker


def test_blockage_geometry():
    ob = Obstruction(x=1, y=0, radius=0.5)
    # On the segment (0,0)-(2,0): centre blocked.
    assert ob.blockage(0, 0, 2, 0) > 0.9
    # Off to the side: not blocked.
    assert ob.blockage(0, 3, 2, 3) == 0.0
    # Behind the antenna (segment not reaching it): not blocked.
    assert ob.blockage(2, 0, 4, 0) == 0.0


# --------------------------------------------------------------------------- #
# Obstruction detection
# --------------------------------------------------------------------------- #
def _capture(scene, n):
    reads = []
    for _ in range(n):
        reads.extend(scene.read_once())
    return reads


def test_detector_flags_clear_vs_obstructed():
    ant = AntennaConfig(1, x=0, y=0)
    tag = SimTag("E280-AAA", x=3, y=0)
    clear = Scene({1: ant}, [tag], [], bounds=(-2, -3, 6, 3), seed=1)
    base = obstruction.build_baselines(_capture(clear, 30), duration_s=5.0)

    clear_v = obstruction.detect(_capture(clear, 20), base, window_s=3.3)
    assert clear_v and not clear_v[0].obstructed

    blocked = Scene({1: ant}, [tag],
                    [Obstruction(x=1.5, y=0, radius=0.5, attenuation_db=9)],
                    bounds=(-2, -3, 6, 3), seed=2)
    blk_v = obstruction.detect(_capture(blocked, 20), base, window_s=3.3)
    assert blk_v and blk_v[0].obstructed
    assert blk_v[0].confidence > 0.4


def test_missing_reads_flagged_obstructed():
    # Baseline expects reads, but the window has none -> full blockage.
    base = {(1, "E"): obstruction.Baseline(1, "E", -55, 2.0, 6.0)}
    verdicts = obstruction.detect([], base, window_s=3.0)
    assert verdicts[0].obstructed and verdicts[0].confidence > 0.9


def test_ml_classifier_learns():
    X, y = obstruction.make_training_set(120, seed=3)
    clf = obstruction.ObstructionClassifier(epochs=300).fit(X, y)
    preds = np.array([clf.predict_proba(x) for x in X]) > 0.5
    assert (preds == y).mean() > 0.85


# --------------------------------------------------------------------------- #
# Discovery + sources (no network required)
# --------------------------------------------------------------------------- #
def test_demo_devices_have_connect_hints():
    devs = discovery.demo_devices()
    assert len(devs) >= 3
    readers = [d for d in devs if d.kind is DeviceKind.READER]
    printers = [d for d in devs if d.kind is DeviceKind.PRINTER]
    assert readers and printers
    for d in devs:
        assert d.connect_hint()
    assert "5084" in readers[0].connect_hint()
    assert "9100" in printers[0].connect_hint()


def test_classify_hostname():
    assert discovery._classify_hostname("FX9600CD3B0D")[0] is DeviceKind.READER
    assert discovery._classify_hostname("ZT411R7A2C10")[0] is DeviceKind.PRINTER
    assert discovery._classify_hostname("randompc")[0] is None


def test_udp_response_parser():
    payload = bytes((0x3A, 0x2C, 0x2E, 0x03)) + b"ZT411R\x00ZebraNet\x00V93.21\x00"
    dev = discovery._parse_udp_response(payload, "10.0.0.9")
    assert dev is not None and dev.kind is DeviceKind.PRINTER
    assert dev.model.startswith("ZT411")


def test_llrp_tag_normalisation():
    tag = {"EPC": bytes.fromhex("E28011700001"), "PeakRSSI": -57,
           "AntennaID": 2, "ImpinjRFPhaseAngle": 2048, "TagSeenCount": 4,
           "LastSeenTimestampUTC": 1_600_000_000_000_000}
    r = LLRPTagSource._to_read(tag, "192.168.1.50")
    assert r.epc == "E28011700001"
    assert r.rssi == -57 and r.antenna == 2 and r.seen_count == 4
    assert math.isclose(r.phase, math.pi, rel_tol=1e-3)  # 2048/4096 * 2pi


def test_llrp_impinj_fine_rssi():
    tag = {"EPC": b"\x01\x02", "ImpinjPeakRSSI": -5012, "AntennaID": 1}
    r = LLRPTagSource._to_read(tag, "h")
    assert math.isclose(r.rssi, -50.12, rel_tol=1e-6)


def test_simulated_source_streams():
    src = SimulatedTagSource(demo_scene(), rate_hz=20)
    src.start()
    import time
    time.sleep(0.4)
    reads = src.drain()
    src.stop()
    assert len(reads) > 0
    assert all(isinstance(r, TagRead) for r in reads)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
