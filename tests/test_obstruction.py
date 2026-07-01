"""Tests for the Zebra RFID obstruction detector (no hardware needed).

The headline tests prove the motion-invariant detector tells a *moving tag*
apart from a *blocked path* — the whole point of the tool.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

from rfid.models import AntennaConfig, TagRead
from rfid import obstruction, ranging
from rfid.simulator import Obstruction, Scene, SimTag, room_scene
from rfid.sources import LLRPTagSource, MQTTTagSource, SimulatedTagSource


# --- a 4-antenna room so one path can be blocked and 3 still fix position --- #
def _room():
    return {
        1: AntennaConfig(1, x=0, y=0), 2: AntennaConfig(2, x=6, y=0),
        3: AntennaConfig(3, x=6, y=5), 4: AntennaConfig(4, x=0, y=5),
    }


def _capture(scene, steps):
    reads = []
    for _ in range(steps):
        scene.step(0.2)
        reads.extend(scene.read_once())
    return reads


# --------------------------------------------------------------------------- #
# Motion-invariant detection — the core of the tool
# --------------------------------------------------------------------------- #
def test_moving_tag_is_not_flagged_as_obstruction():
    """A fast-moving, UNOBSTRUCTED tag must stay CLEAR across many windows."""
    ants = _room()
    tag = SimTag("E-MOVE", x=1.0, y=1.0, vx=0.6, vy=0.5)
    scene = Scene(ants, [tag], [], bounds=(0, 0, 6, 5), seed=1)
    false_positives = 0
    windows = 0
    for _ in range(8):
        reports = obstruction.analyze(_capture(scene, 6), ants)
        for r in reports:
            windows += 1
            false_positives += len(r.blocked_antennas)
    assert windows > 0
    assert false_positives == 0, "movement was mistaken for an obstruction"


def test_static_tag_blocked_path_is_flagged():
    ants = _room()
    tag = SimTag("E-BLK", x=3.0, y=2.5)
    scene = Scene(ants, [tag],
                  [Obstruction(x=1.3, y=1.1, radius=0.5, attenuation_db=12)],
                  bounds=(0, 0, 6, 5), seed=2)
    reports = obstruction.analyze(_capture(scene, 12), ants)
    assert reports and reports[0].obstructed
    assert 1 in reports[0].blocked_antennas          # Ant 1's path is the blocked one


def test_moving_and_blocked_is_still_detected():
    """The hard case: a tag that is BOTH moving and has one path blocked."""
    ants = _room()
    tag = SimTag("E-BOTH", x=3.0, y=2.5, vx=0.3, vy=0.0)
    scene = Scene(ants, [tag],
                  [Obstruction(x=1.5, y=1.25, radius=0.6, attenuation_db=13)],
                  bounds=(0, 0, 6, 5), seed=3)
    saw_block = False
    for _ in range(4):
        for r in obstruction.analyze(_capture(scene, 6), ants):
            if 1 in r.blocked_antennas:
                saw_block = True
    assert saw_block, "failed to flag a blocked path while the tag was moving"


def test_needs_three_antennas():
    ants = {1: AntennaConfig(1, x=0, y=0), 2: AntennaConfig(2, x=6, y=0)}
    tag = SimTag("E", x=3, y=2)
    scene = Scene(ants, [tag], [], bounds=(0, 0, 6, 5), seed=4)
    # Fewer than 3 antennas -> cannot separate motion from blockage -> no report.
    assert obstruction.analyze(_capture(scene, 10), ants) == []


def test_position_estimate_is_reasonable():
    ants = _room()
    tag = SimTag("E", x=2.5, y=3.0)
    scene = Scene(ants, [tag], [], bounds=(0, 0, 6, 5), seed=5)
    rep = obstruction.analyze(_capture(scene, 15), ants)[0]
    assert rep.position is not None
    # Localization is coarse from RSSI, but should be in the right neighbourhood.
    assert math.hypot(rep.position[0] - 2.5, rep.position[1] - 3.0) < 2.0


# --------------------------------------------------------------------------- #
# Simulator Doppler (the motion signal)
# --------------------------------------------------------------------------- #
def test_simulator_doppler_tracks_motion():
    ants = {1: AntennaConfig(1, x=0, y=0)}
    mover = Scene(ants, [SimTag("M", x=2, y=0, vx=0.8, vy=0.0)], [], seed=1)
    still = Scene(ants, [SimTag("S", x=2, y=0)], [], seed=1)
    md = np.mean([abs(r.doppler) for r in _capture(mover, 20) if r.doppler is not None])
    sd = np.mean([abs(r.doppler) for r in _capture(still, 20) if r.doppler is not None])
    assert md > sd + 2.0     # a moving tag carries clearly more Doppler


# --------------------------------------------------------------------------- #
# Fixed-zone baseline detector (retained fallback) + ML classifier
# --------------------------------------------------------------------------- #
def test_baseline_detector_clear_vs_blocked():
    ant = AntennaConfig(1, x=0, y=0)
    tag = SimTag("E280-AAA", x=3, y=0)
    clear = Scene({1: ant}, [tag], [], bounds=(-2, -3, 6, 3), seed=1)
    base = obstruction.build_baselines(_capture(clear, 30), duration_s=5.0)
    assert not obstruction.detect(_capture(clear, 20), base, window_s=3.3)[0].obstructed
    blocked = Scene({1: ant}, [tag],
                    [Obstruction(x=1.5, y=0, radius=0.5, attenuation_db=9)],
                    bounds=(-2, -3, 6, 3), seed=2)
    assert obstruction.detect(_capture(blocked, 20), base, window_s=3.3)[0].obstructed


def test_baseline_missing_reads_flagged():
    base = {(1, "E"): obstruction.Baseline(1, "E", -55, 2.0, 6.0)}
    v = obstruction.detect([], base, window_s=3.0)[0]
    assert v.obstructed and v.confidence > 0.9


def test_ml_classifier_learns():
    X, y = obstruction.make_training_set(120, seed=3)
    clf = obstruction.ObstructionClassifier(epochs=300).fit(X, y)
    acc = (np.array([clf.predict_proba(x) for x in X]) > 0.5).astype(int)
    assert (acc == y).mean() > 0.85


# --------------------------------------------------------------------------- #
# Ranging bits the detector depends on
# --------------------------------------------------------------------------- #
def test_rssi_distance_roundtrip():
    cfg = AntennaConfig(1, rssi_ref=-45, path_loss_n=2.5)
    for d in [0.5, 1, 2, 5]:
        assert math.isclose(ranging.rssi_to_distance(ranging.expected_rssi(d, cfg), cfg),
                            d, rel_tol=1e-6)


def test_localize_recovers_known_point():
    ants = _room()
    tx, ty = 2.3, 3.1
    ranges = {a: math.hypot(tx - c.x, ty - c.y) for a, c in ants.items()}
    fix = ranging.localize(ranges, ants)
    assert abs(fix.x - tx) < 0.05 and abs(fix.y - ty) < 0.05


# --------------------------------------------------------------------------- #
# Sources
# --------------------------------------------------------------------------- #
def test_simulated_source_streams():
    src = SimulatedTagSource(room_scene(), rate_hz=20)
    src.start()
    import time
    time.sleep(0.4)
    reads = src.drain()
    src.stop()
    assert reads and all(isinstance(r, TagRead) for r in reads)


def test_mqtt_parses_wrapped_simpletagevent():
    import json
    payload = json.dumps({
        "data": {"format": "epc", "idHex": "E28011700001", "peakRssi": -56,
                 "antenna": 2, "channel": 4, "phase": 180},
        "timestamp": 1600000000000, "type": "SimpleTagEvent"})
    r = MQTTTagSource._parse_payload(payload, reader="myfxIN")[0]
    assert r.epc == "E28011700001" and r.rssi == -56.0 and r.antenna == 2
    assert math.isclose(r.phase, math.pi, rel_tol=1e-3)     # 180 deg -> pi rad
    assert abs(r.timestamp - 1600000000.0) < 1              # epoch ms -> s


def test_mqtt_parses_flat_and_batched():
    import json
    flat = json.dumps({"epc": "AABB", "rssi": -61, "antennaPort": 3,
                       "seenCount": 7, "doppler": 12.0, "timestamp": 1600000000})
    r = MQTTTagSource._parse_payload(flat)[0]
    assert r.antenna == 3 and r.seen_count == 7 and r.doppler == 12.0
    batch = json.dumps([{"idHex": "AAAA", "peakRssi": -50, "antenna": 1},
                        {"idHex": "BBBB", "peakRssi": -70, "antenna": 4}])
    assert len(MQTTTagSource._parse_payload(batch)) == 2


def test_mqtt_ignores_non_tag_messages():
    import json
    mgmt = json.dumps({"type": "ManagementEvent", "data": {"status": "ok"}})
    assert MQTTTagSource._parse_payload(mgmt) == []
    assert MQTTTagSource._parse_payload(b"not json") == []


def test_llrp_tag_normalisation():
    tag = {"EPC": bytes.fromhex("E28011700001"), "PeakRSSI": -57, "AntennaID": 2,
           "ImpinjRFPhaseAngle": 2048, "LastSeenTimestampUTC": 1_600_000_000_000_000}
    r = LLRPTagSource._to_read(tag, "h")
    assert r.epc == "E28011700001" and r.rssi == -57 and r.antenna == 2
    assert math.isclose(r.phase, math.pi, rel_tol=1e-3)


# --------------------------------------------------------------------------- #
# Streamlit app smoke test
# --------------------------------------------------------------------------- #
def test_antenna_count_is_auto_detected_from_reads():
    """The antenna count comes from the stream, not a user-entered guess."""
    from app_common import distinct_antenna_ids
    ants = _room()
    scene = Scene(ants, [SimTag("E", x=3, y=2.5)], [], bounds=(0, 0, 6, 5), seed=1)
    hist = {}
    for r in _capture(scene, 10):
        hist.setdefault(r.epc, []).append(r)
    assert distinct_antenna_ids(hist) == [1, 2, 3, 4]     # all four detected
    # A reader with only two antennas wired -> only two detected, no guess of 4.
    two = {1: AntennaConfig(1, x=0, y=0), 2: AntennaConfig(2, x=6, y=0)}
    scene2 = Scene(two, [SimTag("E", x=3, y=1)], [], bounds=(0, 0, 6, 5), seed=1)
    h2 = {}
    for r in _capture(scene2, 10):
        h2.setdefault(r.epc, []).append(r)
    assert distinct_antenna_ids(h2) == [1, 2]
    assert distinct_antenna_ids({}) == []


def test_app_runs_headless():
    from streamlit.testing.v1 import AppTest
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    scene = room_scene()
    hist = {}
    for _ in range(20):
        scene.step(0.2)
        for r in scene.read_once():
            hist.setdefault(r.epc, []).append(r)
    at = AppTest.from_file(os.path.join(root, "app.py"), default_timeout=30)
    at.session_state["scene"] = scene
    at.session_state["history"] = hist
    at.run()
    assert not at.exception


def test_app_live_mqtt_mode_renders():
    """The Live-reader / MQTT sidebar branch must render without error."""
    from streamlit.testing.v1 import AppTest
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    at = AppTest.from_file(os.path.join(root, "app.py"), default_timeout=30)
    at.session_state["mode"] = "Live reader (LLRP)"
    at.run()
    assert not at.exception


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
