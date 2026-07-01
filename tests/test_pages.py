"""Smoke tests that execute each Streamlit page headlessly via AppTest.

They pre-populate session state with simulated reads so the pages run their
full rendering path (plots, tables, detectors) and assert no exception is
raised.  No hardware or browser required.
"""

from __future__ import annotations

import os

import pytest
from streamlit.testing.v1 import AppTest

from rfid.discovery import demo_devices
from rfid.obstruction import build_baselines
from rfid.simulator import spread_scene

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _history(scene, steps=25):
    """Run a scene and group reads into the per-EPC history dict pages expect."""
    hist: dict = {}
    for _ in range(steps):
        scene.step(0.2)
        for r in scene.read_once():
            hist.setdefault(r.epc, []).append(r)
    return hist


def _run(page_relpath, session=None, timeout=30):
    at = AppTest.from_file(os.path.join(ROOT, page_relpath), default_timeout=timeout)
    for k, v in (session or {}).items():
        at.session_state[k] = v
    at.run()
    return at


def test_home_runs():
    at = _run("app.py")
    assert not at.exception


def test_select_device_page_runs_with_demo_devices():
    at = _run("pages/1_Select_Device.py", {"devices": demo_devices()})
    assert not at.exception


def test_select_device_page_empty():
    at = _run("pages/1_Select_Device.py")
    assert not at.exception


def test_antenna_location_page_runs():
    scene = spread_scene()
    hist = _history(scene)
    at = _run("pages/2_Antenna_and_Location.py",
              {"scene": scene, "history": hist})
    assert not at.exception


def test_denoising_page_runs():
    scene = spread_scene()
    hist = _history(scene)
    at = _run("pages/3_Signal_Denoising.py",
              {"scene": scene, "history": hist})
    assert not at.exception


def test_obstruction_page_runs_with_baseline():
    scene = spread_scene()
    hist = _history(scene, steps=30)
    flat = [r for buf in hist.values() for r in buf]
    baselines = build_baselines(flat)
    at = _run("pages/4_Obstruction_Check.py",
              {"scene": scene, "history": hist, "baselines": baselines})
    assert not at.exception


def test_obstruction_page_runs_without_baseline():
    scene = spread_scene()
    hist = _history(scene)
    at = _run("pages/4_Obstruction_Check.py", {"scene": scene, "history": hist})
    assert not at.exception


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
