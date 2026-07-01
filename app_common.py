"""Shared Streamlit helpers for the Obstruction Detector app.

Session-state wiring, the data-source lifecycle (simulator or a live LLRP
reader), a rolling read buffer and a live-refresh loop.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence

import streamlit as st

from rfid.models import AntennaConfig, TagRead
from rfid.simulator import Scene, room_scene
from rfid.sources import LLRPTagSource, MQTTTagSource, SimulatedTagSource, TagSource

BRAND = "#00B5E2"


def sllurp_available() -> bool:
    try:
        __import__("sllurp")
        return True
    except Exception:
        return False


def paho_available() -> bool:
    try:
        __import__("paho.mqtt.client")
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
def _ss():
    ss = st.session_state
    ss.setdefault("source", None)
    ss.setdefault("source_kind", None)     # "sim" | "live"
    ss.setdefault("scene", None)
    ss.setdefault("history", {})           # epc -> list[TagRead]
    ss.setdefault("live_antennas", {})     # antenna_id -> AntennaConfig (live mode)
    return ss


def get_scene() -> Scene:
    ss = _ss()
    if ss.scene is None:
        ss.scene = room_scene()
    return ss.scene


def set_scene(scene: Scene) -> None:
    stop_source()
    _ss().scene = scene
    clear_history()


def antennas() -> Dict[int, AntennaConfig]:
    """Antenna geometry currently in use (scene in sim mode, editor in live)."""
    ss = _ss()
    if ss.source_kind == "live" and ss.live_antennas:
        return ss.live_antennas
    return get_scene().antennas


# --------------------------------------------------------------------------- #
# Source lifecycle
# --------------------------------------------------------------------------- #
def start_sim(rate_hz: float = 6.0) -> TagSource:
    ss = _ss()
    stop_source()
    src = SimulatedTagSource(get_scene(), rate_hz=rate_hz)
    src.start()
    ss.source, ss.source_kind = src, "sim"
    return src


def start_live(ip: str, port: int = 5084, tx_power: Optional[int] = None) -> TagSource:
    ss = _ss()
    stop_source()
    src = LLRPTagSource(ip, port=port, tx_power=tx_power)
    src.start()
    ss.source, ss.source_kind = src, "live"
    return src


def start_mqtt(broker: str, port: int = 1883, topic: str = "zebra/+/data",
               username: str = "", password: str = "", tls: bool = False) -> TagSource:
    """Connect to a Zebra IoT Connector tag-data topic on an MQTT broker."""
    ss = _ss()
    stop_source()
    src = MQTTTagSource(broker, port=port, topic=topic, username=username,
                        password=password, tls=tls)
    src.start()
    ss.source, ss.source_kind = src, "live"
    return src


def stop_source() -> None:
    ss = _ss()
    if ss.source is not None:
        try:
            ss.source.stop()
        except Exception:
            pass
    ss.source, ss.source_kind = None, None


def source_running() -> bool:
    ss = _ss()
    return ss.source is not None and ss.source.running


# --------------------------------------------------------------------------- #
# Rolling read history
# --------------------------------------------------------------------------- #
def pump(maxlen: int = 600) -> List[TagRead]:
    ss = _ss()
    if ss.source is None:
        return []
    new = ss.source.drain()
    hist: Dict[str, List[TagRead]] = ss.history
    for r in new:
        buf = hist.setdefault(r.epc, [])
        buf.append(r)
        if len(buf) > maxlen:
            del buf[: len(buf) - maxlen]
    return new


def history() -> Dict[str, List[TagRead]]:
    return _ss().history


def clear_history() -> None:
    _ss().history = {}


def recent_reads(window_s: float) -> List[TagRead]:
    """Flatten the last ``window_s`` seconds of reads across all tags."""
    flat = [r for buf in _ss().history.values() for r in buf]
    if not flat:
        return []
    latest = max(r.timestamp for r in flat)
    return [r for r in flat if r.timestamp >= latest - window_s]


# --------------------------------------------------------------------------- #
# Misc UI
# --------------------------------------------------------------------------- #
def live_rerun(interval_s: float = 0.8) -> None:
    time.sleep(interval_s)
    st.rerun()


def page_header(icon: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"<h1 style='margin-bottom:0'>{icon} {title}</h1>"
        f"<p style='color:#9aa0a6;margin-top:4px'>{subtitle}</p>",
        unsafe_allow_html=True,
    )
