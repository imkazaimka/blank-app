"""Shared Streamlit helpers for the Zebra RFID tool suite.

Keeps the per-page code small: session-state wiring, the source-selection
sidebar (simulator vs. live LLRP reader), a live-refresh loop and a few
formatting helpers.  Importable by ``app.py`` and every file in ``pages/``.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import streamlit as st

from rfid.models import Device, TagRead
from rfid.simulator import Scene, demo_scene
from rfid.sources import LLRPTagSource, SimulatedTagSource, TagSource

BRAND = "#00B5E2"


# --------------------------------------------------------------------------- #
# Optional-dependency probe (shown on the home page / sidebars)
# --------------------------------------------------------------------------- #
def optional_capabilities() -> Dict[str, bool]:
    caps = {}
    for mod, label in [
        ("sllurp", "Live LLRP reads (sllurp)"),
        ("zeroconf", "mDNS discovery (zeroconf)"),
        ("wsdiscovery", "WS-Discovery (WSDiscovery)"),
        ("pysnmp", "SNMP enrich (pysnmp)"),
    ]:
        try:
            __import__(mod)
            caps[label] = True
        except Exception:
            caps[label] = False
    return caps


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
def _ss() -> "st.session_state":
    ss = st.session_state
    ss.setdefault("source", None)          # active TagSource
    ss.setdefault("source_kind", None)     # "sim" | "live"
    ss.setdefault("scene", None)           # active Scene (sim mode)
    ss.setdefault("devices", [])           # discovered devices
    ss.setdefault("selected_device", None)
    ss.setdefault("history", {})           # epc -> list[TagRead] rolling buffer
    return ss


def get_scene() -> Scene:
    ss = _ss()
    if ss.scene is None:
        ss.scene = demo_scene()
    return ss.scene


def set_scene(scene: Scene) -> None:
    ss = _ss()
    stop_source()
    ss.scene = scene


def get_source() -> Optional[TagSource]:
    return _ss().source


def start_sim_source(rate_hz: float = 6.0) -> TagSource:
    ss = _ss()
    stop_source()
    src = SimulatedTagSource(get_scene(), rate_hz=rate_hz)
    src.start()
    ss.source = src
    ss.source_kind = "sim"
    return src


def start_live_source(device: Device, tx_power_dbm: Optional[int] = None) -> TagSource:
    ss = _ss()
    stop_source()
    src = LLRPTagSource(device.ip, port=device.port or 5084, tx_power=tx_power_dbm)
    src.start()
    ss.source = src
    ss.source_kind = "live"
    return src


def stop_source() -> None:
    ss = _ss()
    if ss.source is not None:
        try:
            ss.source.stop()
        except Exception:
            pass
    ss.source = None
    ss.source_kind = None


# --------------------------------------------------------------------------- #
# Rolling history of reads (per EPC), populated from the active source
# --------------------------------------------------------------------------- #
def pump(maxlen: int = 400) -> List[TagRead]:
    """Drain the active source into the per-EPC rolling history.

    Returns the newly-drained reads.  Safe to call on every rerun.
    """
    ss = _ss()
    src = ss.source
    if src is None:
        return []
    new = src.drain()
    hist: Dict[str, List[TagRead]] = ss.history
    for r in new:
        buf = hist.setdefault(r.epc, [])
        buf.append(r)
        if len(buf) > maxlen:
            del buf[: len(buf) - maxlen]
    return new


def clear_history() -> None:
    _ss().history = {}


def history() -> Dict[str, List[TagRead]]:
    return _ss().history


# --------------------------------------------------------------------------- #
# Sidebar: choose & control the data source
# --------------------------------------------------------------------------- #
def source_sidebar(key_prefix: str = "") -> Tuple[TagSource, bool]:
    """Render the standard source controls in the sidebar.

    Returns ``(source, live_refresh)`` where ``live_refresh`` indicates the page
    should auto-rerun to animate.
    """
    ss = _ss()
    with st.sidebar:
        st.markdown("### Data source")
        dev: Optional[Device] = ss.selected_device
        can_live = dev is not None and dev.kind.value == "reader"

        options = ["Simulator"]
        if can_live:
            options.append(f"Live reader ({dev.label})")
        mode = st.radio(
            "Read from",
            options,
            key=f"{key_prefix}src_mode",
            help="Pick a device on the 'Select Device' page to enable live reads.",
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶ Start", key=f"{key_prefix}start", use_container_width=True):
                if mode.startswith("Live") and dev is not None:
                    src = start_live_source(dev)
                    if src.last_error:
                        st.error(src.last_error)
                        start_sim_source()
                else:
                    start_sim_source()
        with col2:
            if st.button("⏹ Stop", key=f"{key_prefix}stop", use_container_width=True):
                stop_source()

        running = ss.source is not None and ss.source.running
        kind = ss.source_kind
        if running:
            badge = "🟢 LIVE reader" if kind == "live" else "🟡 Simulator"
            st.success(f"Streaming — {badge}")
        else:
            st.info("Stopped. Press Start to stream.")

        live_refresh = st.checkbox(
            "Auto-refresh", value=running, key=f"{key_prefix}auto",
            help="Continuously rerun the page to animate live data.",
        )
        st.caption("Tip: pick/scan devices on the **Select Device** page.")

    # Guarantee a source exists so pages always have data to show.
    if ss.source is None:
        start_sim_source()
    return ss.source, live_refresh and (ss.source is not None and ss.source.running)


def live_rerun(interval_s: float = 0.8) -> None:
    """Sleep briefly then rerun — the simple Streamlit live-animation loop."""
    time.sleep(interval_s)
    st.rerun()


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #
def rssi_color(rssi: float) -> str:
    """Green (strong) -> red (weak) colour for an RSSI value in dBm."""
    # Map -80..-35 dBm to 0..1.
    t = max(0.0, min(1.0, (rssi + 80) / 45.0))
    r = int(255 * (1 - t))
    g = int(200 * t)
    return f"rgb({r},{g},80)"


def rssi_bar(rssi: float) -> str:
    t = max(0.0, min(1.0, (rssi + 80) / 45.0))
    filled = int(round(t * 10))
    return "▮" * filled + "▯" * (10 - filled)


def page_header(icon: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"<h1 style='margin-bottom:0'>{icon} {title}</h1>"
        f"<p style='color:#9aa0a6;margin-top:4px'>{subtitle}</p>",
        unsafe_allow_html=True,
    )
