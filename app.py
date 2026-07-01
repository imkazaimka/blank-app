"""Zebra RFID Testing & Calibration Toolkit - home page.

A Streamlit multipage app with four tools for bringing up and calibrating Zebra
UHF RFID installations:

  1. Select Device     - scan the LAN for readers/printers and show how to connect.
  2. Antenna & Location - live 2D field with signal strength + approximate tag position.
  3. Signal Denoising  - smooth noisy RSSI/phase (Kalman + learned ML denoiser).
  4. Obstruction Check - detect when something blocks the reader<->tag line of sight.

Everything runs against real hardware over LLRP when a reader is reachable, and
falls back to a physics-based simulator so the tools are always demonstrable.
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Zebra RFID Toolkit",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

from app_common import BRAND, optional_capabilities, _ss

_ss()

st.markdown(
    f"<h1 style='margin-bottom:0'>📡 Zebra RFID <span style='color:{BRAND}'>Testing & Calibration</span> Toolkit</h1>",
    unsafe_allow_html=True,
)
st.caption(
    "Bring up, test and calibrate Zebra UHF RFID readers (FX7500 / FX9600 / FXR90 / ATR7000) "
    "and RFID printers (ZT411 / ZT421 / ZD621R). Works over LLRP against real hardware, with a "
    "physics-based simulator fallback when no reader is on the network."
)

st.divider()

# --------------------------------------------------------------------------- #
# The four tools
# --------------------------------------------------------------------------- #
tools = [
    (
        "1 · Select Device",
        "🔎",
        "Scan the local network for every Zebra reader and printer, identify the model, "
        "check RFID capability, and get a ready-to-paste connection command.",
        "pages/1_Select_Device.py",
    ),
    (
        "2 · Antenna & Location",
        "🎯",
        "Turn live RSSI into distance and plot each tag on a 2D field around your antennas, "
        "with a signal-strength heatmap and an uncertainty ring.",
        "pages/2_Antenna_and_Location.py",
    ),
    (
        "3 · Signal Denoising",
        "🧠",
        "Clean up jumpy readings with a Kalman filter or a learned (ML) denoiser, and see the "
        "noise-reduction gain in real time — the smoothing people use online.",
        "pages/3_Signal_Denoising.py",
    ),
    (
        "4 · Obstruction Check",
        "🚧",
        "Detect when a person, metal or liquid is blocking the reader-to-tag path by watching "
        "RSSI drop vs. a calibrated baseline, rising variance and falling read rate.",
        "pages/4_Obstruction_Check.py",
    ),
]

cols = st.columns(2)
for i, (name, icon, desc, path) in enumerate(tools):
    with cols[i % 2]:
        with st.container(border=True):
            st.markdown(f"### {icon} {name}")
            st.write(desc)
            try:
                st.page_link(path, label=f"Open {name.split('·')[1].strip()} →")
            except Exception:
                st.caption(f"Open **{name}** from the sidebar.")

st.divider()

# --------------------------------------------------------------------------- #
# Environment / capabilities + how it connects
# --------------------------------------------------------------------------- #
left, right = st.columns([1, 1])

with left:
    st.subheader("Environment")
    caps = optional_capabilities()
    for label, ok in caps.items():
        st.write(("✅ " if ok else "⚪ ") + label + ("" if ok else "  _(optional — pip install)_"))
    if not any(caps.values()):
        st.info(
            "No hardware libraries detected — that's fine. Every tool runs in **Simulator** "
            "mode. Install extras with `pip install sllurp zeroconf WSDiscovery pysnmp` to talk "
            "to real devices."
        )

with right:
    st.subheader("How devices connect")
    st.markdown(
        """
| Device | Discovery | Control |
|---|---|---|
| **FX / FXR readers** | WS-Discovery (UDP mcast 3702), TCP sweep 5084 | LLRP · TCP **5084** (TLS 5085) |
| **ATR7000** (RTLS) | WS-Discovery | LLRP + azimuth/elevation direction |
| **RFID printers** | UDP broadcast **4201**, mDNS, SNMP | Raw ZPL · TCP **9100** |
| **Handhelds** (RFD40/90) | not networked | Bluetooth SPP / USB |

Readers report per-tag **EPC, PeakRSSI (dBm), antenna, timestamp**; Impinj-class
readers and the ATR7000 add **RF phase / direction** used for fine ranging.
        """
    )

st.divider()
st.caption(
    "Start at **Select Device** to find hardware, or jump straight into any tool — each one "
    "auto-starts the simulator so you always have live data to work with."
)
