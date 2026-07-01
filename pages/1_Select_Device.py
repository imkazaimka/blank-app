"""Tool 1 - Select Device: scan the LAN for Zebra readers & printers."""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Select Device", page_icon="🔎", layout="wide")

from app_common import _ss, page_header
from rfid.discovery import demo_devices, discover_all, local_ipv4_networks
from rfid.models import DeviceKind

_ss()
page_header("🔎", "Select Device", "Find every Zebra reader and printer on the network — and how to connect.")

# --------------------------------------------------------------------------- #
# Sidebar: scan configuration
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("### Scan options")
    nets = local_ipv4_networks()
    net_label = ", ".join(str(n) for n in nets) or "unknown"
    st.caption(f"Detected subnet(s): `{net_label}`")
    subnet = st.text_input("Subnet to scan (CIDR)", value=str(nets[0]) if nets else "192.168.1.0/24")
    st.markdown("**Discovery layers**")
    do_sweep = st.checkbox("TCP port sweep (5084 / 9100)", value=True)
    do_udp = st.checkbox("UDP/4201 printer broadcast", value=True)
    do_ws = st.checkbox("WS-Discovery (readers)", value=True)
    do_mdns = st.checkbox("mDNS / Bonjour", value=True)
    do_snmp = st.checkbox("SNMP enrich (slower)", value=False)
    timeout = st.slider("Per-layer timeout (s)", 1.0, 8.0, 3.0, 0.5)

    st.divider()
    c1, c2 = st.columns(2)
    scan = c1.button("📡 Scan", type="primary", use_container_width=True)
    demo = c2.button("🧪 Demo", use_container_width=True,
                     help="Load example devices to preview the UI without hardware.")

# --------------------------------------------------------------------------- #
# Run discovery
# --------------------------------------------------------------------------- #
if scan:
    with st.spinner(f"Scanning {subnet} — WS-Discovery, UDP/4201, mDNS, TCP sweep…"):
        try:
            devs = discover_all(
                do_sweep=do_sweep, do_udp=do_udp, do_ws=do_ws, do_mdns=do_mdns,
                do_snmp=do_snmp, timeout=timeout, network=subnet or None,
            )
        except Exception as exc:
            st.error(f"Scan error: {exc}")
            devs = []
    _ss().devices = devs
    if not devs:
        st.warning(
            "No Zebra devices answered on this network. If you have no hardware here, "
            "click **Demo** to preview the tool, or check that readers/printers are on "
            "the same L2 subnet (WS-Discovery & UDP broadcast do not cross routers)."
        )

if demo:
    _ss().devices = demo_devices()

devices = _ss().devices

# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
if not devices:
    st.info("Press **Scan** to search your network, or **Demo** to load example devices.")
    st.stop()

readers = [d for d in devices if d.kind is DeviceKind.READER]
printers = [d for d in devices if d.kind is DeviceKind.PRINTER]
others = [d for d in devices if d.kind is DeviceKind.UNKNOWN]

m1, m2, m3 = st.columns(3)
m1.metric("Readers", len(readers))
m2.metric("Printers", len(printers))
m3.metric("Other / unknown", len(others))

sel = _ss().selected_device


def device_card(dev) -> None:
    global sel
    kind_icon = {"reader": "📶", "printer": "🖨️", "unknown": "❓"}[dev.kind.value]
    with st.container(border=True):
        top, action = st.columns([4, 1])
        with top:
            title = f"{kind_icon} **{dev.model or dev.kind.value.title()}**  ·  `{dev.ip}`"
            if sel and sel.ip == dev.ip:
                title += "  ✅ _selected_"
            st.markdown(title)
            meta = []
            if dev.hostname:
                meta.append(f"host `{dev.hostname}`")
            if dev.firmware:
                meta.append(f"fw {dev.firmware}")
            meta.append(f"via {dev.source}")
            if dev.kind is DeviceKind.PRINTER and dev.rfid_capable is not None:
                meta.append("🏷️ RFID-capable" if dev.rfid_capable else "no RFID encoder")
            st.caption(" · ".join(meta))
            if dev.services:
                chips = "  ".join(f"`{lbl}:{port}`" for lbl, port in dev.services.items())
                st.markdown("Open ports: " + chips)
            st.markdown("**Connect:** " + dev.connect_hint())
            if dev.extra.get("discovery_tokens"):
                with st.expander("Raw discovery data"):
                    st.code(dev.extra["discovery_tokens"])
        with action:
            if dev.kind is DeviceKind.READER:
                if st.button("Select", key=f"sel_{dev.ip}", use_container_width=True):
                    _ss().selected_device = dev
                    st.rerun()
            else:
                st.caption("Printer\n(control via ZPL)")


if readers:
    st.subheader("📶 RFID Readers")
    st.caption("Selecting a reader lets the other tools stream **live** reads from it over LLRP.")
    for d in readers:
        device_card(d)

if printers:
    st.subheader("🖨️ RFID Printers")
    for d in printers:
        device_card(d)

if others:
    st.subheader("❓ Other devices")
    for d in others:
        device_card(d)

# --------------------------------------------------------------------------- #
# Selected device banner
# --------------------------------------------------------------------------- #
st.divider()
sel = _ss().selected_device
if sel:
    st.success(
        f"**Active device:** {sel.label} — the Antenna, Denoising and Obstruction tools "
        f"will offer live reads from this reader. (They fall back to the simulator otherwise.)"
    )
    if st.button("Clear selection"):
        _ss().selected_device = None
        st.rerun()
else:
    st.info("No reader selected yet — the other tools will run against the **simulator**.")
