"""Zebra RFID — Obstruction Detector.

A focused, single-purpose tool: decide whether the line of sight between a reader
antenna and a tag is CLEAR or OBSTRUCTED — and, crucially, tell a *blocked path*
apart from a tag that has simply *moved away*.

It does that with a motion-invariant method: every window it re-estimates the
tag's position from the antennas that agree, predicts the RSSI each antenna
*should* see at that position (log-distance path loss), and flags only antennas
whose measured RSSI sits well below prediction. Movement is absorbed into the
position estimate, so it no longer masquerades as an obstruction.

Runs against a live Zebra reader over LLRP, or a physics-based simulator.
"""

from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="RFID Obstruction Detector", page_icon="🚧", layout="wide")

from app_common import (
    BRAND, _ss, antennas, get_scene, live_rerun, page_header, paho_available, pump,
    recent_reads, set_scene, sllurp_available, source_running, start_live, start_mqtt,
    start_sim, stop_source,
)
from rfid.models import AntennaConfig
from rfid.obstruction import DOPPLER_MOVING_HZ, RESID_DROP_DB, analyze
from rfid.simulator import Obstruction, room_scene

_ss()
page_header("🚧", "RFID Obstruction Detector",
            "Is the reader→tag path blocked, or did the tag just move? This tells them apart.")

# --------------------------------------------------------------------------- #
# Sidebar — source + detection settings
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("### Data source")
    mode = st.radio("Read from", ["Simulator", "Live reader (LLRP)"], key="mode")

    if mode == "Simulator":
        scene = get_scene()
        st.markdown("**Path-loss calibration**")
        n = st.slider("Path-loss exponent n", 1.8, 4.0, 2.2, 0.1)
        rssi_ref = st.slider("Reference RSSI @1 m (dBm)", -60, -30, -45, 1)
        for cfg in scene.antennas.values():
            cfg.path_loss_n, cfg.rssi_ref = n, rssi_ref

        st.markdown("**Tag motion**")
        moving = st.checkbox("Tag is moving", value=False)
        speed = st.slider("Speed (m/s)", 0.1, 1.2, 0.5, 0.1, disabled=not moving)
        if scene.tags:
            t = scene.tags[0]
            if moving:
                # keep a steady heading; bounce off walls handles reversal
                if t.vx == 0 and t.vy == 0:
                    t.vx, t.vy = speed, speed * 0.6
                else:
                    mag = math.hypot(t.vx, t.vy) or 1.0
                    t.vx, t.vy = t.vx / mag * speed, t.vy / mag * speed
            else:
                t.vx, t.vy = 0.0, 0.0

        st.markdown("**Inject an obstruction**")
        block_ant = st.selectbox("Block the path to antenna",
                                 ["none"] + [f"Ant {a}" for a in scene.antennas])
        atten = st.slider("Blocker attenuation (dB)", 3.0, 20.0, 12.0, 1.0,
                          help="Body ≈ 3–8 dB · metal/liquid much more.")
        scene.obstructions = [o for o in scene.obstructions if o.label != "injected"]
        if block_ant != "none" and scene.tags:
            aid = int(block_ant.split()[1])
            a, t = scene.antennas[aid], scene.tags[0]
            mx, my = (a.x + t.x) / 2, (a.y + t.y) / 2
            scene.obstructions.append(
                Obstruction(x=mx, y=my, radius=0.55, attenuation_db=atten, label="injected"))

        c1, c2 = st.columns(2)
        if c1.button("▶ Start", use_container_width=True):
            start_sim()
        if c2.button("⏹ Stop", use_container_width=True):
            stop_source()
        if st.button("Reset scene", use_container_width=True):
            set_scene(room_scene())
            st.rerun()

    else:  # Live reader
        transport = st.radio("Transport", ["MQTT (IoT Connector)", "LLRP"],
                             help="MQTT is Zebra's recommended path: the reader's IoT "
                                  "Connector publishes tag JSON to a broker and we subscribe.")
        if transport.startswith("MQTT"):
            broker = st.text_input("MQTT broker host", value="192.168.1.100")
            mqtt_port = st.number_input("Broker port", value=1883, step=1,
                                        help="1883 plain · 8883 TLS")
            topic = st.text_input("Tag-data topic", value="zebra/+/data",
                                  help="IoT Connector publishes to zebra/<reader>/data. "
                                       "'+' is an MQTT wildcard for any reader name.")
            with st.expander("Auth / TLS"):
                user = st.text_input("Username", value="")
                pw = st.text_input("Password", value="", type="password")
                tls = st.checkbox("Use TLS (8883)", value=False)
            if not paho_available():
                st.warning("`paho-mqtt` not installed — run `pip install paho-mqtt` for MQTT.")
        else:
            ip = st.text_input("Reader IP", value="192.168.1.50")
            llrp_port = st.number_input("LLRP port", value=5084, step=1)
            if not sllurp_available():
                st.warning("`sllurp` not installed — run `pip install sllurp` for LLRP.")

        n_ant = st.number_input("Number of antennas", 3, 8, 4, 1,
                                help="Motion-aware detection needs ≥3 antennas with known positions.")
        st.caption("Antenna positions (metres):")
        live_ants = {}
        for i in range(1, int(n_ant) + 1):
            cc = st.columns(2)
            x = cc[0].number_input(f"A{i} x", value=float(0 if i in (1, 4) else 6), key=f"ax{i}")
            y = cc[1].number_input(f"A{i} y", value=float(0 if i in (1, 2) else 5), key=f"ay{i}")
            live_ants[i] = AntennaConfig(antenna_id=i, x=x, y=y)
        _ss().live_antennas = live_ants

        c1, c2 = st.columns(2)
        if c1.button("▶ Connect", use_container_width=True):
            if transport.startswith("MQTT"):
                src = start_mqtt(broker, int(mqtt_port), topic, user, pw, tls)
            else:
                src = start_live(ip, int(llrp_port))
            if src.last_error:
                st.error(src.last_error)
        if c2.button("⏹ Stop", use_container_width=True):
            stop_source()

    st.divider()
    st.markdown("### Detection")
    resid_drop = st.slider("Blocked if RSSI is this far below prediction (dB)",
                           3.0, 15.0, RESID_DROP_DB, 0.5)
    window_s = st.slider("Detection window (s)", 0.5, 4.0, 1.5, 0.5,
                         help="Keep short for fast-moving tags so the position stays well-defined.")
    dop_thr = st.slider("‘Moving’ Doppler threshold (Hz)", 1.0, 10.0, DOPPLER_MOVING_HZ, 0.5)
    auto = st.checkbox("Auto-refresh", value=source_running())

# Auto-start the simulator so there's always something to show.
if _ss().source is None and mode == "Simulator":
    start_sim()

pump()
ants = antennas()
reads = recent_reads(window_s)
reports = analyze(reads, ants, resid_drop_db=resid_drop, doppler_moving_hz=dop_thr)

# --------------------------------------------------------------------------- #
# Status banner
# --------------------------------------------------------------------------- #
running = source_running()
kind = _ss().source_kind
badge = ("🟢 LIVE reader" if kind == "live" else "🟡 Simulator") if running else "⚪ stopped"
n_blocked = sum(len(r.blocked_antennas) for r in reports)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Source", badge)
m2.metric("Tags tracked", len(reports))
m3.metric("Blocked paths", n_blocked)
m4.metric("Antennas", len(ants))

if len(ants) < 3:
    st.warning("Motion-aware detection needs **≥3 antennas** with known positions to separate "
               "movement from blockage. Add more antennas (or their positions) in the sidebar.")

if not reports:
    st.info("Waiting for reads from a tag seen by ≥3 antennas… press **Start** in the sidebar.")
    if auto and running:
        live_rerun()
    st.stop()


# --------------------------------------------------------------------------- #
# 2D field
# --------------------------------------------------------------------------- #
def field_figure() -> go.Figure:
    scene = get_scene()
    x0, y0, x1, y1 = scene.bounds if _ss().source_kind != "live" else (
        min(a.x for a in ants.values()) - 1, min(a.y for a in ants.values()) - 1,
        max(a.x for a in ants.values()) + 1, max(a.y for a in ants.values()) + 1)
    fig = go.Figure()
    # Antennas
    fig.add_trace(go.Scatter(
        x=[a.x for a in ants.values()], y=[a.y for a in ants.values()],
        mode="markers+text", text=[f"Ant {a.antenna_id}" for a in ants.values()],
        textposition="top center", marker=dict(size=16, symbol="square", color="white",
                                                line=dict(color="black", width=2)),
        name="Antennas", textfont=dict(color="white")))
    # Per tag: antenna→tag links coloured by blocked, plus the position star
    for rep in reports:
        if rep.position is None:
            continue
        px, py = rep.position
        for p in rep.paths:
            a = ants[p.antenna]
            color = "#ff4d4d" if p.obstructed else "rgba(0,200,120,0.7)"
            fig.add_trace(go.Scatter(
                x=[a.x, px], y=[a.y, py], mode="lines",
                line=dict(color=color, width=4 if p.obstructed else 2),
                hovertext=f"Ant {p.antenna}: {p.measured_rssi:.1f} dBm "
                          f"(expected {p.predicted_rssi:.1f}, Δ{p.residual:+.1f})",
                hoverinfo="text", showlegend=False))
        star = "#ff4d4d" if rep.obstructed else BRAND
        fig.add_trace(go.Scatter(
            x=[px], y=[py], mode="markers+text", text=[rep.epc[-4:]],
            textposition="bottom center",
            marker=dict(size=18, symbol="star", color=star, line=dict(color="white", width=1.5)),
            name=f"{rep.epc[-4:]} {'⛔' if rep.obstructed else '✓'}"))
    # Obstructions + true positions (simulator only)
    if _ss().source_kind != "live":
        for ob in scene.obstructions:
            th = np.linspace(0, 2 * np.pi, 40)
            fig.add_trace(go.Scatter(
                x=ob.x + ob.radius * np.cos(th), y=ob.y + ob.radius * np.sin(th),
                mode="lines", fill="toself", fillcolor="rgba(255,77,77,0.25)",
                line=dict(color="#ff4d4d"), name=f"blocker ({ob.attenuation_db:.0f} dB)",
                hoverinfo="name"))
        for t in scene.tags:
            fig.add_trace(go.Scatter(
                x=[t.x], y=[t.y], mode="markers",
                marker=dict(size=10, symbol="circle-open", color="lime", line=dict(width=2)),
                name=f"true {t.epc[-4:]}", hoverinfo="name"))
    fig.update_layout(
        height=520, xaxis_title="x (m)", yaxis_title="y (m)",
        xaxis=dict(range=[x0 - 0.5, x1 + 0.5]),
        yaxis=dict(range=[y0 - 0.5, y1 + 0.5], scaleanchor="x", scaleratio=1),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=10, r=10, t=30, b=10))
    return fig


left, right = st.columns([3, 2])
with left:
    st.plotly_chart(field_figure(), use_container_width=True)
    st.caption("Green link = path matches the tag's position (clear). Red link = RSSI far below "
               "what that position predicts (blocked). The star is the estimated tag position.")

with right:
    for rep in reports:
        with st.container(border=True):
            icon = "🟥 OBSTRUCTED" if rep.obstructed else "🟩 CLEAR"
            move = "🏃 moving" if rep.moving else "🧍 static"
            st.markdown(f"#### {icon} · `{rep.epc}`  ·  {move}")
            if rep.position:
                st.caption(f"Estimated position ({rep.position[0]:.1f}, {rep.position[1]:.1f}) m "
                           f"· ±{rep.position_uncertainty:.2f} m")
            rows = [{
                "Ant": p.antenna,
                "RSSI": round(p.measured_rssi, 1),
                "Expected": round(p.predicted_rssi, 1),
                "Δ dB": round(p.residual, 1),
                "": "⛔" if p.obstructed else "✓",
            } for p in rep.paths]
            st.dataframe(rows, hide_index=True, use_container_width=True)
            for reason in rep.reasons:
                st.markdown(f"- {reason}")

st.caption("Method: each window re-estimates the tag position from the agreeing antennas and "
           "compares every antenna's RSSI to the path-loss prediction at that position. Movement "
           "shifts the position (all paths stay green); a blocker drops one path below prediction "
           "(that link turns red). Doppler adds the moving/static label.")

if auto and running:
    live_rerun()
