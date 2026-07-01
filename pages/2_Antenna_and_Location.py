"""Tool 2 - Antenna & Location: 2D field with signal strength + tag position."""

from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Antenna & Location", page_icon="🎯", layout="wide")

from app_common import (
    _ss, get_scene, live_rerun, page_header, pump, rssi_bar, set_scene, source_sidebar,
)
from rfid.denoise import make_filter
from rfid.ranging import (
    Fix, distance_uncertainty, expected_rssi, localize, rssi_heatmap, rssi_to_distance,
)
from rfid.simulator import demo_scene, single_antenna_scene, spread_scene

_ss()
page_header("🎯", "Antenna & Location",
            "Signal strength ➜ distance ➜ approximate tag position on a live 2D field.")

# --------------------------------------------------------------------------- #
# Scene / calibration controls
# --------------------------------------------------------------------------- #
SCENES = {
    "Spread (3 antennas — true 2D)": spread_scene,
    "Dock door (2 antennas)": demo_scene,
    "Single antenna (range only)": single_antenna_scene,
}
with st.sidebar:
    st.markdown("### Scene")
    scene_name = st.selectbox("Antenna layout (simulator)", list(SCENES.keys()))
    if st.button("Load / reset scene", use_container_width=True):
        set_scene(SCENES[scene_name]())
        st.rerun()

    st.markdown("### Path-loss calibration")
    n = st.slider("Path-loss exponent n", 1.8, 4.0, 2.2, 0.1,
                  help="~2 free space, 2.5-3.5 cluttered indoor.")
    rssi_ref = st.slider("Reference RSSI @1 m (dBm)", -60, -30, -45, 1)
    smooth = st.checkbox("Denoise RSSI before ranging", value=True,
                         help="Apply the recommended Median+Kalman filter to each tag's RSSI.")

source, live = source_sidebar(key_prefix="loc_")

# Ensure a scene matching the selection is active.
scene = get_scene()
# Apply calibration to the live scene's antennas so the model & sim agree.
for cfg in scene.antennas.values():
    cfg.path_loss_n = n
    cfg.rssi_ref = rssi_ref

pump()
hist = _ss().history

# --------------------------------------------------------------------------- #
# Per-tag ranging from the most recent reads
# --------------------------------------------------------------------------- #
WINDOW = 12  # reads per antenna to average


def latest_ranges(epc: str):
    """Return {antenna: (distance, rssi, n_used)} for one tag."""
    reads = hist.get(epc, [])
    by_ant: dict[int, list[float]] = {}
    for r in reads[-WINDOW * len(scene.antennas):]:
        by_ant.setdefault(r.antenna, []).append(r.rssi)
    out = {}
    for ant, rssis in by_ant.items():
        if ant not in scene.antennas:
            continue
        vals = rssis[-WINDOW:]
        if smooth and len(vals) >= 3:
            vals = make_filter("Median + Kalman (recommended)").smooth(vals)
        rssi = float(np.mean(vals[-4:])) if len(vals) else float(np.mean(vals))
        out[ant] = (rssi_to_distance(rssi, scene.antennas[ant]), rssi, len(rssis))
    return out


tags = sorted(hist.keys())
if not tags:
    st.info("Waiting for reads… press **Start** in the sidebar.")
    live_rerun() if live else st.stop()

sel_tag = st.selectbox("Focus tag (draws distance rings + position)", tags)

# --------------------------------------------------------------------------- #
# Build the 2D field figure
# --------------------------------------------------------------------------- #
x0, y0, x1, y1 = scene.bounds
xs, ys, grid = rssi_heatmap(scene.antennas.values(), (x0, x1), (y0, y1), resolution=70)

fig = go.Figure()
fig.add_trace(go.Heatmap(
    z=grid, x=xs, y=ys, colorscale="Turbo", zmin=-85, zmax=-35,
    colorbar=dict(title="Best RSSI<br>(dBm)"), opacity=0.75, hoverinfo="skip",
))

# Antennas + heading arrows.
ax = [c.x for c in scene.antennas.values()]
ay = [c.y for c in scene.antennas.values()]
alabels = [f"Ant {c.antenna_id}" for c in scene.antennas.values()]
fig.add_trace(go.Scatter(
    x=ax, y=ay, mode="markers+text", text=alabels, textposition="top center",
    marker=dict(size=16, symbol="square", color="white", line=dict(color="black", width=2)),
    name="Antennas", textfont=dict(color="white", size=12),
))
for c in scene.antennas.values():
    th = math.radians(c.heading_deg)
    fig.add_annotation(x=c.x + 0.6 * math.cos(th), y=c.y + 0.6 * math.sin(th),
                       ax=c.x, ay=c.y, xref="x", yref="y", axref="x", ayref="y",
                       showarrow=True, arrowhead=2, arrowsize=1.2, arrowcolor="white")

# Focus tag: distance rings, true position, estimated position.
ranges = latest_ranges(sel_tag)
if ranges:
    for ant, (dist, rssi, npts) in ranges.items():
        c = scene.antennas[ant]
        band = distance_uncertainty(rssi, c)
        t = np.linspace(0, 2 * np.pi, 80)
        fig.add_trace(go.Scatter(
            x=c.x + dist * np.cos(t), y=c.y + dist * np.sin(t),
            mode="lines", line=dict(color="rgba(255,255,255,0.55)", dash="dot"),
            name=f"Ant {ant}: {dist:.2f} m", hoverinfo="name",
        ))

    fix: Fix | None = localize({a: d for a, (d, _, _) in ranges.items()}, scene.antennas)
    if fix:
        ring_t = np.linspace(0, 2 * np.pi, 60)
        fig.add_trace(go.Scatter(
            x=fix.x + fix.radius_m * np.cos(ring_t), y=fix.y + fix.radius_m * np.sin(ring_t),
            mode="lines", line=dict(color="#00B5E2", width=1),
            fill="toself", fillcolor="rgba(0,181,226,0.15)",
            name=f"±{fix.radius_m:.2f} m", hoverinfo="name",
        ))
        fig.add_trace(go.Scatter(
            x=[fix.x], y=[fix.y], mode="markers+text", text=[sel_tag[-4:]],
            textposition="bottom center",
            marker=dict(size=18, symbol="star", color="#00B5E2",
                        line=dict(color="white", width=1.5)),
            name="Estimated position",
        ))

# Ground-truth tag positions (simulator only).
if not source.is_live:
    for t in scene.tags:
        fig.add_trace(go.Scatter(
            x=[t.x], y=[t.y], mode="markers",
            marker=dict(size=11, symbol="circle-open", color="lime", line=dict(width=2)),
            name=f"true {t.epc[-4:]}", hoverinfo="name",
        ))

fig.update_layout(
    height=560, xaxis_title="x (m)", yaxis_title="y (m)",
    xaxis=dict(range=[x0 - 0.5, x1 + 0.5], constrain="domain"),
    yaxis=dict(range=[y0 - 0.5, y1 + 0.5], scaleanchor="x", scaleratio=1),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    margin=dict(l=10, r=10, t=30, b=10),
)

left, right = st.columns([3, 2])
with left:
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Dashed rings = per-antenna range from RSSI · cyan star = estimated position · "
               "cyan disc = uncertainty · green rings = true position (simulator).")

with right:
    st.markdown("#### Live signal & range")
    if ranges:
        fix = localize({a: d for a, (d, _, _) in ranges.items()}, scene.antennas)
        if fix:
            st.metric("Estimated position", f"({fix.x:.2f}, {fix.y:.2f}) m",
                      help=f"{fix.method}, {fix.n_antennas} antenna(s)")
            st.caption(f"Uncertainty ±{fix.radius_m:.2f} m")
        rows = []
        for ant, (dist, rssi, npts) in sorted(ranges.items()):
            rows.append({
                "Ant": ant, "RSSI (dBm)": round(rssi, 1),
                "Signal": rssi_bar(rssi), "Range (m)": round(dist, 2),
                "±(m)": round(distance_uncertainty(rssi, scene.antennas[ant]), 2),
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)
    st.markdown("#### All tags")
    summary = []
    for epc in tags:
        rr = latest_ranges(epc)
        if not rr:
            continue
        best = max(rr.values(), key=lambda v: v[1])
        summary.append({"EPC": epc, "Best RSSI": round(best[1], 1),
                        "Nearest range (m)": round(min(d for d, _, _ in rr.values()), 2),
                        "Antennas": len(rr)})
    if summary:
        st.dataframe(summary, use_container_width=True, hide_index=True)

if live:
    live_rerun()
