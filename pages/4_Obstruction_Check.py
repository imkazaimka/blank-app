"""Tool 4 - Obstruction Check: detect a blocked reader<->tag line of sight."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Obstruction Check", page_icon="🚧", layout="wide")

from app_common import _ss, get_scene, live_rerun, page_header, pump, source_sidebar
from rfid.obstruction import (
    ObstructionClassifier, build_baselines, detect, make_training_set,
)
from rfid.simulator import Obstruction

_ss()
_ss().setdefault("baselines", {})
_ss().setdefault("obst_clf", None)
page_header("🚧", "Obstruction Check",
            "Is something blocking the reader ➜ tag path? Watch RSSI drop, rising variance and read-rate loss.")

# --------------------------------------------------------------------------- #
# Controls
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("### Detection thresholds")
    drop = st.slider("RSSI drop to flag (dB)", 2.0, 12.0, 5.0, 0.5)
    ostd = st.slider("Obstructed RSSI std (dB)", 2.0, 8.0, 4.5, 0.5)
    rrlow = st.slider("Low read-rate ratio", 0.1, 0.9, 0.5, 0.05)
    window_s = st.slider("Detection window (s)", 1.0, 6.0, 3.0, 0.5)

    st.markdown("### Learned classifier")
    use_ml = st.checkbox("Blend ML classifier", value=False)
    if st.button("Train ML on simulator", use_container_width=True):
        with st.spinner("Generating labelled windows & fitting logistic regression…"):
            X, y = make_training_set(240)
            clf = ObstructionClassifier().fit(X, y)
            _ss().obst_clf = clf
        st.success("Classifier trained.")

source, live = source_sidebar(key_prefix="ob_")
scene = get_scene()

# --------------------------------------------------------------------------- #
# Simulator: inject / clear an obstruction for the demo
# --------------------------------------------------------------------------- #
if not source.is_live:
    with st.sidebar:
        st.markdown("### Simulate a blocker")
        block = st.checkbox("Put an object between Ant 1 and a tag")
        atten = st.slider("Blocker attenuation (dB)", 3.0, 20.0, 8.0, 1.0,
                          help="Body ~3-8 dB · metal/liquid much more.")
        if block and scene.tags:
            a = next(iter(scene.antennas.values()))
            t = scene.tags[0]
            mx, my = (a.x + t.x) / 2, (a.y + t.y) / 2
            scene.obstructions = [Obstruction(x=mx, y=my, radius=0.45,
                                              attenuation_db=atten, label="injected")]
        elif not block:
            scene.obstructions = [o for o in scene.obstructions if o.label != "injected"]

pump()
hist = _ss().history

# --------------------------------------------------------------------------- #
# Baseline calibration
# --------------------------------------------------------------------------- #
c1, c2, c3 = st.columns([1, 1, 2])
if c1.button("📏 Calibrate clear-LOS baseline", type="primary", use_container_width=True):
    flat = [r for buf in hist.values() for r in buf]
    if flat:
        _ss().baselines = build_baselines(flat)
        st.success(f"Baseline captured for {len(_ss().baselines)} antenna/tag pair(s).")
    else:
        st.warning("No reads yet — start streaming first.")
if c2.button("Clear baseline", use_container_width=True):
    _ss().baselines = {}

baselines = _ss().baselines
if not baselines:
    c3.info("Capture a **clear line-of-sight** baseline first, then introduce the obstruction.")
    if live:
        live_rerun()
    st.stop()

# --------------------------------------------------------------------------- #
# Build the recent detection window and run the detector
# --------------------------------------------------------------------------- #
flat = [r for buf in hist.values() for r in buf]
if flat:
    latest = max(r.timestamp for r in flat)
    window = [r for r in flat if r.timestamp >= latest - window_s]
else:
    window = []

clf = _ss().obst_clf if use_ml else None
verdicts = detect(window, baselines, window_s=window_s, classifier=clf,
                  rssi_drop_db=drop, obst_std_db=ostd, read_rate_low=rrlow)

n_obs = sum(1 for v in verdicts if v.obstructed)
m1, m2, m3 = st.columns(3)
m1.metric("Pairs monitored", len(verdicts))
m2.metric("Obstructed", n_obs, delta=None)
m3.metric("Clear", len(verdicts) - n_obs)

# --------------------------------------------------------------------------- #
# Per-pair status
# --------------------------------------------------------------------------- #
for v in verdicts:
    base = baselines.get((v.antenna, v.epc))
    with st.container(border=True):
        head, gauge = st.columns([3, 2])
        with head:
            icon = "🟥 OBSTRUCTED" if v.obstructed else "🟩 CLEAR"
            st.markdown(f"### {icon} — Ant {v.antenna} · `{v.epc}`")
            st.progress(min(v.confidence, 1.0), text=f"confidence {v.confidence*100:.0f}%")
            for reason in v.reasons:
                st.markdown(f"- {reason}")
        with gauge:
            f = v.features
            st.metric("RSSI Δ vs baseline", f"{f.rssi_delta:+.1f} dB")
            cc1, cc2 = st.columns(2)
            cc1.metric("RSSI std", f"{f.rssi_std:.1f} dB")
            cc2.metric("Read rate", f"{f.read_rate_ratio*100:.0f}%")

        # RSSI-vs-baseline trace for this pair.
        pair_reads = [r for r in hist.get(v.epc, []) if r.antenna == v.antenna][-80:]
        if pair_reads and base:
            ys = [r.rssi for r in pair_reads]
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=ys, mode="lines+markers", name="RSSI",
                                     line=dict(color="#00B5E2"), marker=dict(size=3)))
            fig.add_hline(y=base.mean_rssi, line=dict(color="lime", dash="dash"),
                          annotation_text="clear baseline")
            fig.add_hline(y=base.mean_rssi - drop, line=dict(color="orange", dash="dot"),
                          annotation_text="obstruction threshold")
            fig.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10),
                              yaxis_title="RSSI (dBm)", showlegend=False)
            st.plotly_chart(fig, use_container_width=True, key=f"tr_{v.antenna}_{v.epc}")

st.caption("Detector: threshold vote on RSSI-drop-vs-baseline + RSSI variance + read-rate ratio "
           "(a fully blocked path yields **no reads**, treated as strong evidence). Optionally "
           "blended with a numpy logistic-regression classifier trained on simulated windows.")

if live:
    live_rerun()
