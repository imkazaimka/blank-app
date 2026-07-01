"""Tool 3 - Signal Denoising: smooth noisy RSSI/phase (Kalman + learned ML)."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Signal Denoising", page_icon="🧠", layout="wide")

from app_common import _ss, live_rerun, page_header, pump, source_sidebar
from rfid.denoise import FILTERS, denoise_phase, make_filter, snr_improvement_db

_ss()
page_header("🧠", "Signal Denoising",
            "Turn jumpy RSSI/phase into a clean, readable signal — the smoothing people use online.")

# --------------------------------------------------------------------------- #
# Filter controls
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("### Denoiser")
    fname = st.selectbox("Filter", list(FILTERS.keys()))
    params: dict = {}
    if fname.startswith("Median + Kalman"):
        params["median_window"] = st.slider("Median window", 3, 11, 5, 2)
        params["q"] = st.select_slider("Process noise Q", [0.001, 0.005, 0.01, 0.05, 0.1], 0.01)
        params["r"] = st.slider("Measurement noise R (dB²)", 1.0, 12.0, 4.0, 0.5)
    elif fname == "Kalman":
        params["q"] = st.select_slider("Process noise Q", [0.001, 0.005, 0.01, 0.05, 0.1], 0.01)
        params["r"] = st.slider("Measurement noise R (dB²)", 1.0, 12.0, 4.0, 0.5)
    elif fname.startswith("Learned"):
        params["window"] = st.slider("Input window", 5, 15, 9, 2)
        params["hidden"] = st.slider("Hidden units", 4, 32, 16, 4)
        params["lr"] = st.select_slider("Learning rate", [0.001, 0.005, 0.01, 0.05], 0.01)
    elif fname.startswith("Exponential"):
        params["alpha"] = st.slider("Alpha (smaller = smoother)", 0.05, 0.9, 0.3, 0.05)
    elif fname == "Moving average":
        params["window"] = st.slider("Window", 3, 21, 7, 2)
    elif fname.startswith("Savitzky"):
        params["window"] = st.slider("Window (odd)", 5, 21, 9, 2)
        params["poly"] = st.slider("Poly order", 1, 4, 2, 1)

source, live = source_sidebar(key_prefix="dn_")
pump()
hist = _ss().history

tags = sorted(hist.keys())
if not tags:
    st.info("Waiting for reads… press **Start** in the sidebar.")
    live_rerun() if live else st.stop()

sel = st.selectbox("Tag / antenna stream", tags)
reads = hist.get(sel, [])
# Focus on a single antenna's stream for a clean 1-D comparison.
ant_ids = sorted({r.antenna for r in reads})
ant = st.radio("Antenna", ant_ids, horizontal=True) if ant_ids else None
stream = [r for r in reads if r.antenna == ant]

raw = np.array([r.rssi for r in stream], dtype=float)
if len(raw) < 4:
    st.info("Collecting samples…")
    live_rerun() if live else st.stop()

clean = make_filter(fname, **params).smooth(raw)
idx = np.arange(len(raw))

# --------------------------------------------------------------------------- #
# RSSI: raw vs denoised
# --------------------------------------------------------------------------- #
fig = go.Figure()
fig.add_trace(go.Scatter(x=idx, y=raw, mode="lines+markers", name="raw RSSI",
                         line=dict(color="rgba(255,120,120,0.55)"), marker=dict(size=4)))
fig.add_trace(go.Scatter(x=idx, y=clean, mode="lines", name=f"denoised · {fname}",
                         line=dict(color="#00B5E2", width=3)))
fig.update_layout(height=380, xaxis_title="read #", yaxis_title="RSSI (dBm)",
                  legend=dict(orientation="h", y=1.02),
                  margin=dict(l=10, r=10, t=30, b=10))
st.plotly_chart(fig, use_container_width=True)

c1, c2, c3 = st.columns(3)
c1.metric("Noise reduction", f"{snr_improvement_db(raw, clean):.1f} dB",
          help="Drop in sample-to-sample jitter after filtering.")
c2.metric("Raw jitter (σ of Δ)", f"{np.std(np.diff(raw)):.2f} dB")
c3.metric("Denoised jitter", f"{np.std(np.diff(clean)):.2f} dB")

# --------------------------------------------------------------------------- #
# Compare every filter on the current stream
# --------------------------------------------------------------------------- #
with st.expander("Compare all filters on this stream", expanded=False):
    rows = []
    for name in FILTERS:
        out = make_filter(name).smooth(raw)
        rows.append({"Filter": name,
                     "Noise reduction (dB)": round(snr_improvement_db(raw, out), 1),
                     "Residual jitter (dB)": round(float(np.std(np.diff(out))), 2)})
    st.dataframe(sorted(rows, key=lambda r: -r["Noise reduction (dB)"]),
                 use_container_width=True, hide_index=True)
    st.caption("Median+Kalman and Kalman are the recommended online smoothers; the learned "
               "denoiser is a small self-supervised autoencoder that adapts to the live stream.")

# --------------------------------------------------------------------------- #
# Phase denoising (must unwrap first)
# --------------------------------------------------------------------------- #
phases = [r.phase for r in stream if r.phase is not None]
if len(phases) >= 5:
    st.markdown("#### RF phase (unwrapped before smoothing)")
    ph = np.array(phases, dtype=float)
    ph_clean = denoise_phase(ph)
    pfig = go.Figure()
    pfig.add_trace(go.Scatter(y=ph, mode="markers", name="raw phase (wrapped)",
                              marker=dict(size=5, color="rgba(255,180,80,0.6)")))
    pfig.add_trace(go.Scatter(y=ph_clean, mode="lines", name="denoised phase",
                              line=dict(color="#00B5E2", width=2)))
    pfig.update_layout(height=300, xaxis_title="read #", yaxis_title="phase (rad)",
                       legend=dict(orientation="h", y=1.02),
                       margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(pfig, use_container_width=True)
    st.caption("Phase is reported modulo 2π; it is unwrapped (`np.unwrap`) before filtering and "
               "re-wrapped — never smooth wrapped phase directly.")
else:
    st.caption("💡 Phase denoising appears when the stream carries RF phase "
               "(Impinj readers / ATR7000 / simulator).")

if live:
    live_rerun()
