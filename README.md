# 📡 Zebra RFID — Testing & Calibration Toolkit

A Streamlit app with **four tools** for bringing up, testing and calibrating
Zebra UHF RFID installations. It talks to real hardware over **LLRP**, and falls
back to a physics‑based **simulator** so every tool is fully usable without a
reader on your desk.

| # | Tool | What it does |
|---|------|--------------|
| 1 | **Select Device** | Scans the LAN for every Zebra reader & printer, identifies the model, checks RFID capability, and shows exactly how to connect. |
| 2 | **Antenna & Location** | Turns live RSSI into distance and plots each tag on a 2D field with a signal‑strength heatmap and an uncertainty ring. |
| 3 | **Signal Denoising** | Cleans up jumpy RSSI/phase with a Kalman filter or a learned (ML) denoiser — the smoothing people actually use online. |
| 4 | **Obstruction Check** | Detects when a person / metal / liquid is blocking the reader‑to‑tag path, from RSSI drop, rising variance and read‑rate loss. |

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the app, start at **Select Device** (or hit **Demo** to load example
devices), then jump into any tool — each one auto‑starts the simulator so you
always have live data to work with.

To talk to **real hardware**, install the optional extras:

```bash
pip install sllurp zeroconf WSDiscovery pysnmp
```

## The four tools

### 1 · Select Device — network discovery
Fixed readers and printers are found with *different* mechanisms, layered and
de‑duplicated:

| Device | Discovery | Control port |
|--------|-----------|--------------|
| **FX7500 / FX9600 / FXR90 / ATR7000** | WS‑Discovery (SOAP‑over‑UDP multicast `239.255.255.250:3702`) + TCP sweep of `5084/5085` | LLRP · TCP **5084** (TLS **5085**) |
| **ZT411/R, ZT421, ZT610/ZT620R, ZD621R, R110Xi4** | UDP broadcast **4201** (probe `2E 2C 3A 01 00 00`, reply magic `3A 2C 2E`), mDNS `_pdl-datastream._tcp`, SNMP | Raw ZPL · TCP **9100** |
| **RFD40 / RFD90 / RFD8500 handhelds** | *not networked* — Bluetooth SPP / USB | — |

- There is **no** Zebra mDNS `_llrp._tcp` service — readers use **WS‑Discovery**
  (RDMP / ISO 24791‑3 / DPWS), so we don’t assume one.
- Zero‑config readers self‑assign `169.254.x.x` and a hostname like
  `FX9600CD3B0D` (model + last 3 MAC octets).
- Printer **RFID capability** is confirmed live with an SGD
  `! U1 getvar "rfid.tag.type"` over port 9100.
- A dependency‑free **TCP port sweep** works everywhere; WS‑Discovery, mDNS and
  SNMP are optional enrichers.

### 2 · Antenna & Location — 2D field
- **Log‑distance path‑loss** converts `PeakRSSI (dBm)` to distance:
  `d = d₀·10^((RSSI_ref − RSSI)/(10·n))`.
- **Calibrate** `RSSI_ref` and `n` from field data (least‑squares fit).
- **Localize**: one antenna → range + heading arc; two or more →
  weighted‑least‑squares (Gauss‑Newton) **multilateration** — verified to
  recover a known position exactly (no sign bug).
- Overhead **ATR7000** azimuth/elevation → floor `(x, y)` for RTLS.
- Live heatmap of best expected RSSI, per‑antenna range rings, and a
  `±metres` uncertainty disc that grows with distance.

### 3 · Signal Denoising — clean, readable reads
Raw UHF RSSI is jumpy (multipath, tag orientation, antenna switching). Available
denoisers:

- **Median + Kalman** *(recommended)* — the empirically best classical pipeline
  for RFID/BLE RSSI (median kills impulsive dropouts, Kalman smooths).
- **Kalman** — the single most‑used online RSSI smoother (random‑walk latent
  state; defaults tuned per the widely‑cited Wouter Bulten RSSI post).
- **Learned denoiser (ML)** — a tiny self‑supervised denoising autoencoder
  (numpy only) that learns the live stream’s temporal structure; no offline
  dataset or heavyweight framework.
- Exponential (EWMA), moving average, Savitzky‑Golay.

RF **phase** is unwrapped (`np.unwrap`) *before* smoothing and re‑wrapped —
never smooth wrapped phase.

### 4 · Obstruction Check — is the line of sight blocked?
A blocked path shows four correlated symptoms, so we threshold the **delta from
a calibrated clear‑LOS baseline**, not absolute RSSI:

1. **RSSI drops** vs baseline (a body ≈ 3–8 dB; metal/liquid far more).
2. **RSSI variance rises** (the dominant ray is gone — reflections dominate).
3. **Read rate falls** — a *fully* blocked path yields **no reads**, treated as
   strong evidence.
4. **Phase jumps** not explained by tag motion.

The real‑time detector is an explainable threshold vote; an optional numpy
**logistic‑regression classifier** (trained on simulator‑labelled windows)
provides a learned second opinion.

## Architecture

```
app.py                     # home / launcher
app_common.py              # shared Streamlit helpers (source selection, live loop)
pages/
  1_Select_Device.py       # Tool 1 UI
  2_Antenna_and_Location.py# Tool 2 UI
  3_Signal_Denoising.py    # Tool 3 UI
  4_Obstruction_Check.py   # Tool 4 UI
rfid/                      # hardware-agnostic core library (numpy/scipy only)
  models.py                # Device, TagRead, AntennaConfig, ports/constants
  discovery.py             # Tool 1 backend (WS-Discovery, UDP/4201, mDNS, SNMP, sweep)
  ranging.py               # Tool 2 backend (path loss, calibration, localization, AoA)
  denoise.py               # Tool 3 backend (Kalman, median+Kalman, learned DAE)
  obstruction.py           # Tool 4 backend (features, thresholds, ML classifier)
  sources.py               # TagSource: LLRP (sllurp) + simulated, one interface
  simulator.py             # physics-based tag-read simulator (fallback / demos)
tests/                     # 34 unit + AppTest smoke tests (no hardware needed)
```

The four UI pages never talk to a reader directly — they consume a
`TagSource`, which is either an `LLRPTagSource` (real reader over LLRP via
`sllurp`) or a `SimulatedTagSource`. Same `start()/drain()/stop()` contract, so
the tools behave identically on hardware and in simulation.

## Connecting to a real reader
1. On **Select Device**, press **Scan** and pick a reader (needs it on the same
   L2 subnet — WS‑Discovery and UDP broadcast don’t cross routers).
2. Any tool’s sidebar now offers **Live reader** as a source. sllurp drives the
   LLRP handshake (`GET_READER_CAPABILITIES → ADD/ENABLE/START_ROSPEC →
   RO_ACCESS_REPORT`).
3. Zebra FX readers report standard fields (EPC, PeakRSSI dBm, antenna,
   timestamp, count). **Per‑tag phase/Doppler** is an Impinj / ATR7000 feature —
   the simulator synthesizes phase so the phase‑based views stay demonstrable.

## Tests

```bash
pip install pytest
pytest -q          # 34 passing: library maths + Streamlit page smoke tests
```

## Notes & limitations
- RSSI ranging is inherently coarse (meters); average many reads and calibrate
  `n`/`RSSI_ref` per site. Phase gives cm‑precision but is ambiguous every
  ~16 cm, so it’s used for motion, not absolute range.
- The UDP/4201 printer response field layout is from a single reverse‑engineered
  source — the IP (from the reply’s source address) and model token are robust;
  finer fields are best‑effort.
- Handheld sleds are Bluetooth/USB, not network‑discoverable, and are out of
  scope for LAN discovery.
