# 🚧 Zebra RFID — Obstruction Detector

A focused, single-purpose Streamlit tool: decide whether the line of sight
between a Zebra reader antenna and a tag is **CLEAR** or **OBSTRUCTED** — and,
crucially, tell a **blocked path** apart from a tag that has simply **moved
away**. It runs against a live reader over **LLRP** or a physics-based
**simulator**.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501`. It auto-starts the simulator, so you immediately
see a tag tracked by four antennas. Use the sidebar to **move the tag** and
**inject an obstruction** on any antenna's path, and watch the verdict update.

## The problem this solves

RSSI drops for two different reasons: (a) something blocks the path, or (b) the
tag just moved farther away. From one antenna's RSSI alone they're
**indistinguishable** — both just make the number go down. A naïve
"RSSI dropped below a baseline" detector therefore raises false alarms every
time a tag moves.

## How it tells them apart (motion-invariant detection)

Every window, the detector:

1. **Re-estimates the tag's position** from the antennas that agree
   (multilateration, with blocked-path outliers rejected).
2. **Predicts the RSSI each antenna should see** at that position from the
   log-distance path-loss model.
3. **Flags an antenna only when its measured RSSI sits well below prediction**
   (default ≥ 6 dB).

Because the position is recomputed each window, **movement is absorbed into the
position estimate** — all paths stay "green". Only an *unexplained* per-antenna
drop (a real blockage) turns a path "red". **Doppler** adds a moving/static
label on top.

| Situation | Result |
|---|---|
| Tag moves across the room, no blocker | **CLEAR** (all paths match the new position) |
| Static tag, one antenna blocked | **OBSTRUCTED** on that antenna |
| Tag moving *and* one path blocked | **OBSTRUCTED** on the blocked antenna |

### Requirements & limits
- Needs **≥ 3 antennas** with known positions (4 is more robust — one path can be
  blocked and the other three still fix the tag). With a single antenna and a
  moving tag the problem is **fundamentally ambiguous** from RSSI alone.
- Keep the **detection window short** for fast movers so the tag doesn't travel
  far within one window.
- It flags *unexplained attenuation*; it can't distinguish a person from metal —
  only that a specific path is blocked.

## Using a live Zebra reader

```bash
pip install sllurp
```

In the sidebar choose **Live reader (LLRP)**, enter the reader IP (LLRP port
5084) and each antenna's floor position, then **Connect**. Reads stream over
LLRP via `sllurp`. Note Zebra FX readers report RSSI (used here); the simulator
additionally supplies Doppler/phase so the moving/static label is demonstrable
without an Impinj/ATR7000 reader.

## Architecture

```
app.py                 # the Obstruction Detector UI (single page)
app_common.py          # session state, data-source lifecycle, live-refresh loop
rfid/
  obstruction.py       # motion-invariant analyzer + fixed-zone baseline + ML classifier
  ranging.py           # RSSI<->distance + 2D localization (used to place the tag)
  sources.py           # TagSource: live LLRP (sllurp) or simulator, one interface
  simulator.py         # physics-based reads (RSSI, phase, Doppler, obstructions)
  models.py            # TagRead, AntennaConfig
tests/
  test_obstruction.py  # 14 tests, incl. "moving tag is not flagged as an obstruction"
```

## Tests

```bash
pip install pytest
pytest -q      # 14 passing, no hardware required
```

The key test, `test_moving_tag_is_not_flagged_as_obstruction`, runs a fast-moving
unobstructed tag through many windows and asserts **zero** false obstructions.
