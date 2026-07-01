# 🚧 Zebra RFID — Obstruction Detector

A focused, single-purpose Streamlit tool: decide whether the line of sight
between a Zebra reader antenna and a tag is **CLEAR** or **OBSTRUCTED** — and,
crucially, tell a **blocked path** apart from a tag that has simply **moved
away**. It ingests reads from a live reader over **MQTT** (Zebra IoT Connector)
or **LLRP**, or from a physics-based **simulator**.

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

Two transports are supported; pick one in the sidebar under **Live reader**.

### MQTT — Zebra IoT Connector (recommended)

```bash
pip install paho-mqtt
```

This is Zebra's recommended path: the reader's built-in **IoT Connector**
publishes tag JSON to an MQTT broker and the app subscribes — no LLRP state
machine to manage.

1. On the reader (FX7500 / FX9600 / FXR90 / ATR7000), enable **IoT Connector**
   and point its **Tag-Data** interface at your MQTT broker, using a topic like
   `zebra/<reader-name>/data`.
2. In the sidebar choose **MQTT (IoT Connector)**, enter the **broker host**
   (port 1883, or 8883 with TLS) and the **topic** (`zebra/+/data` matches any
   reader), optional auth, then **Connect**.

The parser is tolerant of IoT Connector message variants — managed
`SimpleTagEvent` or raw, single or batched, wrapped in a `data` object or flat —
and normalises `idHex`/`peakRssi`/`antenna`/`phase`/`timestamp` into reads.

### LLRP (direct)

```bash
pip install sllurp
```

Choose **LLRP** and enter the reader IP (port 5084). Note Zebra FX readers report
RSSI (used here); the simulator additionally supplies Doppler/phase so the
moving/static label is demonstrable without an Impinj / ATR7000 reader.

### Antennas are auto-detected

You don't tell the app how many antennas you have. It **reads the antenna count
straight from the tag stream** (the `antenna` field on every read), so present a
tag to the reader after connecting and it lists exactly the antennas that are
reporting. The only thing it asks for is each detected antenna's **floor
position** — which the reader genuinely can't know — used to place and localize
the tag. (Motion-aware detection needs ≥ 3 antennas reporting.)

## Architecture

```
app.py                 # the Obstruction Detector UI (single page)
app_common.py          # session state, data-source lifecycle, live-refresh loop
rfid/
  obstruction.py       # motion-invariant analyzer + fixed-zone baseline + ML classifier
  ranging.py           # RSSI<->distance + 2D localization (used to place the tag)
  sources.py           # TagSource: MQTT (IoT Connector), LLRP (sllurp), or simulator
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
