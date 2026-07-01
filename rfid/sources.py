"""A single streaming interface over real and simulated tag data.

The tool never talks to a reader directly; it consumes a :class:`TagSource`.
Three implementations are provided:

* :class:`SimulatedTagSource` - drives a :class:`~rfid.simulator.Scene` on a
  background thread (always available).
* :class:`MQTTTagSource` - subscribes to a Zebra **IoT Connector** tag-data
  topic on an MQTT broker and parses the JSON tag events (the recommended
  production path: the reader publishes, we subscribe - no LLRP state machine).
* :class:`LLRPTagSource` - wraps ``sllurp`` to stream real reads from a Zebra
  FX/FXR reader over LLRP.

Both expose the same tiny contract::

    src.start()
    reads = src.drain()      # list[TagRead] accumulated since last drain
    src.stop()

so a Streamlit page can poll ``drain()`` on every rerun regardless of the
underlying transport.
"""

from __future__ import annotations

import json
import math
import threading
import time
from collections import deque
from typing import Any, Deque, List, Optional, Sequence

from .models import PORT_LLRP, PORT_MQTT, TagRead
from .simulator import Scene


class TagSource:
    """Base class: a thread-safe buffer of recent reads."""

    def __init__(self, maxlen: int = 5000) -> None:
        self._buf: Deque[TagRead] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._running = False
        self.last_error: str = ""

    # -- lifecycle (override) --------------------------------------------- #
    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False

    @property
    def running(self) -> bool:
        return self._running

    @property
    def is_live(self) -> bool:
        return False

    # -- data ------------------------------------------------------------- #
    def _push(self, reads: Sequence[TagRead]) -> None:
        if not reads:
            return
        with self._lock:
            self._buf.extend(reads)

    def drain(self) -> List[TagRead]:
        with self._lock:
            out = list(self._buf)
            self._buf.clear()
        return out


class SimulatedTagSource(TagSource):
    """Streams reads from a :class:`Scene` at a fixed rate on a worker thread."""

    def __init__(self, scene: Scene, rate_hz: float = 5.0, maxlen: int = 5000) -> None:
        super().__init__(maxlen)
        self.scene = scene
        self.rate_hz = rate_hz
        self._thread: Optional[threading.Thread] = None

    @property
    def is_live(self) -> bool:
        return False

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        dt = 1.0 / max(self.rate_hz, 0.5)
        while self._running:
            self.scene.step(dt)
            self._push(self.scene.read_once())
            time.sleep(dt)

    def stop(self) -> None:
        self._running = False
        t = self._thread
        if t and t.is_alive():
            t.join(timeout=1.0)


class LLRPTagSource(TagSource):
    """Streams real reads from a Zebra reader over LLRP via ``sllurp``.

    The reader is driven on a background thread; each ``sllurp`` tag-report
    callback is normalised into :class:`~rfid.models.TagRead`.  If ``sllurp`` is
    not installed or the reader is unreachable, :meth:`start` records the error
    in ``last_error`` and the source simply yields nothing (the UI can then fall
    back to simulation).
    """

    def __init__(
        self,
        host: str,
        port: int = PORT_LLRP,
        antennas: Sequence[int] = (0,),
        tx_power: Optional[int] = None,
        maxlen: int = 5000,
    ) -> None:
        super().__init__(maxlen)
        self.host = host
        self.port = port
        self.antennas = list(antennas)
        self.tx_power = tx_power
        self._reader = None
        self._thread: Optional[threading.Thread] = None

    @property
    def is_live(self) -> bool:
        return True

    # -- normalisation ---------------------------------------------------- #
    @staticmethod
    def _to_read(tag: dict, host: str) -> Optional[TagRead]:
        """Convert one sllurp tag dict into a :class:`TagRead`.

        Field semantics follow sllurp 3.x (llrp_decoder.py / llrp_proto.py):

        * ``EPC``                  - bytes (EPC-96 is copied into ``EPC``).
        * ``PeakRSSI``             - signed dBm (standard LLRP).
        * ``ImpinjPeakRSSI``       - fine RSSI in centi-dBm (Impinj only).
        * ``ImpinjRFPhaseAngle``   - 12-bit 0..4095 == 0..2*pi (Impinj only).
        * ``ImpinjRFDopplerFrequency`` - signed Hz (Impinj only).
        * ``LastSeenTimestampUTC`` - microseconds since the Unix epoch.

        Zebra FX7500/FX9600 report only the *standard* fields (no per-tag phase
        or Doppler); those come from Impinj readers or Zebra's native API.
        """

        def pick(*keys, default=None):
            for k in keys:
                if k in tag and tag[k] is not None:
                    return tag[k]
            return default

        epc = pick("EPC", "EPC-96", "epc")
        if isinstance(epc, (bytes, bytearray)):
            epc = epc.hex().upper()
        if epc is None:
            return None

        # Prefer Impinj fine RSSI (centi-dBm) when present, else standard dBm.
        if tag.get("ImpinjPeakRSSI") is not None:
            rssi = float(tag["ImpinjPeakRSSI"]) / 100.0
        else:
            rssi = pick("PeakRSSI", "RSSI", "peak_rssi")
            if rssi is None:
                return None
            rssi = float(rssi)

        # Impinj RF phase: 12-bit unsigned, 0..4095 -> 0..2*pi radians.
        phase = None
        phase_raw = pick("ImpinjRFPhaseAngle", "RFPhaseAngle")
        if phase_raw is not None:
            phase = (float(phase_raw) / 4096.0) * 2.0 * 3.141592653589793

        ts_utc = pick("LastSeenTimestampUTC", "FirstSeenTimestampUTC")
        timestamp = float(ts_utc) / 1e6 if ts_utc is not None else time.time()

        return TagRead(
            epc=str(epc),
            rssi=rssi,
            antenna=int(pick("AntennaID", "antenna_id", default=1)),
            phase=phase,
            channel=pick("ChannelIndex", "channel_index"),
            timestamp=timestamp,
            seen_count=int(pick("TagSeenCount", "tag_seen_count", default=1)),
            doppler=pick("ImpinjRFDopplerFrequency", "RFDopplerFrequency"),
            reader=host,
        )

    def _callback(self, reader, tags) -> None:
        reads = []
        for tag in tags or []:
            r = self._to_read(tag, self.host)
            if r is not None:
                reads.append(r)
        self._push(reads)

    def start(self) -> None:
        if self._running:
            return
        try:
            from sllurp.llrp import (  # type: ignore
                LLRPReaderClient,
                LLRPReaderConfig,
            )
        except Exception as exc:  # pragma: no cover - hardware path
            self.last_error = f"sllurp not installed: {exc}"
            return

        cfg_dict = {
            "antennas": self.antennas,
            "report_every_n_tags": 1,
            "start_inventory": True,
            # Ask Impinj readers for phase/Doppler too (ignored by Zebra FX).
            "impinj_reports": True,
            "tag_content_selector": {
                "EnableROSpecID": False,
                "EnableAntennaID": True,
                "EnablePeakRSSI": True,
                "EnableFirstSeenTimestamp": True,
                "EnableLastSeenTimestamp": True,
                "EnableTagSeenCount": True,
                "EnableChannelIndex": True,
            },
        }
        if self.tx_power is not None:
            # sllurp maps this to the nearest legal power-table index.
            cfg_dict["tx_power_dbm"] = self.tx_power
        try:
            config = LLRPReaderConfig(cfg_dict)
            self._reader = LLRPReaderClient(self.host, self.port, config)
            self._reader.add_tag_report_callback(self._callback)
            self._reader.connect()
        except Exception as exc:  # pragma: no cover - hardware path
            self.last_error = f"LLRP connect failed: {exc}"
            self._reader = None
            return

        self._running = True
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def _spin(self) -> None:  # pragma: no cover - hardware path
        try:
            self._reader.join(None)
        except Exception as exc:
            self.last_error = f"LLRP stream error: {exc}"

    def stop(self) -> None:
        self._running = False
        r = self._reader
        if r is not None:  # pragma: no cover - hardware path
            try:
                r.disconnect()
            except Exception:
                pass
        self._reader = None


class MQTTTagSource(TagSource):
    """Streams reads from a Zebra reader via the **IoT Connector** over MQTT.

    A Zebra FX7500 / FX9600 / FXR90 / ATR7000 with IoT Connector configured to
    publish its *tag-data* interface to an MQTT broker sends JSON tag events on a
    topic like ``zebra/<reader-name>/data``.  We subscribe with ``paho-mqtt`` and
    normalise each event into :class:`~rfid.models.TagRead`.

    The reader's own message shape varies with firmware / configuration (managed
    ``SimpleTagEvent`` vs. raw, single object vs. batched array, wrapped in a
    ``data`` object or flat), so :meth:`_parse_payload` is deliberately tolerant
    of field-name and structure variants.  If ``paho-mqtt`` is missing or the
    broker is unreachable, :meth:`start` records the reason in ``last_error``.
    """

    # Accepted spellings for each field across firmware / config variants.
    _EPC_KEYS = ("idHex", "epc", "EPC", "epcHex", "id", "tagId")
    _RSSI_KEYS = ("peakRssi", "peak_rssi", "rssi", "PeakRSSI", "RSSI")
    _ANT_KEYS = ("antenna", "antennaPort", "antennaId", "antennaID", "AntennaID", "port")
    _TS_KEYS = ("timestamp", "reportTime", "lastSeenTime", "firstSeenTime", "time")
    _PHASE_KEYS = ("phase", "phaseAngle", "rfPhase", "RFPhaseAngle")
    _CH_KEYS = ("channel", "channelIndex", "ChannelIndex")
    _COUNT_KEYS = ("seenCount", "tagSeenCount", "numberOfReads", "count", "reads")
    _DOP_KEYS = ("doppler", "dopplerFrequency", "rfDoppler", "RFDopplerFrequency")

    def __init__(
        self,
        broker: str,
        port: int = PORT_MQTT,
        topic: str = "zebra/+/data",
        username: str = "",
        password: str = "",
        tls: bool = False,
        maxlen: int = 5000,
    ) -> None:
        super().__init__(maxlen)
        self.broker = broker
        self.port = port
        self.topic = topic
        self.username = username
        self.password = password
        self.tls = tls
        self._client = None

    @property
    def is_live(self) -> bool:
        return True

    # -- parsing ---------------------------------------------------------- #
    @staticmethod
    def _first(d: dict, keys, default=None):
        for k in keys:
            if k in d and d[k] is not None:
                return d[k]
        return default

    @classmethod
    def _coerce_timestamp(cls, val: Any) -> float:
        """IoT Connector timestamps may be epoch ms, epoch s, or ISO-8601."""
        if val is None:
            return time.time()
        if isinstance(val, (int, float)):
            v = float(val)
            if v > 1e14:      # microseconds
                return v / 1e6
            if v > 1e11:      # milliseconds
                return v / 1e3
            return v          # seconds
        if isinstance(val, str):
            s = val.strip().replace("Z", "+00:00")
            try:
                from datetime import datetime

                return datetime.fromisoformat(s).timestamp()
            except Exception:
                try:
                    return float(val)
                except Exception:
                    return time.time()
        return time.time()

    @classmethod
    def _event_to_read(cls, ev: dict, reader: str) -> Optional[TagRead]:
        # Tag fields live at the top level or inside a "data" object.
        body = ev.get("data") if isinstance(ev.get("data"), dict) else ev
        epc = cls._first(body, cls._EPC_KEYS)
        rssi = cls._first(body, cls._RSSI_KEYS)
        if epc is None or rssi is None:
            return None  # not a tag-read message (e.g. heartbeat/management)
        if isinstance(epc, (bytes, bytearray)):
            epc = epc.hex().upper()

        phase = cls._first(body, cls._PHASE_KEYS)
        if phase is not None:
            phase = float(phase)
            # IoT Connector reports phase in degrees; normalise to radians.
            if abs(phase) > 6.5:
                phase = math.radians(phase % 360.0)

        ts = cls._coerce_timestamp(
            cls._first(body, cls._TS_KEYS, default=cls._first(ev, cls._TS_KEYS))
        )
        reader_name = ev.get("hostName") or ev.get("reader") or reader
        return TagRead(
            epc=str(epc).upper(),
            rssi=float(rssi),
            antenna=int(cls._first(body, cls._ANT_KEYS, default=1)),
            phase=phase,
            channel=cls._first(body, cls._CH_KEYS),
            timestamp=ts,
            seen_count=int(cls._first(body, cls._COUNT_KEYS, default=1)),
            doppler=cls._first(body, cls._DOP_KEYS),
            reader=str(reader_name),
        )

    @classmethod
    def _parse_payload(cls, payload, reader: str = "MQTT") -> List[TagRead]:
        """Parse one MQTT message body into zero or more :class:`TagRead`."""
        if isinstance(payload, (bytes, bytearray)):
            payload = payload.decode("utf-8", "ignore")
        try:
            obj = json.loads(payload)
        except (ValueError, TypeError):
            return []
        # A message may be a single event, a list of events, or an envelope
        # whose "data" holds a list of events.
        events: List[dict]
        if isinstance(obj, list):
            events = [e for e in obj if isinstance(e, dict)]
        elif isinstance(obj, dict):
            data = obj.get("data")
            if isinstance(data, list):
                events = [e if isinstance(e, dict) else {} for e in data]
            else:
                events = [obj]
        else:
            return []
        out: List[TagRead] = []
        for ev in events:
            r = cls._event_to_read(ev, reader)
            if r is not None:
                out.append(r)
        return out

    # -- lifecycle -------------------------------------------------------- #
    def start(self) -> None:
        if self._running:
            return
        try:
            import paho.mqtt.client as mqtt  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dep
            self.last_error = f"paho-mqtt not installed: {exc}"
            return

        try:
            # paho-mqtt v2 requires an explicit callback API version; v1 doesn't.
            try:
                client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)  # type: ignore[attr-defined]
            except (AttributeError, TypeError):
                client = mqtt.Client()
            if self.username:
                client.username_pw_set(self.username, self.password or None)
            if self.tls:
                client.tls_set()
            client.on_connect = self._on_connect
            client.on_message = self._on_message
            client.connect(self.broker, self.port, keepalive=30)
            client.loop_start()  # paho runs its own background thread
        except Exception as exc:  # pragma: no cover - broker path
            self.last_error = f"MQTT connect failed: {exc}"
            self._client = None
            return
        self._client = client
        self._running = True

    def _on_connect(self, client, userdata, flags, rc, *args) -> None:  # pragma: no cover
        if rc == 0:
            client.subscribe(self.topic, qos=1)
        else:
            self.last_error = f"MQTT connect refused (rc={rc})"

    def _on_message(self, client, userdata, msg) -> None:  # pragma: no cover
        reader = ""
        try:
            parts = msg.topic.split("/")
            reader = parts[1] if len(parts) > 1 else msg.topic
        except Exception:
            pass
        self._push(self._parse_payload(msg.payload, reader or "MQTT"))

    def stop(self) -> None:
        self._running = False
        c = self._client
        if c is not None:  # pragma: no cover - broker path
            try:
                c.loop_stop()
                c.disconnect()
            except Exception:
                pass
        self._client = None
