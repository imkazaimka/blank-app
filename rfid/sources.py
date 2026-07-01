"""A single streaming interface over real and simulated tag data.

The four tools never talk to a reader directly; they consume a
:class:`TagSource`.  Two implementations are provided:

* :class:`SimulatedTagSource` - drives a :class:`~rfid.simulator.Scene` on a
  background thread (always available).
* :class:`LLRPTagSource` - wraps ``sllurp`` to stream real reads from a Zebra
  FX/FXR reader over LLRP (used when the ``sllurp`` package is installed and a
  reader is reachable).

Both expose the same tiny contract::

    src.start()
    reads = src.drain()      # list[TagRead] accumulated since last drain
    src.stop()

so a Streamlit page can poll ``drain()`` on every rerun regardless of the
underlying transport.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Deque, List, Optional, Sequence

from .models import PORT_LLRP, TagRead
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
