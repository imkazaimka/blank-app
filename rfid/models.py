"""Core data models shared across the Zebra RFID tool suite.

These dataclasses describe the two entities the tools revolve around:

* :class:`Device` - a reader or printer found on the network.
* :class:`TagRead` - a single tag observation coming back from a reader.

Both are deliberately transport-agnostic so the same structures are produced
whether the reads come from a real LLRP session (:mod:`rfid.llrp_client`) or the
built-in :mod:`rfid.simulator`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class DeviceKind(str, Enum):
    """What kind of Zebra device we believe we found."""

    READER = "reader"
    PRINTER = "printer"
    UNKNOWN = "unknown"


# Well-known ports used by Zebra devices.  Kept here so discovery, connection
# hints and the UI all agree on a single source of truth.  (Values verified
# against Zebra FX/FXR integration guides & Link-OS docs — see rfid/discovery.py.)
PORT_LLRP = 5084          # Low Level Reader Protocol (fixed RFID readers)
PORT_LLRP_TLS = 5085      # LLRP over TLS (reader "secure mode")
PORT_RAW_PRINT = 9100     # Raw/JetDirect ZPL print channel
PORT_PRINTER_DISCOVERY = 4201  # Zebra multicast/broadcast printer discovery
PORT_WS_DISCOVERY = 3702  # WS-Discovery (fixed-reader auto discovery, multicast)
PORT_HTTP = 80
PORT_HTTPS = 443
PORT_SSH = 22
PORT_SNMP = 161
PORT_MQTT = 1883          # Zebra IoT Connector tag-data stream (8883 = TLS)

# WS-Discovery multicast group used by Zebra fixed readers (RDMP / DPWS).
WS_DISCOVERY_GROUP = "239.255.255.250"

# Fixed-reader default hostname prefixes (model + last 3 MAC octets) and creds.
READER_HOST_PREFIXES = ("FX9600", "FX7500", "FXR90", "ATR7000", "FXR")
DEFAULT_READER_LOGIN = ("admin", "change")


@dataclass
class Device:
    """A discovered network device (reader or printer)."""

    ip: str
    kind: DeviceKind = DeviceKind.UNKNOWN
    model: str = ""
    name: str = ""
    hostname: str = ""
    mac: str = ""
    # Primary control port: LLRP for readers, raw ZPL (9100) for printers.
    port: Optional[int] = None
    # Map of human label -> open port, e.g. {"LLRP": 5084, "Web UI": 443}.
    services: Dict[str, int] = field(default_factory=dict)
    # Tri-state: True/False when we can tell, None when unknown.
    rfid_capable: Optional[bool] = None
    # How we found it: "mdns", "llrp-probe", "udp-broadcast", "snmp", "arp".
    source: str = ""
    firmware: str = ""
    extra: Dict[str, str] = field(default_factory=dict)

    @property
    def label(self) -> str:
        """Short, human-friendly identifier for lists and dropdowns."""
        bits = [b for b in (self.model or self.name, self.hostname) if b]
        head = bits[0] if bits else self.kind.value.title()
        return f"{head} @ {self.ip}"

    def connect_hint(self) -> str:
        """A ready-to-paste instruction on how to connect to this device."""
        if self.kind is DeviceKind.READER:
            port = self.port or PORT_LLRP
            return (
                f"LLRP over TCP: connect an LLRP client to {self.ip}:{port} "
                f"(e.g. `sllurp inventory {self.ip} -p {port}`). "
                f"Web admin: https://{self.ip}/"
            )
        if self.kind is DeviceKind.PRINTER:
            port = self.port or PORT_RAW_PRINT
            return (
                f"Raw ZPL over TCP: stream ZPL to {self.ip}:{port}. "
                f"Web admin: http://{self.ip}/ . "
                f"SNMP status on udp/{PORT_SNMP}."
            )
        return f"Open http://{self.ip}/ to inspect this device."


@dataclass
class TagRead:
    """A single tag observation.

    Field names mirror the LLRP ``TagReportData`` parameters that Zebra FX
    readers emit so a real reader and the simulator are interchangeable.
    """

    epc: str
    rssi: float                       # PeakRSSI in dBm (typically -30..-80)
    antenna: int = 1                  # AntennaID
    phase: Optional[float] = None     # RF phase angle in radians (0..2*pi)
    channel: Optional[int] = None     # ChannelIndex
    timestamp: float = 0.0            # epoch seconds (LastSeenTimestampUTC)
    seen_count: int = 1               # TagSeenCount within the report window
    doppler: Optional[float] = None   # Doppler frequency in Hz, if reported
    reader: str = ""                  # source reader ip/host

    def as_row(self) -> Dict[str, object]:
        """Flatten to a dict suitable for a pandas DataFrame row."""
        return {
            "epc": self.epc,
            "antenna": self.antenna,
            "rssi": self.rssi,
            "phase": self.phase,
            "channel": self.channel,
            "timestamp": self.timestamp,
            "seen_count": self.seen_count,
            "doppler": self.doppler,
            "reader": self.reader,
        }


@dataclass
class AntennaConfig:
    """Physical placement + calibration of a single reader antenna.

    Positions are in metres on the 2D field used by the localization tool.
    """

    antenna_id: int
    x: float = 0.0
    y: float = 0.0
    # Bearing the antenna faces, degrees, 0 = +x axis, CCW positive.
    heading_deg: float = 0.0
    tx_power_dbm: float = 30.0
    # Calibration anchors for the log-distance model.
    rssi_ref: float = -45.0   # RSSI measured at the reference distance
    d_ref: float = 1.0        # reference distance in metres
    path_loss_n: float = 2.2  # environment path-loss exponent

    @property
    def position(self) -> "tuple[float, float]":
        return (self.x, self.y)


def reads_to_rows(reads: List[TagRead]) -> List[Dict[str, object]]:
    """Convenience for building a DataFrame from many reads."""
    return [r.as_row() for r in reads]
