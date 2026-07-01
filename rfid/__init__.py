"""Zebra RFID testing & calibration toolkit.

A small, dependency-light library that powers four tools:

1. :mod:`rfid.discovery`   - find readers/printers on the LAN and show how to connect.
2. :mod:`rfid.ranging`     - turn RSSI into an approximate distance / 2D position.
3. :mod:`rfid.denoise`     - smooth noisy RSSI/phase streams (Kalman + learned filter).
4. :mod:`rfid.obstruction` - decide whether the reader<->tag path is blocked.

Everything works against real hardware over LLRP (:mod:`rfid.llrp_client`) and,
when no hardware is present, against a physically-motivated
:mod:`rfid.simulator` so the tools stay fully demonstrable.
"""

from .models import (
    AntennaConfig,
    Device,
    DeviceKind,
    TagRead,
    PORT_LLRP,
    PORT_RAW_PRINT,
    PORT_PRINTER_DISCOVERY,
    PORT_SNMP,
)

__all__ = [
    "AntennaConfig",
    "Device",
    "DeviceKind",
    "TagRead",
    "PORT_LLRP",
    "PORT_RAW_PRINT",
    "PORT_PRINTER_DISCOVERY",
    "PORT_SNMP",
]

__version__ = "0.1.0"
