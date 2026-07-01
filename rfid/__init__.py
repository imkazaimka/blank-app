"""Zebra RFID obstruction-detection toolkit.

A dependency-light library behind a single tool: decide whether the line of
sight between a reader antenna and a tag is CLEAR or OBSTRUCTED, and tell a
blocked path apart from a tag that simply moved.

* :mod:`rfid.obstruction` - motion-invariant blockage detection (position
  residuals + Doppler), plus a fixed-zone baseline detector and an optional
  learned classifier.
* :mod:`rfid.ranging`     - RSSI<->distance and 2D localization used to
  re-estimate the tag position each window.
* :mod:`rfid.sources`     - one streaming interface over a live LLRP reader
  (:class:`~rfid.sources.LLRPTagSource`) and the simulator.
* :mod:`rfid.simulator`   - a physics-based tag-read simulator (RSSI, phase,
  Doppler, obstructions) so the tool is demonstrable without hardware.
"""

from .models import AntennaConfig, Device, DeviceKind, TagRead, PORT_LLRP

__all__ = [
    "AntennaConfig",
    "Device",
    "DeviceKind",
    "TagRead",
    "PORT_LLRP",
]

__version__ = "0.2.0"
