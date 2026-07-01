"""Tool 1 - Select device: find Zebra readers & printers on the LAN.

Discovery layers (each is best-effort and degrades gracefully):

* **TCP port sweep** (stdlib only) - connect-scan the local subnet for the
  tell-tale ports: LLRP 5084/5085 (readers) and raw-ZPL 9100/6101 (printers).
  Works in any environment with no extra dependencies.
* **UDP/4201 broadcast** (stdlib only) - Zebra's native printer discovery.
  Sends the exact probe ``2E 2C 3A 01 00 00`` and parses the ``3A 2C 2E``
  response, mirroring the Link-OS ``NetworkDiscoverer.localBroadcast``.
* **WS-Discovery** (optional ``WSDiscovery``) - the SOAP-over-UDP multicast
  that Zebra fixed readers (FXR90/FX9600/ATR7000) answer, i.e. what
  ``123RFID Desktop -> Find Readers`` uses.
* **mDNS/Bonjour** (optional ``zeroconf``) - browse ``_pdl-datastream._tcp``
  (printers) and generic types, filtering by Zebra hostname prefixes.
* **SNMP** (optional ``pysnmp``) - enrich model/serial via the Zebra
  enterprise OID root ``1.3.6.1.4.1.10642``.

Every discovered host is returned as a :class:`~rfid.models.Device` with a
ready-to-use ``connect_hint()``.  When a printer is found we additionally probe
its RFID capability with an SGD ``getvar "rfid.tag.type"``.

Ports, probe bytes, OIDs and service types here are taken from the Zebra FX/FXR
Integration Guides and Link-OS SDK docs (see module-level references in the PR).
"""

from __future__ import annotations

import ipaddress
import socket
import struct
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple

from .models import (
    DEFAULT_READER_LOGIN,
    Device,
    DeviceKind,
    PORT_LLRP,
    PORT_LLRP_TLS,
    PORT_PRINTER_DISCOVERY,
    PORT_RAW_PRINT,
    PORT_SNMP,
    READER_HOST_PREFIXES,
    WS_DISCOVERY_GROUP,
)

# Ports that identify each device kind, with a friendly label.
READER_PORTS: Dict[int, str] = {PORT_LLRP: "LLRP", PORT_LLRP_TLS: "LLRP/TLS"}
PRINTER_PORTS: Dict[int, str] = {PORT_RAW_PRINT: "Raw ZPL (9100)", 6101: "ZQ raw (6101)"}
COMMON_PORTS: Dict[int, str] = {22: "SSH", 80: "Web UI", 443: "HTTPS", 9200: "Status/JSON"}

# Zebra native UDP discovery request; printers reply with a 3A 2C 2E packet.
_UDP_PROBE = bytes((0x2E, 0x2C, 0x3A, 0x01, 0x00, 0x00))
_UDP_RESP_MAGIC = bytes((0x3A, 0x2C, 0x2E))

# Printer model tokens we recognise in discovery/hostname strings.
_PRINTER_TOKENS = ("ZT", "ZD", "ZQ", "ZR", "ZE", "R110", "R110XI", "ZEBRANET", "ZTC")


# --------------------------------------------------------------------------- #
# Local network topology
# --------------------------------------------------------------------------- #
def local_ipv4_networks() -> List[ipaddress.IPv4Network]:
    """Best-effort list of the /24 subnets this host sits on.

    We avoid hard dependencies (netifaces): use the "connect a UDP socket to a
    public IP" trick to learn our primary address, then assume a /24.  Falls
    back to loopback if the host is offline.
    """
    nets: List[ipaddress.IPv4Network] = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
        finally:
            s.close()
        nets.append(ipaddress.ip_network(f"{ip}/24", strict=False))
    except OSError:
        pass
    # Also include anything resolvable for our hostname.
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127."):
                net = ipaddress.ip_network(f"{ip}/24", strict=False)
                if net not in nets:
                    nets.append(net)
    except OSError:
        pass
    return nets


def _broadcast_addresses() -> List[str]:
    addrs = ["255.255.255.255"]
    for net in local_ipv4_networks():
        b = str(net.broadcast_address)
        if b not in addrs:
            addrs.append(b)
    return addrs


# --------------------------------------------------------------------------- #
# Low-level probes
# --------------------------------------------------------------------------- #
def tcp_open(ip: str, port: int, timeout: float = 0.4) -> bool:
    """True if a TCP connection to ip:port succeeds within ``timeout``."""
    try:
        with socket.create_connection((ip, port), timeout=timeout):
            return True
    except OSError:
        return False


def reverse_dns(ip: str) -> str:
    try:
        return socket.gethostbyaddr(ip)[0]
    except OSError:
        return ""


def _classify_hostname(host: str) -> Tuple[Optional[DeviceKind], str]:
    """Guess (kind, model) from a hostname such as 'FX9600ABCDEF'."""
    up = host.upper()
    for pref in READER_HOST_PREFIXES:
        if up.startswith(pref):
            return DeviceKind.READER, pref
    for tok in _PRINTER_TOKENS:
        if up.startswith(tok):
            return DeviceKind.PRINTER, tok
    return None, ""


def probe_printer_rfid(ip: str, port: int = PORT_RAW_PRINT, timeout: float = 1.0) -> Tuple[Optional[bool], str]:
    """Ask a printer if it has an RFID encoder via SGD ``getvar rfid.tag.type``.

    Returns ``(is_rfid, product_name)``.  ``is_rfid`` is ``None`` if we could
    not talk to the printer at all.
    """
    try:
        with socket.create_connection((ip, port), timeout=timeout) as s:
            s.settimeout(timeout)
            s.sendall(b'! U1 getvar "device.product_name"\r\n')
            product = s.recv(256).decode("latin-1", "ignore").strip().strip('"')
            s.sendall(b'! U1 getvar "rfid.tag.type"\r\n')
            tag = s.recv(256).decode("latin-1", "ignore").strip().strip('"')
    except OSError:
        return None, ""
    # A non-RFID printer answers '?' or empty to the rfid getvar.
    is_rfid = bool(tag) and tag not in {"?", "", '""'}
    # An 'R' in the model (e.g. ZT411R) is a corroborating signal.
    if not is_rfid and product and product.upper().rstrip("-").endswith("R"):
        is_rfid = True
    return is_rfid, product


# --------------------------------------------------------------------------- #
# UDP/4201 broadcast printer discovery
# --------------------------------------------------------------------------- #
def _parse_udp_response(payload: bytes, ip: str) -> Optional[Device]:
    if not payload.startswith(_UDP_RESP_MAGIC):
        return None
    # Extract printable ASCII tokens (fields are space/null padded).
    tokens: List[str] = []
    cur = bytearray()
    for b in payload[4:]:
        if 32 <= b < 127:
            cur.append(b)
        else:
            if len(cur) >= 3:
                tokens.append(cur.decode("ascii", "ignore").strip())
            cur.clear()
    if len(cur) >= 3:
        tokens.append(cur.decode("ascii", "ignore").strip())

    model = ""
    for t in tokens:
        if any(t.upper().startswith(tok) for tok in _PRINTER_TOKENS):
            model = t
            break
    dev = Device(
        ip=ip,
        kind=DeviceKind.PRINTER,
        model=model,
        name=tokens[0] if tokens else "",
        source="udp-broadcast",
    )
    dev.extra["discovery_tokens"] = " | ".join(tokens[:8])
    return dev


def discover_printers_udp(timeout: float = 2.0) -> List[Device]:
    """Broadcast the Zebra UDP/4201 probe and collect printer replies."""
    found: Dict[str, Device] = {}
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        s.bind(("", 0))
        s.settimeout(0.6)
    except OSError:
        return []
    try:
        for addr in _broadcast_addresses():
            for _ in range(3):  # SDK sends the probe 3x
                try:
                    s.sendto(_UDP_PROBE, (addr, PORT_PRINTER_DISCOVERY))
                except OSError:
                    break
        # Collect replies until the timeout budget is spent.
        deadline = _now() + timeout
        while _now() < deadline:
            try:
                payload, src = s.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                break
            dev = _parse_udp_response(payload, src[0])
            if dev and src[0] not in found:
                found[src[0]] = dev
    finally:
        s.close()
    return list(found.values())


def _now() -> float:
    # time.monotonic isolated so the module has one clock source.
    import time

    return time.monotonic()


# --------------------------------------------------------------------------- #
# TCP port sweep (dependency-free workhorse)
# --------------------------------------------------------------------------- #
def _scan_host(ip: str, timeout: float) -> Optional[Device]:
    services: Dict[str, int] = {}
    for port, label in {**READER_PORTS, **PRINTER_PORTS, **COMMON_PORTS}.items():
        if tcp_open(ip, port, timeout):
            services[label] = port
    if not services:
        return None

    host = reverse_dns(ip)
    kind, model = _classify_hostname(host) if host else (None, "")

    # Infer kind from open ports if the hostname was unhelpful.
    if kind is None:
        if any(p in services.values() for p in READER_PORTS):
            kind = DeviceKind.READER
        elif any(p in services.values() for p in PRINTER_PORTS):
            kind = DeviceKind.PRINTER
        else:
            kind = DeviceKind.UNKNOWN

    dev = Device(
        ip=ip,
        kind=kind,
        model=model,
        hostname=host,
        services=services,
        source="tcp-sweep",
    )
    if kind is DeviceKind.READER:
        dev.port = services.get("LLRP", services.get("LLRP/TLS", PORT_LLRP))
    elif kind is DeviceKind.PRINTER:
        dev.port = services.get("Raw ZPL (9100)", services.get("ZQ raw (6101)", PORT_RAW_PRINT))
        is_rfid, product = probe_printer_rfid(ip, dev.port or PORT_RAW_PRINT)
        dev.rfid_capable = is_rfid
        if product:
            dev.model = dev.model or product
    return dev


def sweep_subnet(
    network: Optional[ipaddress.IPv4Network] = None,
    timeout: float = 0.4,
    max_hosts: int = 256,
    workers: int = 64,
) -> List[Device]:
    """Concurrent TCP connect-scan of a subnet for Zebra ports."""
    nets = [network] if network else local_ipv4_networks()
    hosts: List[str] = []
    for net in nets:
        for h in net.hosts():
            hosts.append(str(h))
            if len(hosts) >= max_hosts:
                break
    found: List[Device] = []
    if not hosts:
        return found
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_scan_host, ip, timeout): ip for ip in hosts}
        for fut in as_completed(futs):
            dev = fut.result()
            if dev:
                found.append(dev)
    return found


# --------------------------------------------------------------------------- #
# Optional enrichment layers (WS-Discovery / mDNS / SNMP)
# --------------------------------------------------------------------------- #
def discover_readers_ws(timeout: float = 4.0) -> List[Device]:
    """WS-Discovery probe for fixed readers (needs the ``WSDiscovery`` pkg)."""
    try:
        from wsdiscovery.discovery import ThreadedWSDiscovery  # type: ignore
    except Exception:
        return []
    devices: List[Device] = []
    wsd = ThreadedWSDiscovery()
    try:
        wsd.start()
        for svc in wsd.searchServices(timeout=timeout):
            xaddrs = list(svc.getXAddrs())
            ip = ""
            for xa in xaddrs:
                # xaddr looks like http://192.168.1.50:8080/...
                try:
                    ip = xa.split("//", 1)[1].split(":", 1)[0].split("/", 1)[0]
                except Exception:
                    continue
                if ip:
                    break
            if not ip:
                continue
            dev = Device(ip=ip, kind=DeviceKind.READER, source="ws-discovery")
            dev.extra["scopes"] = " ".join(str(x) for x in svc.getScopes())
            dev.extra["xaddrs"] = " ".join(xaddrs)
            devices.append(dev)
    except Exception:
        return devices
    finally:
        try:
            wsd.stop()
        except Exception:
            pass
    return devices


def discover_mdns(timeout: float = 3.0) -> List[Device]:
    """mDNS/Bonjour browse for Zebra printers/readers (needs ``zeroconf``)."""
    try:
        from zeroconf import ServiceBrowser, Zeroconf  # type: ignore
    except Exception:
        return []
    import time

    service_types = [
        "_pdl-datastream._tcp.local.",  # raw 9100 printing (Zebra printers)
        "_printer._tcp.local.",
        "_ipp._tcp.local.",
        "_http._tcp.local.",
    ]
    results: Dict[str, Device] = {}

    class _Listener:
        def add_service(self, zc, type_, name):
            info = zc.get_service_info(type_, name, timeout=1500)
            if not info:
                return
            addrs = info.parsed_addresses() if hasattr(info, "parsed_addresses") else []
            if not addrs:
                return
            ip = addrs[0]
            props = {}
            for k, v in (info.properties or {}).items():
                try:
                    props[k.decode()] = v.decode() if isinstance(v, bytes) else str(v)
                except Exception:
                    pass
            kind, model = _classify_hostname(name)
            if "pdl-datastream" in type_ or "printer" in type_ or "ipp" in type_:
                kind = kind or DeviceKind.PRINTER
            dev = Device(
                ip=ip,
                kind=kind or DeviceKind.UNKNOWN,
                model=model or props.get("ty", ""),
                name=name,
                source="mdns",
            )
            dev.extra.update(props)
            results[ip] = dev

        def update_service(self, *a):
            pass

        def remove_service(self, *a):
            pass

    zc = Zeroconf()
    try:
        listener = _Listener()
        for st in service_types:
            ServiceBrowser(zc, st, listener)
        time.sleep(timeout)
    finally:
        zc.close()
    return list(results.values())


def snmp_identify(ip: str, community: str = "public", timeout: float = 1.0) -> Dict[str, str]:
    """Fetch model/serial via SNMP (needs ``pysnmp``).  Empty dict on failure."""
    try:
        from pysnmp.hlapi import (  # type: ignore
            CommunityData,
            ContextData,
            ObjectIdentity,
            ObjectType,
            SnmpEngine,
            UdpTransportTarget,
            getCmd,
        )
    except Exception:
        return {}
    oids = {
        "sysDescr": "1.3.6.1.2.1.1.1.0",
        "model": "1.3.6.1.4.1.10642.1.1.0",
        "serial": "1.3.6.1.4.1.10642.1.4.0",
    }
    out: Dict[str, str] = {}
    for key, oid in oids.items():
        try:
            it = getCmd(
                SnmpEngine(),
                CommunityData(community, mpModel=0),
                UdpTransportTarget((ip, PORT_SNMP), timeout=timeout, retries=0),
                ContextData(),
                ObjectType(ObjectIdentity(oid)),
            )
            err_ind, err_stat, _, binds = next(it)
            if err_ind or err_stat:
                continue
            for _, val in binds:
                out[key] = str(val)
        except Exception:
            continue
    return out


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _merge(into: Dict[str, Device], devices: Sequence[Device]) -> None:
    for d in devices:
        if d.ip in into:
            cur = into[d.ip]
            # Prefer a concrete kind/model and union the services + sources.
            if cur.kind is DeviceKind.UNKNOWN and d.kind is not DeviceKind.UNKNOWN:
                cur.kind = d.kind
            cur.model = cur.model or d.model
            cur.hostname = cur.hostname or d.hostname
            cur.services.update(d.services)
            if d.rfid_capable is not None and cur.rfid_capable is None:
                cur.rfid_capable = d.rfid_capable
            if d.source and d.source not in cur.source:
                cur.source = f"{cur.source}+{d.source}" if cur.source else d.source
            cur.extra.update(d.extra)
        else:
            into[d.ip] = d


def discover_all(
    do_sweep: bool = True,
    do_udp: bool = True,
    do_ws: bool = True,
    do_mdns: bool = True,
    do_snmp: bool = False,
    timeout: float = 3.0,
    network: Optional[str] = None,
) -> List[Device]:
    """Run every available discovery layer and return a de-duplicated list."""
    merged: Dict[str, Device] = {}
    net = ipaddress.ip_network(network, strict=False) if network else None

    if do_udp:
        _merge(merged, discover_printers_udp(timeout=timeout))
    if do_ws:
        _merge(merged, discover_readers_ws(timeout=timeout))
    if do_mdns:
        _merge(merged, discover_mdns(timeout=min(timeout, 3.0)))
    if do_sweep:
        _merge(merged, sweep_subnet(net, timeout=min(0.5, timeout / 4 or 0.4)))
    if do_snmp:
        for dev in merged.values():
            info = snmp_identify(dev.ip)
            if info:
                dev.model = dev.model or info.get("model", "")
                dev.extra.update(info)

    return sorted(merged.values(), key=lambda d: tuple(int(o) for o in d.ip.split(".") if o.isdigit()) or (0,))


# --------------------------------------------------------------------------- #
# Demo devices (used when no hardware is on the network)
# --------------------------------------------------------------------------- #
def demo_devices() -> List[Device]:
    """A realistic set of devices for demoing the UI without hardware."""
    fx = Device(
        ip="192.168.1.50",
        kind=DeviceKind.READER,
        model="FX9600",
        hostname="FX9600CD3B0D",
        services={"LLRP": PORT_LLRP, "SSH": 22, "HTTPS": 443},
        port=PORT_LLRP,
        rfid_capable=True,
        firmware="3.27.10",
        source="demo",
    )
    fx.extra["antennas"] = "4"
    atr = Device(
        ip="192.168.1.51",
        kind=DeviceKind.READER,
        model="ATR7000",
        hostname="ATR7000AA17F2",
        services={"LLRP": PORT_LLRP, "Web UI": 80},
        port=PORT_LLRP,
        rfid_capable=True,
        firmware="1.9.3",
        source="demo",
    )
    atr.extra["note"] = "RTLS beam-steering (azimuth/elevation)"
    zt = Device(
        ip="192.168.1.60",
        kind=DeviceKind.PRINTER,
        model="ZT411R",
        hostname="ZT411R7A2C10",
        services={"Raw ZPL (9100)": PORT_RAW_PRINT, "Status/JSON": 9200, "Web UI": 80},
        port=PORT_RAW_PRINT,
        rfid_capable=True,
        firmware="V93.21.10Z",
        source="demo",
    )
    zd = Device(
        ip="192.168.1.61",
        kind=DeviceKind.PRINTER,
        model="ZD621",
        hostname="ZD62100918F4",
        services={"Raw ZPL (9100)": PORT_RAW_PRINT, "Web UI": 80},
        port=PORT_RAW_PRINT,
        rfid_capable=False,
        firmware="V84.21.10Z",
        source="demo",
    )
    return [fx, atr, zt, zd]
