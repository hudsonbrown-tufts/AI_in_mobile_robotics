"""Minimal DSU ("cemuhook") motion server, so Dolphin sees us as a controller.

Dolphin's DSU Client (Controllers > Alternate Input Sources) talks this UDP
protocol to motion servers like DS4Windows/BetterJoy. We pretend to be one
controller in slot 0 with buttons, two sticks and a full 6-axis IMU:

    accel in g, gyro in deg/s, in the DSU (DualShock-style) axis convention
    -- see wiimote_to_dsu() in hub.py for the conversion from Wii Remote axes.

Protocol summary (all little-endian):
    header  = magic(4) version u16 length u16 crc32 u32 id u32   (16 bytes)
              length = bytes after the header; crc32 over the packet with the crc field zeroed
    payload = message type u32 + data
        0x100000 version        -> reply: u16 1001
        0x100001 ports info     <- u32 count + slot bytes   -> one reply per slot
        0x100002 pad data       <- subscribe; server then streams pad data to that client
"""

import socket
import struct
import threading
import time
import zlib
from dataclasses import dataclass, field

PORT = 26760
PROTOCOL_VERSION = 1001
MSG_VERSION = 0x100000
MSG_PORTS = 0x100001
MSG_PAD_DATA = 0x100002
SEND_HZ = 125
CLIENT_TIMEOUT_S = 5.0          # clients re-subscribe every second or so
MAC = bytes([0x4C, 0x45, 0x47, 0x4F, 0x00, 0x01])   # "LEGO" + 0001, just needs to be stable

# buttons1 bits
DPAD_LEFT, DPAD_DOWN, DPAD_RIGHT, DPAD_UP = 0x80, 0x40, 0x20, 0x10
OPTIONS, R3, L3, SHARE = 0x08, 0x04, 0x02, 0x01
# buttons2 bits
TRIANGLE, CIRCLE, CROSS, SQUARE = 0x80, 0x40, 0x20, 0x10
R1, L1, R2, L2 = 0x08, 0x04, 0x02, 0x01

# Wii Remote button -> (byte, bit) on the DSU "DualShock". The Dolphin profile in
# profiles/ binds these back, e.g. Wii A <- `Cross`.
WII_BUTTONS = {
    "A": (2, CROSS), "B": (2, CIRCLE), "1": (2, SQUARE), "2": (2, TRIANGLE),
    "+": (1, OPTIONS), "-": (1, SHARE),
    "UP": (1, DPAD_UP), "DOWN": (1, DPAD_DOWN), "LEFT": (1, DPAD_LEFT), "RIGHT": (1, DPAD_RIGHT),
    "HOME": ("home", 1),
}


@dataclass
class PadState:
    buttons: set = field(default_factory=set)     # names from WII_BUTTONS
    left_stick: tuple = (0.0, 0.0)               # -1..1, +y = up
    right_stick: tuple = (0.0, 0.0)              # used for the Wii pointer
    accel: tuple = (0.0, 0.0, 0.0)               # DSU axes, g
    gyro: tuple = (0.0, 0.0, 0.0)                # DSU (pitch, yaw, roll), deg/s
    motion_t: float = 0.0                        # time.monotonic() of the IMU sample


def _stick_byte(v):
    return max(0, min(255, int(round(128 + 127 * v))))


class DSUServer:
    def __init__(self, host="127.0.0.1", port=PORT):
        self.host, self.port = host, port
        self.state = PadState()
        self.server_id = int(time.time()) & 0xFFFFFFFF
        self.clients = {}          # addr -> last subscribe time
        self.packets_sent = 0
        self.status = "starting..."
        self._sock = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

    # ---------------- public ----------------

    def start(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((self.host, self.port))
        self._sock.settimeout(0.2)
        self.status = f"listening on {self.host}:{self.port}"
        threading.Thread(target=self._recv_loop, daemon=True).start()
        threading.Thread(target=self._send_loop, daemon=True).start()
        return self

    def set_state(self, state):
        self.state = state

    @property
    def active_clients(self):
        now = time.monotonic()
        return [a for a, t in list(self.clients.items()) if now - t < CLIENT_TIMEOUT_S]

    def close(self):
        self._stop.set()

    # ---------------- packets ----------------

    def _packet(self, msg_type, data):
        body = struct.pack("<I", msg_type) + data
        header = b"DSUS" + struct.pack("<HHII", PROTOCOL_VERSION, len(body), 0, self.server_id)
        packet = bytearray(header + body)
        struct.pack_into("<I", packet, 8, zlib.crc32(packet) & 0xFFFFFFFF)
        return bytes(packet)

    @staticmethod
    def _slot_info(slot):
        connected = slot == 0
        # slot, state (2 = connected), model (2 = full gyro), connection (2 = bluetooth), mac, battery (5 = full)
        return struct.pack("<BBBB6sB", slot, 2 if connected else 0, 2 if connected else 0,
                           2 if connected else 0, MAC if connected else bytes(6), 5 if connected else 0)

    def _pad_data(self):
        s = self.state
        b = {1: 0, 2: 0, "home": 0}
        for name in s.buttons:
            byte, bit = WII_BUTTONS[name]
            b[byte] |= bit
        analog = lambda name: 255 if name in s.buttons else 0
        self.packets_sent += 1
        data = self._slot_info(0)
        data += struct.pack("<BI", 1, self.packets_sent & 0xFFFFFFFF)
        data += struct.pack("<BBBB", b[1], b[2], b["home"], 0)
        data += struct.pack("<BBBB", _stick_byte(s.left_stick[0]), _stick_byte(s.left_stick[1]),
                            _stick_byte(s.right_stick[0]), _stick_byte(s.right_stick[1]))
        data += struct.pack("<BBBB", analog("LEFT"), analog("DOWN"), analog("RIGHT"), analog("UP"))
        data += struct.pack("<BBBB", analog("2"), analog("B"), analog("A"), analog("1"))
        data += struct.pack("<BBBB", 0, 0, 0, 0)                 # R1, L1, R2, L2 analog
        data += struct.pack("<BBHH", 0, 0, 0, 0) * 2              # two touch points (unused)
        data += struct.pack("<Q", int(s.motion_t * 1_000_000) & 0xFFFFFFFFFFFFFFFF)
        data += struct.pack("<fff", *s.accel)
        data += struct.pack("<fff", *s.gyro)
        return self._packet(MSG_PAD_DATA, data)

    # ---------------- threads ----------------

    def _recv_loop(self):
        while not self._stop.is_set():
            try:
                packet, addr = self._sock.recvfrom(1024)
            except socket.timeout:
                continue
            except OSError:
                continue   # Windows raises this when a client goes away (ICMP port unreachable)
            if len(packet) < 20 or packet[:4] != b"DSUC":
                continue
            (msg_type,) = struct.unpack_from("<I", packet, 16)
            try:
                if msg_type == MSG_VERSION:
                    self._sock.sendto(self._packet(MSG_VERSION, struct.pack("<H", PROTOCOL_VERSION)), addr)
                elif msg_type == MSG_PORTS:
                    (count,) = struct.unpack_from("<i", packet, 20)
                    for i in range(max(0, min(count, 4))):
                        slot = packet[24 + i]
                        self._sock.sendto(self._packet(MSG_PORTS, self._slot_info(slot) + b"\x00"), addr)
                elif msg_type == MSG_PAD_DATA:
                    if addr not in self.clients:
                        print(f"[dsu] Dolphin connected from {addr[0]}:{addr[1]}")
                    self.clients[addr] = time.monotonic()
            except OSError:
                pass

    def _send_loop(self):
        period = 1.0 / SEND_HZ
        while not self._stop.is_set():
            start = time.monotonic()
            clients = self.active_clients
            if clients:
                packet = self._pad_data()
                for addr in clients:
                    try:
                        self._sock.sendto(packet, addr)
                    except OSError:
                        pass
            time.sleep(max(0.0, period - (time.monotonic() - start)))
