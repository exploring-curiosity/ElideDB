"""Minimal SFI reader for the Python sidecar.

The sidecar consumes StreetDex's own frame index to pread single JPEG packets
out of multi-GB captures — the same byte-range discipline as the C++ core,
with zero video dependencies (MJPEG packets are complete JPEGs; PIL decodes
them straight from bytes). Layout: docs/FORMAT.md §2.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

FRAME_DTYPE = np.dtype([
    ("pts_ns", "<i8"), ("dts_ns", "<i8"), ("byte_offset", "<u8"),
    ("packet_size", "<u4"), ("gop_id", "<u4"), ("flags", "<u4"),
    ("pad", "<u4"),
])
GOP_DTYPE = np.dtype([
    ("gop_id", "<u4"), ("frame_count", "<u4"), ("start_byte", "<u8"),
    ("end_byte", "<u8"), ("first_pts_ns", "<i8"), ("last_pts_ns", "<i8"),
    ("first_frame_row", "<u4"), ("pad", "<u4"),
])


@dataclass
class Sfi:
    source_path: str
    width: int
    height: int
    first_pts_ns: int
    last_pts_ns: int
    frames: np.ndarray  # FRAME_DTYPE, sorted by pts_ns


def read_sfi(path: str) -> Sfi:
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != b"SFI1" or data[-4:] != b"SFI1":
        raise ValueError(f"not a complete SFI file: {path}")
    (codec, w, h, tbn, tbd, _pad, first_pts, last_pts, _clk, frame_count,
     gop_count, _src_size) = struct.unpack_from("<6I3q3Q", data, 8)
    (path_len,) = struct.unpack_from("<H", data, 80)
    source = data[82:82 + path_len].decode()
    gop_off, frame_off = struct.unpack_from("<QQ", data, len(data) - 20)
    frames = np.frombuffer(data, dtype=FRAME_DTYPE, count=frame_count,
                           offset=frame_off)
    return Sfi(source, w, h, first_pts, last_pts, frames)


def read_jpeg_packet(sfi: Sfi, row: int) -> bytes:
    """Read exactly one frame's packet bytes from the source video."""
    fr = sfi.frames[row]
    with open(sfi.source_path, "rb") as f:
        f.seek(int(fr["byte_offset"]))
        return f.read(int(fr["packet_size"]))
