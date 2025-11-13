import struct
import os
import numpy as np


def fvecs_to_npy_memmap(src_path: str, out_path: str = None) -> str:
    """Convert a .fvecs file to a .npy memmap file and return the out_path.

    Two-pass: first count vectors and dim, then allocate memmap and fill rows.
    If out_path exists it is returned as-is.
    """
    if out_path is None:
        out_path = src_path + ".npy"

    if os.path.exists(out_path):
        return out_path

    # First pass: count vectors and dim
    n = 0
    dim = None
    with open(src_path, 'rb') as fh:
        while True:
            hdr = fh.read(4)
            if not hdr:
                break
            d = struct.unpack('i', hdr)[0]
            if dim is None:
                dim = d
            elif dim != d:
                raise ValueError(f"Inconsistent dims in {src_path}: {d} vs {dim}")
            # skip floats
            fh.seek(4 * d, os.SEEK_CUR)
            n += 1

    if n == 0:
        # write empty array
        np.save(out_path, np.zeros((0, 0), dtype=np.float32))
        return out_path

    # allocate memmap
    arr = np.lib.format.open_memmap(out_path, mode='w+', dtype=np.float32, shape=(n, dim))

    # second pass: fill
    i = 0
    with open(src_path, 'rb') as fh:
        while True:
            hdr = fh.read(4)
            if not hdr:
                break
            d = struct.unpack('i', hdr)[0]
            buf = fh.read(4 * d)
            if len(buf) != 4 * d:
                raise EOFError(f"Unexpected EOF while reading {src_path}")
            row = np.frombuffer(buf, dtype=np.float32).copy()
            arr[i] = row
            i += 1

    # flush to disk
    del arr
    return out_path


def bvecs_to_npy_memmap(src_path: str, out_path: str = None) -> str:
    """Convert a .bvecs file (uint8 vectors) to .npy memmap as float32."""
    if out_path is None:
        out_path = src_path + ".npy"

    if os.path.exists(out_path):
        return out_path

    n = 0
    dim = None
    with open(src_path, 'rb') as fh:
        while True:
            hdr = fh.read(4)
            if not hdr:
                break
            d = struct.unpack('i', hdr)[0]
            if dim is None:
                dim = d
            elif dim != d:
                raise ValueError(f"Inconsistent dims in {src_path}: {d} vs {dim}")
            fh.seek(d, os.SEEK_CUR)
            n += 1

    if n == 0:
        np.save(out_path, np.zeros((0, 0), dtype=np.float32))
        return out_path

    arr = np.lib.format.open_memmap(out_path, mode='w+', dtype=np.float32, shape=(n, dim))
    i = 0
    with open(src_path, 'rb') as fh:
        while True:
            hdr = fh.read(4)
            if not hdr:
                break
            d = struct.unpack('i', hdr)[0]
            buf = fh.read(d)
            if len(buf) != d:
                raise EOFError(f"Unexpected EOF while reading {src_path}")
            row = np.frombuffer(buf, dtype=np.uint8).astype(np.float32).copy()
            arr[i] = row
            i += 1

    del arr
    return out_path
