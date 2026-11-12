"""Inspect snapshot shapes during compilation without importing NumPy."""

import ast
import math
from pathlib import Path


_SIMPLE = {"b": 1, "i": None, "u": None, "f": None, "c": None, "U": 4, "S": 1, "V": 1}


def _itemsize(descr: object) -> int | None:
    if not isinstance(descr, str) or len(descr) < 3 or descr[1] not in _SIMPLE:
        return None
    try:
        count = int(descr[2:])
    except ValueError:
        return None
    unit = _SIMPLE[descr[1]]
    return count * (unit or 1) if descr[1] != "b" else 1


def read_npy_shape(path: Path) -> tuple[int, ...]:
    """Read an NPY header and check the length of a plain array's payload."""
    with path.open("rb") as handle:
        if handle.read(6) != b"\x93NUMPY":
            raise ValueError("not a .npy file")
        version = handle.read(2)
        if version not in {b"\x01\x00", b"\x02\x00", b"\x03\x00"}:
            raise ValueError("unsupported .npy version")
        width = 2 if version[0] == 1 else 4
        raw = handle.read(width)
        if len(raw) != width:
            raise ValueError("truncated .npy header")
        length = int.from_bytes(raw, "little")
        if length > 10_000:
            raise ValueError("NPY header exceeds the supported size")
        header = handle.read(length)
        if len(header) != length:
            raise ValueError("truncated .npy header")
        try:
            meta = ast.literal_eval(header.decode("latin1" if version[0] < 3 else "utf8"))
        except (SyntaxError, ValueError) as exc:
            raise ValueError("invalid .npy header") from exc
        if not isinstance(meta, dict) or set(meta) != {"descr", "fortran_order", "shape"}:
            raise ValueError("invalid .npy header keys")
        if not isinstance(meta["fortran_order"], bool):
            raise ValueError("invalid .npy storage order")
        shape = meta["shape"]
        if not isinstance(shape, tuple) or not all(isinstance(n, int) and n >= 0 for n in shape):
            raise ValueError("invalid .npy shape")
        descr = meta["descr"]
        if isinstance(descr, str) and "O" in descr:
            raise ValueError("object arrays cannot be memory-mapped")
        itemsize = _itemsize(descr)
        count = math.prod(shape)
        if (
            itemsize is not None
            and count
            and path.stat().st_size < handle.tell() + itemsize * count
        ):
            raise ValueError("truncated .npy payload")
    return tuple(int(size) for size in shape)


def validate_snapshot_header(
    directory: str | Path, step: int, grid_shape: tuple[int, ...], component_count: int
) -> None:
    """Check an existing initial snapshot against either compact-Q layout."""
    path = Path(directory).expanduser().resolve() / f"Q_{step}.npy"
    if not path.exists():
        return
    try:
        observed = read_npy_shape(path)
    except (OSError, ValueError) as exc:
        raise ValueError("Q snapshot header could not be read") from exc
    component_last = (*grid_shape, component_count)
    component_first = (component_count, *grid_shape)
    if observed not in {component_last, component_first}:
        raise ValueError(
            "Q snapshot shape does not match the declared geometry: "
            f"observed={observed!r}, expected one of {component_last!r} or {component_first!r}"
        )
