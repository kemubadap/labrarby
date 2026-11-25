import struct
import zlib
from pathlib import Path
import numpy as np


class BaseDataSet:
    """Core class containing only the data initialization."""

    def __init__(self, filename_or_data):
        if isinstance(filename_or_data, (str, Path)):
            self.filename = str(filename_or_data)
            try:
                if self.filename.lower().endswith(".wxd"):
                    self.data = self._load_wxd(self.filename)
                else:
                    self.data = np.loadtxt(self.filename)
            except Exception as e:
                raise FileNotFoundError(f"Error loading file '{self.filename}': {e}") from e
        elif isinstance(filename_or_data, np.ndarray):
            self.filename = None
            self.data = filename_or_data
        else:
            raise TypeError("Unsupported data type. Expected a string (filename) or numpy array.")

    @staticmethod
    def _extract_zlib_blocks(raw: bytes) -> list[bytes]:
        blocks = []
        pos = 0
        while pos < len(raw) - 2:
            idx = raw.find(b"\x78\x9c", pos)
            if idx == -1:
                break
            try:
                decomp = zlib.decompressobj()
                out = decomp.decompress(raw[idx:])
                if decomp.eof and len(out) > 16:
                    consumed = len(raw[idx:]) - len(decomp.unused_data)
                    blocks.append(out)
                    pos = idx + consumed
                    continue
            except zlib.error:
                pass
            pos = idx + 1
        return blocks

    @staticmethod
    def _find_variant_double_array(block: bytes, expected_len: int | None = None) -> list[float] | None:
        """Find a contiguous sequence of COM VT_R8 (0x05 0x00 + float64) values."""
        if len(block) >= 18:
            count = struct.unpack_from("<Q", block, 0)[0]
            if 10 <= count <= 1_000_000 and len(block) >= 8 + count * 10:
                if all(block[8 + i * 10 : 10 + i * 10] == b"\x05\x00" for i in range(min(count, 10))):
                    if expected_len is None or count == expected_len:
                        return [struct.unpack_from("<d", block, 10 + i * 10)[0] for i in range(count)]

        if expected_len is not None:
            marker = struct.pack("<Q", expected_len) + b"\x05\x00"
            idx = block.find(marker)
            if idx != -1 and len(block) >= idx + 8 + expected_len * 10:
                start = idx + 8
                return [struct.unpack_from("<d", block, start + 2 + i * 10)[0] for i in range(expected_len)]

            needed = expected_len * 10
            for start in range(0, len(block) - needed + 1):
                if block[start : start + 2] == b"\x05\x00":
                    if all(block[start + i * 10 : start + i * 10 + 2] == b"\x05\x00" for i in range(min(expected_len, 20))):
                        return [struct.unpack_from("<d", block, start + 2 + i * 10)[0] for i in range(expected_len)]
        return None

    @classmethod
    def _load_wxd(cls, filepath: str) -> np.ndarray:
        raw = Path(filepath).read_bytes()
        blocks = cls._extract_zlib_blocks(raw)

        x_data = None
        for b in reversed(blocks):
            arr = cls._find_variant_double_array(b)
            if arr is not None:
                x_data = arr
                break

        if x_data is None:
            raise ValueError("Could not locate X-axis array in .wxd file.")

        y_data = None
        for b in reversed(blocks):
            if b"D\x00a\x00t\x00a\x00L\x00i\x00s\x00t" in b[:64]:
                arr = cls._find_variant_double_array(b, expected_len=len(x_data))
                if arr is not None:
                    y_data = arr
                    break

        if y_data is None:
            raise ValueError("Could not locate Intensity array in .wxd file.")

        return np.column_stack((x_data, y_data)).astype(np.float64)