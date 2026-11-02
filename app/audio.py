from __future__ import annotations

import sys
import time
from array import array
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .config import SAMPLE_WIDTH, AudioSettings

if TYPE_CHECKING:
    import numpy as np

_LITTLE_ENDIAN = sys.byteorder == "little"


@dataclass(slots=True)
class AudioBuffer:
    settings: AudioSettings
    _pcm: bytearray = field(default_factory=bytearray, repr=False)
    _carry: bytearray = field(default_factory=bytearray, repr=False)
    chunks: int = 0
    created_at: float = field(default_factory=time.monotonic)
    closed_at: float | None = None

    def append(self, chunk: bytes) -> int:
        if not chunk:
            return self.byte_count

        pending = bytes(self._carry) + chunk if self._carry else chunk
        usable = len(pending) - (len(pending) % self.settings.frame_bytes)

        if usable == 0:
            self._carry = bytearray(pending)
            return self.byte_count

        self._pcm += pending[:usable]
        self._carry = bytearray(pending[usable:])
        self.chunks += 1
        return self.byte_count

    def finish(self) -> "AudioBuffer":
        self.closed_at = time.monotonic()
        return self

    def clear(self) -> None:
        self._pcm.clear()
        self._carry.clear()
        self.chunks = 0
        self.created_at = time.monotonic()
        self.closed_at = None

    @property
    def byte_count(self) -> int:
        return len(self._pcm)

    @property
    def sample_count(self) -> int:
        return len(self._pcm) // SAMPLE_WIDTH

    @property
    def frame_count(self) -> int:
        return self.sample_count // self.settings.channels

    @property
    def duration(self) -> float:
        return len(self._pcm) / self.settings.bytes_per_second

    @property
    def is_empty(self) -> bool:
        return not self._pcm

    @property
    def trailing_bytes(self) -> int:
        return len(self._carry)

    def __len__(self) -> int:
        return len(self._pcm)

    def to_bytes(self) -> bytes:
        return bytes(self._pcm)

    def to_int16(self) -> array:
        samples = array("h")
        samples.frombytes(bytes(self._pcm))
        if not _LITTLE_ENDIAN:
            samples.byteswap()
        return samples

    def to_mono_int16(self) -> array:
        samples = self.to_int16()
        channels = self.settings.channels
        if channels == 1:
            return samples
        mono = array("h", bytes(len(samples) // channels * SAMPLE_WIDTH))
        for index in range(0, len(samples) - channels + 1, channels):
            mono[index // channels] = int(sum(samples[index : index + channels]) / channels)
        return mono

    def to_numpy(self) -> "np.ndarray":
        try:
            import numpy as np
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "numpy is required for AudioBuffer.to_numpy(); run `pip install numpy`"
            ) from exc

        samples = np.frombuffer(bytes(self._pcm), dtype="<i2").astype(np.float32)
        channels = self.settings.channels
        if channels > 1:
            samples = samples.reshape(-1, channels).mean(axis=1, dtype=np.float32)
        return samples / 32768.0

    def describe(self) -> dict[str, object]:
        return {
            "duration": round(self.duration, 3),
            "bytes": self.byte_count,
            "chunks": self.chunks,
            "frameCount": self.frame_count,
            "format": self.settings.wire_format(),
        }
