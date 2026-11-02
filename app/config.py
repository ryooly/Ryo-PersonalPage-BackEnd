from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

ENCODING = "pcm_s16le"
SAMPLE_WIDTH = 2


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


def _env_optional_str(name: str) -> str | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    return raw.strip()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {raw!r}")


@dataclass(frozen=True, slots=True)
class AudioSettings:
    sample_rate: int = 16_000
    channels: int = 1
    min_seconds: float = 0.25
    max_seconds: float = 30.0
    max_chunk_bytes: int = 1_048_576

    def __post_init__(self) -> None:
        if self.sample_rate < 8_000:
            raise ValueError(f"sample_rate must be at least 8000, got {self.sample_rate}")
        if self.channels not in (1, 2):
            raise ValueError(f"channels must be 1 or 2, got {self.channels}")
        if self.min_seconds <= 0:
            raise ValueError(f"min_seconds must be positive, got {self.min_seconds}")
        if self.max_seconds < self.min_seconds:
            raise ValueError("max_seconds must be greater than or equal to min_seconds")
        if self.max_chunk_bytes < self.frame_bytes:
            raise ValueError("max_chunk_bytes is smaller than a single audio frame")

    @classmethod
    def from_env(cls) -> "AudioSettings":
        return cls(
            sample_rate=_env_int("AUDIO_SAMPLE_RATE", 16_000),
            channels=_env_int("AUDIO_CHANNELS", 1),
            min_seconds=_env_float("AUDIO_MIN_SECONDS", 0.25),
            max_seconds=_env_float("AUDIO_MAX_SECONDS", 30.0),
            max_chunk_bytes=_env_int("AUDIO_MAX_CHUNK_BYTES", 1_048_576),
        )

    @property
    def frame_bytes(self) -> int:
        return SAMPLE_WIDTH * self.channels

    @property
    def bytes_per_second(self) -> int:
        return self.frame_bytes * self.sample_rate

    @property
    def min_bytes(self) -> int:
        return int(self.bytes_per_second * self.min_seconds)

    @property
    def max_bytes(self) -> int:
        return int(self.bytes_per_second * self.max_seconds)

    def wire_format(self) -> dict[str, Any]:
        return {
            "encoding": ENCODING,
            "sampleRate": self.sample_rate,
            "channels": self.channels,
        }


@dataclass(frozen=True, slots=True)
class WhisperSettings:
    model_size: str = "base"
    device: str = "auto"
    compute_type: str = "default"
    language: str | None = "en"
    beam_size: int = 1
    vad_filter: bool = True
    download_root: str | None = None

    def __post_init__(self) -> None:
        if not self.model_size.strip():
            raise ValueError("model_size must not be empty")
        if self.beam_size < 1:
            raise ValueError(f"beam_size must be at least 1, got {self.beam_size}")

    @classmethod
    def from_env(cls) -> "WhisperSettings":
        return cls(
            model_size=os.environ.get("WHISPER_MODEL", "base").strip(),
            device=os.environ.get("WHISPER_DEVICE", "auto").strip(),
            compute_type=os.environ.get("WHISPER_COMPUTE_TYPE", "default").strip(),
            language=_env_optional_str("WHISPER_LANGUAGE") or "en",
            beam_size=_env_int("WHISPER_BEAM_SIZE", 1),
            vad_filter=_env_bool("WHISPER_VAD_FILTER", True),
            download_root=_env_optional_str("WHISPER_DOWNLOAD_ROOT"),
        )

    def describe(self) -> dict[str, Any]:
        return {
            "model": self.model_size,
            "device": self.device,
            "computeType": self.compute_type,
            "language": self.language,
            "beamSize": self.beam_size,
            "vadFilter": self.vad_filter,
        }
