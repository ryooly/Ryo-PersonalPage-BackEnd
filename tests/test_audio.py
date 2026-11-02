from __future__ import annotations

import struct

import pytest

from app.audio import AudioBuffer
from app.config import AudioSettings


def pcm(*samples: int) -> bytes:
    return b"".join(struct.pack("<h", sample) for sample in samples)


def array_of(*values: int) -> object:
    from array import array

    return array("h", values)


def test_accumulates_frames_and_reports_duration() -> None:
    settings = AudioSettings(sample_rate=16_000, channels=1)
    buffer = AudioBuffer(settings)

    buffer.append(pcm(*([100] * 8_000)))
    buffer.append(pcm(*([200] * 8_000)))

    assert buffer.byte_count == 32_000
    assert buffer.sample_count == 16_000
    assert buffer.frame_count == 16_000
    assert buffer.duration == pytest.approx(1.0)
    assert buffer.chunks == 2


def test_partial_frames_are_carried_over() -> None:
    buffer = AudioBuffer(AudioSettings())
    payload = pcm(*range(100))

    buffer.append(payload[:7])
    assert buffer.trailing_bytes == 1
    assert buffer.byte_count == 6

    buffer.append(payload[7:])
    assert buffer.trailing_bytes == 0
    assert buffer.to_bytes() == payload


def test_tiny_frame_is_held_until_a_full_sample_arrives() -> None:
    buffer = AudioBuffer(AudioSettings())

    buffer.append(b"\x01")
    assert buffer.byte_count == 0
    assert buffer.trailing_bytes == 1

    buffer.append(b"\x02")
    assert buffer.byte_count == 2
    assert buffer.to_int16() == array_of(513)


def test_stereo_downmix_averages_channels() -> None:
    settings = AudioSettings(channels=2)
    buffer = AudioBuffer(settings)
    buffer.append(struct.pack("<hhhh", 100, 300, -200, 0))

    assert buffer.sample_count == 4
    assert buffer.frame_count == 2
    assert buffer.to_mono_int16() == array_of(200, -100)


def test_float_conversion_normalises_to_unit_range() -> None:
    numpy = pytest.importorskip("numpy")
    buffer = AudioBuffer(AudioSettings())
    buffer.append(struct.pack("<hhh", 0, 16_384, -32_768))

    samples = buffer.to_numpy()

    assert samples.dtype == numpy.float32
    assert samples.shape == (3,)
    assert samples[0] == pytest.approx(0.0)
    assert samples[1] == pytest.approx(0.5)
    assert samples[2] == pytest.approx(-1.0)


def test_finish_stamps_the_utterance_and_clear_resets_it() -> None:
    buffer = AudioBuffer(AudioSettings())
    buffer.append(pcm(*([1] * 10)))

    finished = buffer.finish()
    assert finished.closed_at is not None

    buffer.clear()
    assert buffer.byte_count == 0
    assert buffer.chunks == 0
    assert buffer.closed_at is None
    assert buffer.trailing_bytes == 0


def test_describe_is_json_friendly() -> None:
    buffer = AudioBuffer(AudioSettings(sample_rate=8_000, channels=1))
    buffer.append(pcm(*([1] * 4_000)))

    described = buffer.describe()

    assert described["duration"] == pytest.approx(0.5)
    assert described["frameCount"] == 4_000
    assert described["format"] == {"encoding": "pcm_s16le", "sampleRate": 8_000, "channels": 1}
