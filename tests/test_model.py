from __future__ import annotations

import asyncio
import struct
import sys
import types

import pytest

from app.audio import AudioBuffer
from app.config import AudioSettings, WhisperSettings
from app.model import EchoModel, FasterWhisperModel, default_model


class FakeSegment:
    def __init__(self, start: float, end: float, text: str) -> None:
        self.start = start
        self.end = end
        self.text = text


class FakeInfo:
    language = "en"
    language_probability = 0.9812
    duration = 0.5


class FakeWhisperModel:
    def __init__(
        self,
        model_size_or_path: str,
        device: str = "auto",
        compute_type: str = "default",
        download_root: str | None = None,
    ) -> None:
        self.init_args = (model_size_or_path, device, compute_type, download_root)
        self.calls: list[tuple[object, dict[str, object]]] = []

    def transcribe(self, audio: object, **kwargs: object) -> object:
        self.calls.append((audio, kwargs))
        segments = [FakeSegment(0.0, 0.4, " hey pet"), FakeSegment(0.4, 0.5, "!")]
        return iter(segments), FakeInfo()


def audio_of(sample_rate: int, seconds: float = 0.5) -> AudioBuffer:
    buffer = AudioBuffer(AudioSettings(sample_rate=sample_rate, channels=1))
    buffer.append(struct.pack("<h", 1234) * int(sample_rate * seconds))
    return buffer


@pytest.fixture()
def created_models(monkeypatch: pytest.MonkeyPatch) -> list[FakeWhisperModel]:
    created: list[FakeWhisperModel] = []

    class RecordingWhisperModel(FakeWhisperModel):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)
            created.append(self)

    module = types.ModuleType("faster_whisper")
    module.WhisperModel = RecordingWhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    return created


def test_transcribe_maps_segments_and_reuses_the_loaded_model(
    created_models: list[FakeWhisperModel],
) -> None:
    settings = WhisperSettings(model_size="tiny", device="cpu", compute_type="int8", beam_size=2)
    model = FasterWhisperModel(settings)
    assert model.loaded is False

    result = asyncio.run(model.transcribe(audio_of(16_000)))

    assert result is not None
    assert result.text == "hey pet!"
    assert result.data["model"] == "tiny"
    assert result.data["language"] == "en"
    assert result.data["languageProbability"] == 0.9812
    assert result.data["segments"] == [
        {"start": 0.0, "end": 0.4, "text": "hey pet"},
        {"start": 0.4, "end": 0.5, "text": "!"},
    ]

    assert model.loaded is True
    assert len(created_models) == 1
    assert created_models[0].init_args == ("tiny", "cpu", "int8", None)

    _, kwargs = created_models[0].calls[0]
    assert kwargs["language"] == "en"
    assert kwargs["beam_size"] == 2
    assert kwargs["vad_filter"] is True
    assert kwargs["condition_on_previous_text"] is False


def test_16k_audio_is_handed_over_as_float32_samples(
    created_models: list[FakeWhisperModel],
) -> None:
    model = FasterWhisperModel(WhisperSettings(model_size="tiny"))
    asyncio.run(model.transcribe(audio_of(16_000)))

    payload, _ = created_models[0].calls[0]
    samples = payload.tolist()
    assert len(samples) == 8_000
    assert samples[0] == pytest.approx(1234 / 32768.0, rel=1e-6)


@pytest.mark.parametrize(("source_rate", "expected"), [(8_000, 8_000), (48_000, 8_000)])
def test_other_sample_rates_are_resampled_to_16k(
    created_models: list[FakeWhisperModel], source_rate: int, expected: int
) -> None:
    model = FasterWhisperModel(WhisperSettings(model_size="tiny"))
    asyncio.run(model.transcribe(audio_of(source_rate)))

    payload, _ = created_models[0].calls[0]
    samples = payload.tolist()
    assert len(samples) == expected
    assert samples[0] == pytest.approx(1234 / 32768.0, rel=1e-6)


def test_empty_audio_short_circuits_without_loading_the_model(
    created_models: list[FakeWhisperModel],
) -> None:
    model = FasterWhisperModel(WhisperSettings(model_size="tiny"))
    result = asyncio.run(model.transcribe(AudioBuffer(AudioSettings())))

    assert result is not None
    assert result.text == ""
    assert result.data == {"empty": True}
    assert created_models == []


def test_default_model_falls_back_to_echo_without_faster_whisper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    assert isinstance(default_model(), EchoModel)


def test_default_model_prefers_faster_whisper(created_models: list[FakeWhisperModel]) -> None:
    assert isinstance(default_model(), FasterWhisperModel)
