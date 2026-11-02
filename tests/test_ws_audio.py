from __future__ import annotations

import struct

import pytest
from fastapi.testclient import TestClient

from app.audio import AudioBuffer
from app.config import AudioSettings
from app.main import create_app
from app.model import AudioModel, ModelResult


class RecordingModel(AudioModel):
    def __init__(self, *, fail: bool = False) -> None:
        self.received: list[AudioBuffer] = []
        self.fail = fail

    async def transcribe(self, audio: AudioBuffer) -> ModelResult | None:
        self.received.append(audio)
        if self.fail:
            raise RuntimeError("model exploded")
        return ModelResult(text="hello pet", data={"confidence": 0.9})


def make_pcm(seconds: float, settings: AudioSettings) -> bytes:
    frames = int(settings.sample_rate * seconds)
    sample = struct.pack("<h", 1234)
    return sample * frames * settings.channels


@pytest.fixture()
def settings() -> AudioSettings:
    return AudioSettings(sample_rate=16_000, channels=1)


@pytest.fixture()
def model() -> RecordingModel:
    return RecordingModel()


@pytest.fixture()
def client(model: RecordingModel, settings: AudioSettings) -> TestClient:
    return TestClient(create_app(model=model, settings=settings))


def test_healthz(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_streams_audio_and_returns_model_result(
    client: TestClient, model: RecordingModel, settings: AudioSettings
) -> None:
    pcm = make_pcm(0.5, settings)

    with client.websocket_connect("/ws/audio") as ws:
        ready = ws.receive_json()
        assert ready["type"] == "ready"
        assert ready["format"] == {"encoding": "pcm_s16le", "sampleRate": 16_000, "channels": 1}
        assert ready["limits"]["maxSeconds"] == 30.0

        ws.send_json({"type": "start"})
        assert ws.receive_json()["type"] == "started"

        for offset in range(0, len(pcm), 4096):
            ws.send_bytes(pcm[offset : offset + 4096])

        ws.send_json({"type": "stop"})
        assert ws.receive_json()["type"] == "processing"
        result = ws.receive_json()

    assert result["type"] == "result"
    assert result["text"] == "hello pet"
    assert result["data"] == {"confidence": 0.9}
    assert result["audio"]["duration"] == pytest.approx(0.5, abs=0.01)
    assert result["audio"]["bytes"] == len(pcm)
    assert result["audio"]["chunks"] == 4

    assert len(model.received) == 1
    handed_over = model.received[0]
    assert handed_over.to_bytes() == pcm
    assert handed_over.byte_count == len(pcm)


def test_rejects_audio_before_start(client: TestClient) -> None:
    with client.websocket_connect("/ws/audio") as ws:
        ws.receive_json()
        ws.send_bytes(b"\x00\x01" * 100)
        error = ws.receive_json()

    assert error["type"] == "error"
    assert error["code"] == "not_started"


def test_odd_length_frames_stay_byte_aligned(
    client: TestClient, model: RecordingModel, settings: AudioSettings
) -> None:
    pcm = make_pcm(0.5, settings) + b"\x07"

    with client.websocket_connect("/ws/audio") as ws:
        ws.receive_json()
        ws.send_json({"type": "start"})
        ws.receive_json()
        for offset in range(0, len(pcm), 3):
            ws.send_bytes(pcm[offset : offset + 3])
        ws.send_json({"type": "stop"})
        ws.receive_json()
        result = ws.receive_json()

    assert result["type"] == "result"
    assert result["audio"]["bytes"] == 16_000
    assert model.received[0].trailing_bytes == 1


def test_too_short_audio_is_rejected(client: TestClient, model: RecordingModel) -> None:
    with client.websocket_connect("/ws/audio") as ws:
        ws.receive_json()
        ws.send_json({"type": "start"})
        ws.receive_json()
        ws.send_bytes(b"\x00\x00" * 100)
        ws.send_json({"type": "stop"})
        error = ws.receive_json()

    assert error["code"] == "too_short"
    assert model.received == []


def test_stop_without_audio_is_rejected(client: TestClient) -> None:
    with client.websocket_connect("/ws/audio") as ws:
        ws.receive_json()
        ws.send_json({"type": "start"})
        ws.receive_json()
        ws.send_json({"type": "stop"})
        error = ws.receive_json()

    assert error["code"] == "too_short"


def test_buffer_overflow_clears_the_utterance(model: RecordingModel) -> None:
    settings = AudioSettings(sample_rate=16_000, channels=1, max_seconds=0.5)
    client = TestClient(create_app(model=model, settings=settings))

    with client.websocket_connect("/ws/audio") as ws:
        ws.receive_json()
        ws.send_json({"type": "start"})
        ws.receive_json()
        ws.send_bytes(b"\x00\x00" * 9_000)
        overflow = ws.receive_json()
        ws.send_bytes(b"\x00\x00" * 100)
        after = ws.receive_json()

    assert overflow["code"] == "buffer_full"
    assert after["code"] == "not_started"
    assert model.received == []


def test_model_failure_keeps_the_socket_usable(client: TestClient, model: RecordingModel) -> None:
    model.fail = True

    with client.websocket_connect("/ws/audio") as ws:
        ws.receive_json()
        ws.send_json({"type": "start"})
        ws.receive_json()
        ws.send_bytes(b"\x00\x00" * 8_000)
        ws.send_json({"type": "stop"})
        ws.receive_json()
        error = ws.receive_json()

        ws.send_json({"type": "ping"})
        pong = ws.receive_json()

    assert error["code"] == "model_failed"
    assert pong["type"] == "pong"


def test_format_can_be_negotiated_per_session(client: TestClient, settings: AudioSettings) -> None:
    with client.websocket_connect("/ws/audio") as ws:
        ws.receive_json()
        ws.send_json({"type": "start", "format": {"sampleRate": 8_000, "channels": 2}})
        started = ws.receive_json()

        stereo = AudioSettings(sample_rate=8_000, channels=2)
        ws.send_bytes(make_pcm(0.3, stereo))
        ws.send_json({"type": "stop"})
        ws.receive_json()
        result = ws.receive_json()

    assert started["format"]["sampleRate"] == 8_000
    assert started["format"]["channels"] == 2
    assert result["audio"]["duration"] == pytest.approx(0.3, abs=0.01)
    assert result["audio"]["frameCount"] == 2_400


@pytest.mark.parametrize(
    "requested",
    [
        {"sampleRate": 4_000},
        {"channels": 6},
        {"encoding": "pcm_f32le"},
        "not-an-object",
    ],
)
def test_unsupported_format_is_rejected(client: TestClient, requested: object) -> None:
    with client.websocket_connect("/ws/audio") as ws:
        ws.receive_json()
        ws.send_json({"type": "start", "format": requested})
        error = ws.receive_json()

    assert error["code"] == "unsupported_format"


def test_bad_json_and_unknown_events(client: TestClient) -> None:
    with client.websocket_connect("/ws/audio") as ws:
        ws.receive_json()
        ws.send_text("not json at all")
        assert ws.receive_json()["code"] == "bad_json"

        ws.send_json({"type": "dance"})
        assert ws.receive_json()["code"] == "unknown_type"


def test_reset_drops_buffered_audio(client: TestClient, model: RecordingModel) -> None:
    with client.websocket_connect("/ws/audio") as ws:
        ws.receive_json()
        ws.send_json({"type": "start"})
        ws.receive_json()
        ws.send_bytes(b"\x00\x00" * 8_000)
        ws.send_json({"type": "reset"})
        assert ws.receive_json()["type"] == "reset"

        ws.send_json({"type": "stop"})
        error = ws.receive_json()

    assert error["code"] == "too_short"
    assert model.received == []


def test_closing_mid_stream_does_not_raise(client: TestClient, model: RecordingModel) -> None:
    with client.websocket_connect("/ws/audio") as ws:
        ws.receive_json()
        ws.send_json({"type": "start"})
        ws.receive_json()
        ws.send_bytes(b"\x00\x00" * 1_000)
    assert model.received == []
