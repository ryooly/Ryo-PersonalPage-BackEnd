from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from .audio import AudioBuffer
from .config import ENCODING, AudioSettings
from .llm_model import GPTModel
from .model import AudioModel
from .response_audio import generate_voice_response
from .tts_model import TTSModel

logger = logging.getLogger(__name__)

router = APIRouter()

MIN_SAMPLE_RATE = 8_000
MAX_SAMPLE_RATE = 48_000
MAX_CHANNELS = 2


@dataclass(slots=True)
class SessionStats:
    utterances: int = 0
    bytes_received: int = 0
    rejected_chunks: int = 0


async def stream_audio_session(
    websocket: WebSocket,
    model: AudioModel,
    settings: AudioSettings,
    llm: GPTModel | None = None,
    tts: TTSModel | None = None,
) -> SessionStats:
    stats = SessionStats()

    await websocket.accept()

    session = settings
    buffer = AudioBuffer(session)
    recording = False

    await _send(
        websocket,
        {
            "type": "ready",
            "format": session.wire_format(),
            "limits": {
                "minSeconds": session.min_seconds,
                "maxSeconds": session.max_seconds,
                "maxChunkBytes": session.max_chunk_bytes,
            },
        },
    )

    try:
        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                break

            chunk = message.get("bytes")
            if chunk is not None:
                stats, session, buffer, recording = await _handle_audio_frame(
                    websocket, chunk, stats, session, buffer, recording
                )
                continue

            text = message.get("text")
            if text is None:
                continue

            stats, session, buffer, recording, done = await _handle_control(
                websocket, text, model, llm, tts, settings, stats, session, buffer, recording
            )
            if done:
                break

    except WebSocketDisconnect:
        logger.debug("client disconnected mid-utterance")
    finally:
        logger.info(
            "audio session closed: %d utterance(s), %d bytes buffered",
            stats.utterances,
            stats.bytes_received,
        )

    return stats


async def _handle_audio_frame(
    websocket: WebSocket,
    chunk: bytes,
    stats: SessionStats,
    session: AudioSettings,
    buffer: AudioBuffer,
    recording: bool,
) -> tuple[SessionStats, AudioSettings, AudioBuffer, bool]:
    if not recording:
        stats.rejected_chunks += 1
        await _error(websocket, "not_started", 'Send {"type": "start"} before streaming audio.')
        return stats, session, buffer, recording

    if len(chunk) > session.max_chunk_bytes:
        stats.rejected_chunks += 1
        await _error(
            websocket,
            "chunk_too_large",
            f"Frame is {len(chunk)} bytes, the limit is {session.max_chunk_bytes}.",
        )
        return stats, session, buffer, recording

    buffer.append(chunk)
    stats.bytes_received += len(chunk)

    if buffer.byte_count > session.max_bytes:
        await _error(
            websocket,
            "buffer_full",
            f"Utterance exceeded {session.max_seconds:.0f}s; the buffer was dropped.",
        )
        return stats, session, AudioBuffer(session), False

    return stats, session, buffer, recording


async def _handle_control(
    websocket: WebSocket,
    text: str,
    model: AudioModel,
    llm: GPTModel | None,
    tts: TTSModel | None,
    base_settings: AudioSettings,
    stats: SessionStats,
    session: AudioSettings,
    buffer: AudioBuffer,
    recording: bool,
) -> tuple[SessionStats, AudioSettings, AudioBuffer, bool, bool]:
    try:
        event = json.loads(text)
    except json.JSONDecodeError:
        await _error(websocket, "bad_json", "Text frames must contain JSON.")
        return stats, session, buffer, recording, False

    if not isinstance(event, dict):
        await _error(websocket, "bad_json", "Expected a JSON object.")
        return stats, session, buffer, recording, False

    kind = event.get("type") 

    if kind == "start":
        try:
            session = _resolve_format(base_settings, event.get("format"))
        except ValueError as exc:
            await _error(websocket, "unsupported_format", str(exc))
            return stats, session, buffer, recording, False

        await _send(websocket, {"type": "started", "format": session.wire_format()})
        return stats, session, AudioBuffer(session), True, False

    if kind == "stop":
        if not recording:
            await _error(websocket, "not_started", "Nothing is being recorded.")
            return stats, session, buffer, recording, False

        utterance = buffer.finish()
        next_buffer = AudioBuffer(session)

        if utterance.byte_count == 0:
            await _error(websocket, "too_short", "No audio arrived before stop.")
            return stats, session, next_buffer, False, False

        if utterance.byte_count < session.min_bytes:
            await _error(
                websocket,
                "too_short",
                f"Audio is {utterance.duration:.2f}s, the minimum is {session.min_seconds:.2f}s.",
            ) # ganti bagian ini jadi response ai terhadap aksi users (menggunkan tts)
            return stats, session, next_buffer, False, False

        sent = await _send(
            websocket,
            {
                "type": "processing",
                "duration": round(utterance.duration, 3),
                "audio": utterance.describe(),
            },
        )
        if not sent:
            return stats, session, next_buffer, False, True

        if llm is None or tts is None:
            await _error(
                websocket,
                "pipeline_unavailable",
                "The voice response pipeline (LLM + TTS) is not configured.",
            )
            return stats, session, next_buffer, False, False

        try:
            response_text = await generate_voice_response(
                websocket, utterance, model, llm, tts
            )
            print(f"[pipeline] {utterance.duration:.2f}s -> {response_text!r}", flush=True)
        except Exception:
            logger.exception("voice response failed on %.2fs of audio", utterance.duration)
            await _error(websocket, "pipeline_failed", "The response pipeline raised an exception; check the server log.")
            return stats, session, next_buffer, False, False

        stats.utterances += 1
        delivered = await _send(
            websocket,
            {"type": "turn_complete", "audio": utterance.describe(), "text": response_text},
        )
        return stats, session, next_buffer, False, not delivered

    if kind == "reset":
        await _send(websocket, {"type": "reset"})
        return stats, session, AudioBuffer(session), recording, False

    if kind == "ping":
        await _send(websocket, {"type": "pong"})
        return stats, session, buffer, recording, False

    await _error(websocket, "unknown_type", f"Unknown event type: {kind!r}.")
    return stats, session, buffer, recording, False


def _resolve_format(base: AudioSettings, requested: Any) -> AudioSettings:
    if requested is None:
        return base
    if not isinstance(requested, dict):
        raise ValueError('"format" must be an object.')

    encoding = requested.get("encoding", ENCODING)
    if encoding != ENCODING:
        raise ValueError(f"Only {ENCODING} is supported, got {encoding!r}.")

    sample_rate = _coerce_int(requested.get("sampleRate", requested.get("sample_rate")), base.sample_rate)
    channels = _coerce_int(requested.get("channels"), base.channels)

    if not MIN_SAMPLE_RATE <= sample_rate <= MAX_SAMPLE_RATE:
        raise ValueError(f"sampleRate must be between {MIN_SAMPLE_RATE} and {MAX_SAMPLE_RATE}.")
    if channels not in (1, MAX_CHANNELS):
        raise ValueError(f"channels must be 1 or {MAX_CHANNELS}.")

    return replace(base, sample_rate=sample_rate, channels=channels)


def _coerce_int(value: Any, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Expected an integer, got {value!r}.")
    return value


async def _send(websocket: WebSocket, payload: dict[str, Any]) -> bool:
    if websocket.client_state is not WebSocketState.CONNECTED:
        return False
    try:
        await websocket.send_json(payload)
        return True
    except (WebSocketDisconnect, RuntimeError):
        return False


async def _error(websocket: WebSocket, code: str, message: str) -> None:
    await _send(websocket, {"type": "error", "code": code, "message": message})


@router.websocket("/ws/audio")
async def audio_socket(websocket: WebSocket) -> None:
    await stream_audio_session(
        websocket,
        websocket.app.state.audio_model,
        websocket.app.state.audio_settings,
        websocket.app.state.llm_model,
        websocket.app.state.tts_model,
    )



# tambahkan beberpa feature penting, apila users berbicara terlalu panjang maka akan secara otomatis diehentikan kemudain untuk repsojn kek ucapa yang terllau panang dll juga akan ddihentikan dan dilempari suara ai 