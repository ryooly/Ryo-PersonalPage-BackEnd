from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from .speak_error import speak_error
from .tts_model import TTSModel

logger = logging.getLogger(__name__)

router = APIRouter()
 
MAX_TEXT_BYTES = 8_192


async def tts_session(websocket: WebSocket, tts: TTSModel | None) -> None:
    await websocket.accept()

    if tts is None:
        await _send(
            websocket,
            {
                "type": "error",
                "code": "tts_unavailable",
                "message": "TTS model is not configured on this server.",
            },
        )
        await websocket.close()
        return

    await _send(websocket, {"type": "ready", "format": tts.output_format})

    try:
        while True:
            message = await websocket.receive_text()

            try:
                payload: dict[str, Any] = json.loads(message)
            except json.JSONDecodeError:
                await _error(websocket, tts, "bad_json", "Expected a JSON control frame.")
                continue

            if payload.get("type") == "ping":
                await _send(websocket, {"type": "pong"})
                continue

            text = payload.get("text")
            if not isinstance(text, str) or not text.strip():
                await _error(websocket, tts, "empty_text", 'Send {"type": "tts", "text": "..."} with non-empty text.')
                continue

            if len(text.encode("utf-8")) > MAX_TEXT_BYTES:
                await _error(
                    websocket,
                    tts,
                    "text_too_long",
                    f"Text exceeds the {MAX_TEXT_BYTES}-byte limit.",
                )
                continue

            frames = await tts.stream_over_websocket(text, websocket)
            logger.info("[tts] %d frames -> %.80s", frames, text.strip())

    except WebSocketDisconnect:
        pass
    finally:
        logger.debug("tts session closed")


async def _send(websocket: WebSocket, payload: dict[str, Any]) -> None:
    await websocket.send_json(payload)


async def _error(websocket: WebSocket, tts: TTSModel, code: str, message: str) -> None:
    await speak_error(message, websocket, tts, code=code)


@router.websocket("/ws/tts")
async def tts_socket(websocket: WebSocket) -> None:
    await tts_session(websocket, websocket.app.state.tts_model)
