from __future__ import annotations

import asyncio
import os
from collections.abc import Iterator
from typing import Any

import edge_tts
from fastapi import WebSocket

DEFAULT_VOICE_ID = "en-US-AriaNeural"
DEFAULT_PITCH = "+0Hz"
DEFAULT_RATE = "+0%"
DEFAULT_OUTPUT_FORMAT = "mp3_24khz_48kbitrate"


class TTSModel:
    def __init__(
        self,
        api_key: str | None = None,
        voice_id: str | None = None,
        model_id: str | None = None,
        output_format: str | None = None,
        pitch: str | None = None,
        rate: str | None = None,
    ) -> None:
        self.voice_id = voice_id or os.environ.get("EDGE_TTS_VOICE", DEFAULT_VOICE_ID)
        self.pitch = pitch or os.environ.get("EDGE_TTS_PITCH", DEFAULT_PITCH)
        self.rate = rate or os.environ.get("EDGE_TTS_RATE", DEFAULT_RATE)
        self.output_format = output_format or DEFAULT_OUTPUT_FORMAT

    def synthesize_stream(self, text: str) -> Iterator[bytes]:
        if not text.strip():
            raise ValueError("text must not be empty")

        communicate = edge_tts.Communicate(text, self.voice_id, pitch=self.pitch, rate=self.rate)
        for chunk in communicate.stream_sync():
            if chunk["type"] == "audio":
                yield chunk["data"]

    def synthesize(self, text: str) -> bytes:
        return b"".join(self.synthesize_stream(text))

    def save(self, text: str, path: str) -> str:
        with open(path, "wb") as file:
            for chunk in self.synthesize_stream(text):
                file.write(chunk)
        return path

    async def stream_over_websocket(
        self,
        text: str,
        websocket: WebSocket,
        meta: dict[str, Any] | None = None,
    ) -> int:
        payload: dict[str, Any] = {"type": "tts_start", "format": self.output_format}
        if meta:
            payload.update(meta)
        await websocket.send_json(payload)

        iterator = iter(self.synthesize_stream(text))
        frames = 0
        while True:
            chunk = await asyncio.to_thread(next, iterator, None)
            if chunk is None:
                break
            await websocket.send_bytes(chunk)
            frames += 1

        await websocket.send_json({"type": "tts_end", "frames": frames})
        return frames




# 2moro I ll improve webscoket