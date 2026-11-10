from __future__ import annotations

import asyncio
from collections import OrderedDict
from typing import Any

from fastapi import WebSocket

from .tts_model import TTSModel

# Simple in-memory cache: normalized text + voice settings -> synthesized audio chunks.
# This allows repeated messages to be streamed immediately without waiting for synthesis.
_AUDIO_CACHE: OrderedDict[str, list[bytes]] = OrderedDict()
_CACHE_LOCK = asyncio.Lock()
_MAX_CACHE_ENTRIES = 64
_MAX_CACHE_BYTES = 8 * 1024 * 1024


def _cache_key(text: str, tts: TTSModel) -> str:
    normalized = " ".join(text.split())
    return "\x00".join(
        [
            str(getattr(tts, "voice_id", "")),
            str(getattr(tts, "pitch", "")),
            str(getattr(tts, "rate", "")),
            str(getattr(tts, "output_format", "")),
            normalized,
        ]
    )


async def _cache_get(key: str) -> list[bytes] | None:
    async with _CACHE_LOCK:
        chunks = _AUDIO_CACHE.get(key)
        if chunks is not None:
            _AUDIO_CACHE.move_to_end(key)
            return [bytes(c) for c in chunks]
        return None


async def _cache_put(key: str, chunks: list[bytes]) -> None:
    if not chunks:
        return
    total = sum(len(c) for c in chunks)
    if total > _MAX_CACHE_BYTES:
        return
    async with _CACHE_LOCK:
        _AUDIO_CACHE[key] = [bytes(c) for c in chunks]
        _AUDIO_CACHE.move_to_end(key)
        while len(_AUDIO_CACHE) > _MAX_CACHE_ENTRIES:
            _AUDIO_CACHE.popitem(last=False)


async def stream_text_over_websocket(
    text: str,
    websocket: WebSocket,
    tts: TTSModel,
    meta: dict[str, Any] | None = None,
) -> int:
    """Stream text as TTS audio over WebSocket with caching and pipelined delivery.

    Uses the same WebSocket protocol as TTSModel.stream_over_websocket
    (tts_start, binary audio chunks, tts_end), but:
    - serves cached audio immediately for repeated text, and
    - for uncached text, synthesizes in a background thread and forwards each
      chunk to the client as soon as it is produced (no wait for full completion).
    """
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return 0

    key = _cache_key(cleaned, tts)
    cached = await _cache_get(key)

    payload: dict[str, Any] = {"type": "tts_start", "format": tts.output_format}
    if meta:
        payload.update(meta)
    await websocket.send_json(payload)

    if cached is not None:
        frames = 0
        for chunk in cached:
            await websocket.send_bytes(chunk)
            frames += 1
        await websocket.send_json({"type": "tts_end", "frames": frames, "cached": True})
        return frames

    queue: asyncio.Queue[bytes | BaseException | None] = asyncio.Queue(maxsize=16)
    loop = asyncio.get_running_loop()

    def produce() -> None:
        try:
            for chunk in tts.synthesize_stream(cleaned):
                # Send chunk to async sender immediately; do not wait for full synthesis.
                asyncio.run_coroutine_threadsafe(queue.put(chunk), loop).result()
            asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()
        except BaseException as exc:  # pragma: no cover - surfaced to caller
            asyncio.run_coroutine_threadsafe(queue.put(exc), loop).result()

    producer = asyncio.create_task(asyncio.to_thread(produce))
    frames = 0
    collected: list[bytes] = []
    error: BaseException | None = None

    try:
        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, BaseException):
                error = item
                break
            collected.append(bytes(item))
            await websocket.send_bytes(item)
            frames += 1
    finally:
        if not producer.done():
            producer.cancel()
        else:
            try:
                await producer
            except Exception:
                pass

    if error is not None:
        raise error

    await websocket.send_json({"type": "tts_end", "frames": frames, "cached": False})
    await _cache_put(key, collected)
    return frames


async def speak_error(
    text: str,
    websocket: WebSocket,
    tts: TTSModel,
    meta: dict[str, Any] | None = None,
) -> int:
    """Replace an error text response by speaking it through TTS.

    Accepts the error/message text and processes it through the websocket TTS
    streaming flow (equivalent to TTSModel.stream_over_websocket) with caching.
    """
    message = " ".join((text or "").split()) or "An error occurred."
    merged_meta: dict[str, Any] = {"spoken_error": True, "text": message}
    if meta:
        merged_meta.update(meta)
    return await stream_text_over_websocket(message, websocket, tts, meta=merged_meta)



# edited 2moro