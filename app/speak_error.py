from __future__ import annotations

import asyncio
from collections import OrderedDict
from typing import Any, TYPE_CHECKING

from fastapi import WebSocket

if TYPE_CHECKING:
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
            try:
                # Send chunk to async sender immediately; do not wait for full synthesis.
                asyncio.run_coroutine_threadsafe(queue.put(chunk), loop).result()
            except Exception:
                break
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
    tts: "TTSModel | None",
    code: str = "error",
    meta: dict[str, Any] | None = None,
) -> int:
    """Replace an error text response by speaking it through TTS.

    Always sends the JSON error frame. If a TTS model is available, also speaks
    the message through the websocket TTS streaming flow with caching.
    """
    message = " ".join((text or "").split()) or "An error occurred."
    error_payload: dict[str, Any] = {"type": "error", "code": code, "message": message}
    await websocket.send_json(error_payload)
    if tts is None:
        return 0
    merged_meta: dict[str, Any] = {"spoken_error": True, "code": code, "text": message}
    if meta:
        merged_meta.update(meta)
    try:
        return await stream_text_over_websocket(message, websocket, tts, meta=merged_meta)
    except Exception:
        # Error JSON was already delivered; do not fail the caller if speech fails.
        return 0



# edited 2moro (harusnya koneksikan dengan tts model)