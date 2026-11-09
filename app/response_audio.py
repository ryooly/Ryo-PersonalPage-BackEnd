from __future__ import annotations

import logging

from fastapi import WebSocket

from .audio import AudioBuffer
from .llm_model import GPTModel
from .model import AudioModel
from .tts_model import TTSModel

logger = logging.getLogger(__name__)

SENTENCE_ENDINGS = {".", "!", "?", "।", "؟"}


def _extract_complete_sentence(buffer: str) -> tuple[str, str]:
    """Split off the longest prefix ending with a sentence terminator.

    Returns (sentence, remaining_buffer). A terminator only counts when it
    is followed by whitespace or the end of the buffer, so decimals like
    "3.14" and abbreviations mid-chunk are not cut in half.
    """
    for index, char in enumerate(buffer):
        if char not in SENTENCE_ENDINGS:
            continue
        next_char = buffer[index + 1] if index + 1 < len(buffer) else ""
        if next_char == "" or next_char.isspace():
            sentence = buffer[: index + 1].strip()
            if sentence:
                return sentence, buffer[index + 1 :]
    return "", buffer


async def generate_voice_response(
    websocket: WebSocket,
    audio: AudioBuffer,
    transcriber: AudioModel,
    llm: GPTModel,
    tts: TTSModel,
    history: list[dict[str, str]] | None = None,
) -> str:
    await websocket.send_json({"type": "transcribing"})

    result = await transcriber.transcribe(audio)
    transcript = result.text if result else ""
    if not transcript or not transcript.strip():
        await websocket.send_json(
            {"type": "error", "code": "empty_transcript", "message": "No speech detected."}
        )
        return ""

    await websocket.send_json({"type": "transcript", "text": transcript})

    await websocket.send_json({"type": "thinking"})

    # Stream the LLM answer chunk by chunk; as soon as a full sentence has
    # arrived (ends with . ! ? etc.), hand it straight to TTS while the
    # model keeps generating the rest.
    await websocket.send_json({"type": "speaking"})

    buffer = ""
    response_text = ""
    async for chunk in llm.chat_stream(transcript, history=history):
        buffer += chunk
        sentence, buffer = _extract_complete_sentence(buffer)
        while sentence:
            response_text += sentence + " "
            await websocket.send_json({"type": "llm_response", "text": sentence})
            await tts.stream_over_websocket(sentence, websocket, meta={"text": sentence})
            sentence, buffer = _extract_complete_sentence(buffer)

    remaining = buffer.strip()
    if remaining:
        response_text += remaining + " "
        await websocket.send_json({"type": "llm_response", "text": remaining})
        await tts.stream_over_websocket(remaining, websocket, meta={"text": remaining})

    response_text = response_text.strip()
    if not response_text:
        await websocket.send_json(
            {"type": "error", "code": "empty_response", "message": "The model returned no text."}
        )
        return ""

    return response_text
