from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from .config import ENCODING, AudioSettings
from .llm_model import GPTModel
from .model import AudioModel, FasterWhisperModel, default_model
from .tts_model import TTSModel
from .ws_audio import router as audio_router
from .ws_tts import router as tts_router

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

logger = logging.getLogger(__name__)

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)


def default_llm() -> GPTModel | None:
    try:
        return GPTModel()
    except ValueError as exc:
        logger.warning("LLM disabled, voice replies off: %s", exc)
        return None


def default_tts() -> TTSModel | None:
    try:
        return TTSModel()
    except ValueError as exc:
        logger.warning("TTS disabled, voice replies off: %s", exc)
        return None


class ChatRequest(BaseModel):
    text: str
    history: list[dict[str, str]] | None = None


def create_app(
    model: AudioModel | None = None,
    settings: AudioSettings | None = None,
    llm: GPTModel | None = None,
    tts: TTSModel | None = None,
) -> FastAPI:
    resolved_settings = settings or AudioSettings.from_env()
    resolved_model = model or default_model()
    resolved_llm = llm if llm is not None else default_llm()
    resolved_tts = tts if tts is not None else default_tts()

    app = FastAPI(title="Tamagotchi audio backend", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.audio_settings = resolved_settings
    app.state.audio_model = resolved_model
    app.state.llm_model = resolved_llm
    app.state.tts_model = resolved_tts

    app.include_router(audio_router)
    app.include_router(tts_router)

    @app.get("/healthz")
    async def healthz() -> dict[str, object]:
        payload: dict[str, object] = {
            "status": "ok",
            "model": type(resolved_model).__name__,
            "encoding": ENCODING,
            "format": resolved_settings.wire_format(),
            "limits": {
                "minSeconds": resolved_settings.min_seconds,
                "maxSeconds": resolved_settings.max_seconds,
            },
        }
        if isinstance(resolved_model, FasterWhisperModel):
            payload["modelSettings"] = {
                "loaded": resolved_model.loaded,
                **resolved_model.settings.describe(),
            }
        return payload

    @app.post("/api/chat")
    async def chat(request: ChatRequest) -> dict[str, Any]:
        text = request.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="text must not be empty")
        llm_model = app.state.llm_model
        if llm_model is None:
            raise HTTPException(status_code=503, detail="LLM model is not configured")
        reply = await llm_model.chat(text, history=request.history)
        return {"text": reply}

    return app


app = create_app()
