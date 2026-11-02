from __future__ import annotations

import logging

from fastapi import FastAPI

from .config import ENCODING, AudioSettings
from .model import AudioModel, FasterWhisperModel, default_model
from .ws_audio import router as audio_router

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def create_app(
    model: AudioModel | None = None,
    settings: AudioSettings | None = None,
) -> FastAPI:
    resolved_settings = settings or AudioSettings.from_env()
    resolved_model = model or default_model()

    app = FastAPI(title="Tamagotchi audio backend", version="0.1.0")
    app.state.audio_settings = resolved_settings
    app.state.audio_model = resolved_model

    app.include_router(audio_router)

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

    return app


app = create_app()
