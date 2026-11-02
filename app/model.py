from __future__ import annotations

import abc
import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .audio import AudioBuffer
from .config import WhisperSettings

if TYPE_CHECKING:
    import numpy as np
    from faster_whisper import WhisperModel
    from faster_whisper.transcribe import Segment, TranscriptionInfo

logger = logging.getLogger(__name__)

WHISPER_SAMPLE_RATE = 16_000


@dataclass(slots=True)
class ModelResult:
    text: str | None = None
    data: dict[str, Any] = field(default_factory=dict)


class AudioModel(abc.ABC):
    @abc.abstractmethod
    async def transcribe(self, audio: AudioBuffer) -> ModelResult | None:
        raise NotImplementedError


class EchoModel(AudioModel):
    async def transcribe(self, audio: AudioBuffer) -> ModelResult | None:
        seconds = audio.duration
        return ModelResult(
            text=f"[stub] received {seconds:.2f}s of audio",
            data={"bytes": audio.byte_count, "chunks": audio.chunks},
        )


class FasterWhisperModel(AudioModel):
    def __init__(self, settings: WhisperSettings | None = None) -> None:
        self.settings = settings or WhisperSettings.from_env()
        self._model: WhisperModel | None = None
        self._load_lock = asyncio.Lock()
        self._run_lock = asyncio.Lock()

    @staticmethod
    def is_available() -> bool:
        try:
            import faster_whisper  # noqa: F401
        except ImportError:
            return False
        return True

    @property
    def loaded(self) -> bool:
        return self._model is not None

    async def transcribe(self, audio: AudioBuffer) -> ModelResult | None:
        if audio.is_empty:
            return ModelResult(text="", data={"empty": True})

        model = await self._ensure_model()
        payload = _prepare_audio(audio)

        async with self._run_lock:
            segments, info = await asyncio.to_thread(self._run, model, payload)

        text = "".join(segment.text for segment in segments).strip()

        return ModelResult(
            text=text,
            data={
                "model": self.settings.model_size,
                "language": info.language,
                "languageProbability": round(info.language_probability, 4),
                "duration": round(info.duration, 3),
                "segments": [
                    {
                        "start": round(segment.start, 3),
                        "end": round(segment.end, 3),
                        "text": segment.text.strip(),
                    }
                    for segment in segments
                ],
            },
        )

    def _run(
        self, model: WhisperModel, payload: np.ndarray
    ) -> tuple[list[Segment], TranscriptionInfo]:
        segments, info = model.transcribe(
            payload,
            language=self.settings.language,
            beam_size=self.settings.beam_size,
            vad_filter=self.settings.vad_filter,
            condition_on_previous_text=False,
        )
        return list(segments), info

    async def _ensure_model(self) -> WhisperModel:
        if self._model is not None:
            return self._model

        async with self._load_lock:
            if self._model is None:
                self._model = await asyncio.to_thread(self._load)
        return self._model

    def _load(self) -> WhisperModel:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper is not installed; run `pip install -r requirements.txt`"
            ) from exc

        logger.info(
            "loading faster-whisper model %r (device=%s, compute_type=%s)",
            self.settings.model_size,
            self.settings.device,
            self.settings.compute_type,
        )
        model = WhisperModel(
            self.settings.model_size,
            device=self.settings.device,
            compute_type=self.settings.compute_type,
            download_root=self.settings.download_root,
        )
        logger.info("faster-whisper model %r is ready", self.settings.model_size)
        return model


def _prepare_audio(audio: AudioBuffer) -> np.ndarray:
    import numpy as np

    samples = audio.to_numpy()
    source_rate = audio.settings.sample_rate

    if source_rate == WHISPER_SAMPLE_RATE or len(samples) < 2:
        return samples

    step = source_rate // WHISPER_SAMPLE_RATE
    if step > 1 and source_rate % WHISPER_SAMPLE_RATE == 0:
        usable = len(samples) - len(samples) % step
        if usable:
            return samples[:usable].reshape(-1, step).mean(axis=1, dtype=np.float32)

    return _resample(samples, source_rate)


def _resample(samples: np.ndarray, source_rate: int) -> np.ndarray:
    import numpy as np

    target_length = max(1, round(len(samples) * WHISPER_SAMPLE_RATE / source_rate))
    source_positions = np.linspace(0.0, len(samples) - 1, target_length)
    return np.interp(source_positions, np.arange(len(samples)), samples).astype(np.float32)


def default_model() -> AudioModel:
    if FasterWhisperModel.is_available():
        return FasterWhisperModel()
    logger.warning(
        "faster-whisper is not installed, falling back to EchoModel; "
        "run `pip install -r requirements.txt` to enable transcription"
    )
    return EchoModel()
