from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from typing import Any

from google import genai
from google.genai import types

DEFAULT_MODEL = "gemini-2.0-flash"


class GeminiModel:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        timeout: float | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY, GEMINI_API_KEY, or OPENAI_API_KEY is not set")

        self.model = model or os.environ.get("GEMINI_MODEL") or os.environ.get("GOOGLE_MODEL") or os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens

        timeout_ms = None
        if timeout is not None:
            timeout_ms = int(timeout * 1000)
        elif os.environ.get("GEMINI_TIMEOUT"):
            timeout_ms = int(float(os.environ["GEMINI_TIMEOUT"]) * 1000)
        else:
            timeout_ms = 60_000

        http_options = types.HttpOptions(base_url=base_url, timeout=timeout_ms, api_version="v1beta")
        self._client = genai.Client(api_key=self.api_key, http_options=http_options)

    async def generate(self, prompt: str) -> str:
        return await self.chat(prompt)

    async def chat(self, prompt: str, history: list[dict[str, str]] | None = None) -> str:
        contents = self._build_contents(prompt, history)
        config = self._build_config()

        chunks = [chunk async for chunk in self._stream_complete(contents, config)]
        return "".join(chunks)

    async def chat_stream(
        self, prompt: str, history: list[dict[str, str]] | None = None
    ) -> AsyncGenerator[str, None]:
        contents = self._build_contents(prompt, history)
        config = self._build_config()

        async for chunk in self._stream_complete(contents, config):
            yield chunk

    def _build_contents(self, prompt: str, history: list[dict[str, str]] | None = None) -> list[types.Content]:
        contents: list[types.Content] = []
        for item in history or []:
            role = item.get("role", "user")
            if role == "assistant":
                role = "model"
            elif role not in {"user", "model"}:
                role = "user"
            text = item.get("content", "")
            contents.append(types.Content(role=role, parts=[types.Part.from_text(text=text)]))
        contents.append(types.Content(role="user", parts=[types.Part.from_text(text=prompt)]))
        return contents

    def _build_config(self) -> types.GenerateContentConfig:
        kwargs: dict[str, Any] = {"temperature": self.temperature}
        if self.system_prompt:
            kwargs["system_instruction"] = self.system_prompt
        if self.max_tokens is not None:
            kwargs["max_output_tokens"] = self.max_tokens
        return types.GenerateContentConfig(**kwargs)

    async def _stream_complete(
        self, contents: list[types.Content], config: types.GenerateContentConfig
    ) -> AsyncGenerator[str, None]:
        stream = await self._client.aio.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=config,
        )
        async for event in stream:
            text = getattr(event, "text", None)
            if text:
                yield text


# Backward-compatible name for existing imports.
GPTModel = GeminiModel


if __name__ == "__main__":
    import asyncio
    from pathlib import Path

    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

    async def main() -> None:
        model = GeminiModel(
            system_prompt="You are a concise assistant. Keep answers short.",
            temperature=0.2,
            max_tokens=128,
        )
        print(f"model={model.model}")

        prompt = "Reply with exactly: OK"
        print(f"\n[chat] prompt={prompt!r}")
        reply = await model.chat(prompt)
        print(f"[chat] reply={reply!r}")

        prompt = "Count from 1 to 5 separated by commas."
        print(f"\n[stream] prompt={prompt!r}")
        print("[stream] reply=", end="", flush=True)
        async for chunk in model.chat_stream(prompt):
            print(chunk, end="", flush=True)
        print()

    asyncio.run(main())
