from __future__ import annotations

import argparse
import json
import sys
import time
import wave 

from websockets.sync.client import connect

DEFAULT_URL = "ws://localhost:8000/ws/audio"


def main() -> int:
    parser = argparse.ArgumentParser(description="Stream a WAV file to the audio socket.")
    parser.add_argument("wav", help="path to a 16-bit PCM wav file")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--chunk-ms", type=int, default=100)
    args = parser.parse_args()

    with wave.open(args.wav, "rb") as source:
        if source.getsampwidth() != 2:
            print("Only 16-bit PCM wav files are supported.", file=sys.stderr)
            return 2
        channels = source.getnchannels()
        sample_rate = source.getframerate()
        pcm = source.readframes(source.getnframes())

    chunk_bytes = max(2, sample_rate * channels * 2 * args.chunk_ms // 1000)
    print(f"{len(pcm)} bytes, {sample_rate} Hz, {channels} channel(s), {args.chunk_ms} ms frames")

    with connect(args.url) as ws:
        print("<-", ws.recv())

        ws.send(
            json.dumps(
                {
                    "type": "start",
                    "format": {"sampleRate": sample_rate, "channels": channels},
                }
            )
        )
        print("<-", ws.recv())

        for offset in range(0, len(pcm), chunk_bytes):
            ws.send(pcm[offset : offset + chunk_bytes])
            time.sleep(args.chunk_ms / 1000)

        ws.send(json.dumps({"type": "stop"}))

        while True:
            message = json.loads(ws.recv())
            print("<-", json.dumps(message, indent=2))
            if message.get("type") in {"result", "error"}:
                return 0 if message["type"] == "result" else 1


if __name__ == "__main__":
    raise SystemExit(main())
