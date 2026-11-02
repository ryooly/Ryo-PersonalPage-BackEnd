# Audio backend

Receives microphone audio over a WebSocket, buffers the raw PCM frames into one
contiguous array and hands the finished utterance to your AI model.

Python 3.11+ (built and tested on 3.13), FastAPI + uvicorn.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Tests: `pip install -r requirements-dev.txt; pytest`

Smoke test with a real socket — generate a file, then stream it:

```powershell
python client_example.py sample.wav --chunk-ms 50
```

`GET /healthz` reports the active model and format.

## Where your model plugs in

`app/model.py` is the seam. Nothing in the socket layer knows what happens after
`transcribe()` is called.

`FasterWhisperModel` is the default: local speech-to-text through
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2, no API
key). The weights are downloaded on the first utterance and cached afterwards,
so startup stays instant and `/healthz` shows `"loaded": false` until then.

```python
from app.main import create_app

app = create_app()                    # FasterWhisperModel, tuned by env vars below
```

Audio is converted to mono float32 at 16 kHz before it reaches the engine —
faster-whisper only accepts that rate, and every negotiated stream rate
(8–48 kHz, mono or stereo) is resampled locally.

To plug in something else, subclass `AudioModel` and inject it:

```python
from app.audio import AudioBuffer
from app.model import AudioModel, ModelResult


class MyModel(AudioModel):
    async def transcribe(self, audio: AudioBuffer) -> ModelResult | None:
        waveform = audio.to_numpy()          # float32 mono in [-1, 1]
        text = await my_client.transcribe(waveform, sample_rate=audio.settings.sample_rate)
        return ModelResult(text=text, data={"language": "en"})


app = create_app(model=MyModel())
```

If your model call blocks (most SDKs do), push it off the event loop:
`await anyio.to_thread.run_sync(fn)` or `await asyncio.to_thread(fn)`. Otherwise
every other connected device stalls while one model is thinking.
`FasterWhisperModel` already does this, and runs one utterance at a time.

`EchoModel` is the fallback used when `faster-whisper` cannot be imported — it
just reports how much audio it got, which keeps the frontend honest during
setup.

## Wire protocol

One JSON object per text frame; audio travels as binary frames.

| Direction | Message | Meaning |
| --- | --- | --- |
| server → client | `{"type":"ready","format":{...},"limits":{...}}` | Sent on connect |
| client → server | `{"type":"start","format":{"sampleRate":16000,"channels":1}}` | Opens an utterance; `format` is optional |
| server → client | `{"type":"started","format":{...}}` | Confirms the negotiated format |
| client → server | `<binary>` | Raw `pcm_s16le` audio, any frame size |
| client → server | `{"type":"stop"}` | Ends the utterance |
| server → client | `{"type":"processing","duration":1.0,"audio":{...}}` | Model call started |
| server → client | `{"type":"result","audio":{...},"text":"...","data":{...}}` | Model answer |
| client → server | `{"type":"reset"}` | Drops buffered audio, keeps recording |
| client → server | `{"type":"ping"}` | Keepalive, answered with `pong` |
| server → client | `{"type":"error","code":"...","message":"..."}` | Recoverable failure |

Errors never close the socket, so a client can fix itself and continue. Codes:
`not_started`, `chunk_too_large`, `buffer_full`, `bad_json`, `unknown_type`,
`unsupported_format`, `too_short`, `model_failed`.

Two state details worth knowing: sending `start` again discards whatever is
still buffered, and after `buffer_full` the stream stops accepting audio until
the client sends `start` once more.

## The buffer

`AudioBuffer` (`app/audio.py`) accumulates bytes, nothing else:

```python
buffer.byte_count / sample_count / frame_count / duration
buffer.to_bytes()        # untouched pcm_s16le payload
buffer.to_int16()        # array('h')
buffer.to_mono_int16()   # downmixed if the stream is stereo
buffer.to_numpy()        # float32 mono in [-1, 1]  (needs numpy)
buffer.describe()        # JSON summary sent in result frames
```

Chunk boundaries are not assumed. If a client sends a frame that ends halfway
through a sample, the leftover byte is carried into the next frame, so the
buffered audio is always aligned — that is what `trailing_bytes` exposes.

## Configuration

Environment variables, read once at startup:

| Variable | Default | Meaning |
| --- | --- | --- |
| `AUDIO_SAMPLE_RATE` | `16000` | Default rate when the client sends no format |
| `AUDIO_CHANNELS` | `1` | Default channel count |
| `AUDIO_MIN_SECONDS` | `0.25` | Shorter utterances are rejected as `too_short` |
| `AUDIO_MAX_SECONDS` | `30` | Buffer ceiling; exceeding it drops the utterance |
| `AUDIO_MAX_CHUNK_BYTES` | `1048576` | Largest accepted single frame |
| `WHISPER_MODEL` | `base` | Model size or HF repo: `tiny`, `base`, `small`, `medium`, `large-v3`, `…​.en` variants |
| `WHISPER_DEVICE` | `auto` | `cpu`, `cuda` or `auto` |
| `WHISPER_COMPUTE_TYPE` | `default` | e.g. `int8` on CPU, `float16` on GPU |
| `WHISPER_LANGUAGE` | `en` | Forces one language; empty value means auto-detect |
| `WHISPER_BEAM_SIZE` | `1` | Higher is more accurate and slower |
| `WHISPER_VAD_FILTER` | `true` | Drops silence before decoding |
| `WHISPER_DOWNLOAD_ROOT` | unset | Where model weights are cached |

A client may negotiate its own `sampleRate` (8–48 kHz) and `channels` (1–2) in
`start`; the limits above are then recalculated for that format.

The first utterance downloads the weights (a few hundred MB for `base`) unless
they are already in the Hugging Face cache, so the first reply after a cold
start is slow. On CPU, `WHISPER_COMPUTE_TYPE=int8` is considerably faster than
the default.

```powershell
$env:WHISPER_MODEL="small"; $env:WHISPER_COMPUTE_TYPE="int8"
```

## Layout

```
app/
  main.py       FastAPI app, /healthz, model + settings wiring
  ws_audio.py   stream_audio_session — the receiver
  audio.py      AudioBuffer: PCM accumulation and conversion
  model.py      AudioModel abstraction, FasterWhisperModel, EchoModel fallback
  config.py     AudioSettings + WhisperSettings (format, limits, model tuning)
tests/
  test_ws_audio.py   socket behaviour end to end
  test_audio.py      buffer and conversion unit tests
  test_model.py      faster-whisper plumbing, driven by a stand-in engine
client_example.py    streams a WAV file to the socket
```

## Notes

- Buffer state lives in memory per connection. One uvicorn worker is fine; with
  several workers a client must stay on the same process, which WebSockets do
  anyway for the lifetime of a connection.
- Audio is not persisted anywhere. Nothing is written to disk.
