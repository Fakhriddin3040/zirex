import asyncio
from asyncio import Queue
from typing import NoReturn, Tuple
import sounddevice as sd

from src.components.async_streams import (
    AsyncTextToSpeechService,
    AsyncSpeechToTextService,
)
from src.components.services import (
    KokoroTextToSpeechService,
    WhisperSpeechToTextService,
)
from src.types import FloatArray


async def listen_micro() -> Tuple[sd.InputStream, Queue]:
    q = Queue()

    def callback(indata: FloatArray, frames, time, status) -> None:
        q.put(indata.copy().flatten())

    stream = sd.InputStream(
        samplerate=16000, channels=1, dtype="float32", blocksize=1024, callback=callback
    )

    return stream, q


async def main() -> NoReturn:
    stt = AsyncSpeechToTextService(WhisperSpeechToTextService())
    tts = AsyncTextToSpeechService(KokoroTextToSpeechService())

    stream, q = listen_micro()

    def open_stream() -> NoReturn:
        with stream:
            sd.sleep(5000)

    await asyncio.gather(asyncio.to_thread(open_stream), stt.feed(q), tts.feed(stt))
    print("finished")


if __name__ == "__main__":
    asyncio.run(main())
