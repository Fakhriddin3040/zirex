import asyncio
from typing import AsyncIterator, Generic, Iterator, TypeVar

from src.types import FloatArray
from src.types.protos import (
    AsyncStreamServiceProto,
    SpeechToTextServiceProto,
    TextToSpeechServiceProto,
)
from src.utils.type_checking import ensure_isimplementation

TIn = TypeVar("TIn")
TOut = TypeVar("TOut")

_OUTPUT_END = object()


class _SyncStreamService(Generic[TIn, TOut]):
    """Minimal structural type for the wrapped sync service."""

    def convert(self, stream: Iterator[TIn]) -> Iterator[TOut]: ...


class QueueAsyncStreamService(Generic[TIn, TOut]):
    """
    Adapts a synchronous ``convert(Iterator[TIn]) -> Iterator[TOut]`` service into
    a queue-based async pipeline.

    Each item handed to ``put``/``feed`` is run through ``convert`` on a worker
    thread (via ``asyncio.to_thread``) so the blocking model call never blocks
    the event loop; its output chunks land on an ``asyncio.Queue`` that's drained
    by iterating the instance::

        svc = QueueAsyncStreamService[str, FloatArray](KokoroTextToSpeechService())

        async def produce():
            await svc.feed(text_stream())
            svc.close()

        async def consume():
            async for audio in svc:
                ...
    """

    def __init__(
        self, service: _SyncStreamService[TIn, TOut], *, maxsize: int = 0
    ) -> None:
        self._service = service
        self._queue: "asyncio.Queue[object]" = asyncio.Queue(maxsize=maxsize)
        self._closed = False

    def _convert_one(self, item: TIn) -> list[TOut]:
        return list(self._service.convert(iter([item])))

    async def put(self, item: TIn) -> None:
        if self._closed:
            raise RuntimeError("put() after close()")
        for chunk in await asyncio.to_thread(self._convert_one, item):
            await self._queue.put(chunk)

    async def feed(self, stream: AsyncIterator[TIn]) -> None:
        async for item in stream:
            await self.put(item)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put_nowait(_OUTPUT_END)

    async def __aiter__(self) -> AsyncIterator[TOut]:
        while True:
            item = await self._queue.get()
            if item is _OUTPUT_END:
                return
            yield item


class AsyncTextToSpeechService(QueueAsyncStreamService[str, FloatArray]):
    def __init__(self, service: TextToSpeechServiceProto, **kwargs) -> None:
        super().__init__(service, **kwargs)


class AsyncSpeechToTextService(QueueAsyncStreamService[FloatArray, str]):
    def __init__(self, service: SpeechToTextServiceProto, **kwargs) -> None:
        super().__init__(service, **kwargs)


ensure_isimplementation(QueueAsyncStreamService, AsyncStreamServiceProto)
