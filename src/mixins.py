import asyncio


class AsyncStreamServiceMixin[T]:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[T] = asyncio.Queue()
