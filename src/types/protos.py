from typing import (
    AsyncIterator,
    Iterator,
    Protocol,
    runtime_checkable,
)

from src.types import FloatArray


@runtime_checkable
class SpeechToTextServiceProto(Protocol):
    def convert(self, stream: Iterator[FloatArray]) -> Iterator[str]:
        """
        Converts speech to text using streams.

        Args:
            stream (Iterator[FloatArray]): Speech stream

        Raises:
            ?

        Returns:
            Stream of text
        """


@runtime_checkable
class TextToSpeechServiceProto(Protocol):
    def convert(self, stream: Iterator[str]) -> Iterator[FloatArray]:
        """
        Converts text to speech using text stream

        Args:
            stream (Iterator[str]): Part of text to convert to speech

        Raises:
            ?

        Returns:
            Stream of speech
        """


@runtime_checkable
class AsyncStreamServiceProto[TIn, TOut](Protocol):
    """
    Queue-based async wrapper around a synchronous stream service.

    Producers push items with ``put``/``feed`` from the event loop; the blocking
    conversion runs off-loop and results are consumed by iterating the instance.
    """

    async def put(self, item: TIn) -> None:
        """Enqueue a single input item for conversion."""

    async def feed(self, stream: AsyncIterator[TIn]) -> None:
        """Drain an async input stream into the conversion queue."""

    def close(self) -> None:
        """Signal end-of-input. No further ``put``/``feed`` is allowed."""

    def __aiter__(self) -> AsyncIterator[TOut]:
        """Iterate converted output items as they become available."""


AsyncSpeechToTextServiceProto = AsyncStreamServiceProto[FloatArray, str]
AsyncTextToSpeechServiceProto = AsyncStreamServiceProto[str, FloatArray]
