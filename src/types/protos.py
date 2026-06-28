from typing import (
    Iterable,
    Protocol,
    runtime_checkable,
)

from src.types import FloatArray, TokenizerOutput


@runtime_checkable
class SpeechToTextServiceProto(Protocol):
    def convert(self, speech: FloatArray) -> str:
        """
        Converts speech to text.

        Args:
            speech (Iterable[FloatArray]): speech

        Raises:
            ?

        Returns:
            Whole text
        """


@runtime_checkable
class TextToSpeechServiceProto(Protocol):
    def convert(self, text: str) -> Iterable[FloatArray]:
        """
        Converts text to speech

        Args:
             text (str): Text to convert to speech

        Raises:
            ?

        Returns:
            Stream of speech
        """


@runtime_checkable
class LLMServiceProto(Protocol):
    def ask(self, text: str) -> Iterable[str]:
        """Ask LLM

        Args:
            text (str): Text to ask

        Raises:
            ?
        """


@runtime_checkable
class TextTokenizerProto[TOut: TokenizerOutput](Protocol):
    def tokenize(self, text: str) -> TOut:
        """Converts text to tokens

        Args:
            text (str): Text for tokenization

        Raises:
            ?
        """
