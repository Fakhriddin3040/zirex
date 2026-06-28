from faster_whisper import WhisperModel
from kokoro import KPipeline
from typing import Iterable

from transformers import PreTrainedTokenizerFast

from src.types.protos import (
    FloatArray,
    TextToSpeechServiceProto,
    SpeechToTextServiceProto,
)
from src.utils.type_checking import ensure_isimplementation
from types import TokenizerOutput


class WhisperSpeechToTextService:
    def __init__(self) -> None:
        self._model = WhisperModel(
            "large-v3-turbo",
            device="cuda",
            compute_type="float16",
            download_root="/home/fakhriddin3040/models/wishper-large-v3-turbo",
        )

    def convert(self, speech: FloatArray) -> str:
        segments, _ = self._model.transcribe(
            speech,
            language="en",
            beam_size=5,
            no_speech_threshold=0.6,
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            compression_ratio_threshold=2.4,
            condition_on_previous_text=False,
            log_prob_threshold=1,
        )
        return "".join(seg.text for seg in segments)


class KokoroTextToSpeechService:
    def __init__(self) -> None:
        self._pipeline = KPipeline(lang_code="a", device="cuda")
        self._voice = "am_adam"

    def convert(self, stream: str) -> Iterable[FloatArray]:
        """
        Converts text to speech using text stream

        Args:
            stream (str): Part of text to convert to speech

        Raises:
            ?

        Returns:
            Stream of speech
        """
        for text in stream:
            for res in self._pipeline(text=text, voice=self._voice):
                yield res.audio


class TextTokenizer:
    def __init__(self) -> None:
        self._tokenizer = PreTrainedTokenizerFast(
            "/home/fakhriddin3040/models/llama-3.1-8b"
        )

    def tokenize(self, text: str) -> TokenizerOutput:
        """Converts text to tokens

        Args:
            text (str): Text for tokenization

        Raises:
            ?
        """
        return self._tokenizer.encode(text)


class Llama3D1Service:
    def __init__(self) -> None: ...

    def ask(self, text: str) -> Iterable[str]:
        """Ask LLM

        Args:
            text (str): Text to ask

        Raises:
            ?
        """


ensure_isimplementation(KokoroTextToSpeechService, TextToSpeechServiceProto)
ensure_isimplementation(WhisperSpeechToTextService, SpeechToTextServiceProto)
