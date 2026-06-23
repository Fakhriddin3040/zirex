from faster_whisper import WhisperModel
from kokoro import KPipeline
from typing import Iterator


from src.types.protos import (
    FloatArray,
    TextToSpeechServiceProto,
    SpeechToTextServiceProto,
)
from src.utils.type_checking import ensure_isimplementation


class KokoroTextToSpeechService:
    def __init__(self) -> None:
        self._pipeline = KPipeline(lang_code="a", device="cuda")
        self._voice = "am_adam"

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
        for text in stream:
            for res in self._pipeline(text=text, voice=self._voice):
                yield res.audio


class WhisperSpeechToTextService:
    def __init__(self) -> None:
        self._model = WhisperModel(
            "large-v3-turbo",
            device="cuda",
            compute_type="float16",
            download_root="/home/fakhriddin3040/models/wishper-large-v3-turbo",
        )

    def convert(self, stream: Iterator[FloatArray]) -> Iterator[str]:
        for part in stream:
            segments, _ = self._model.transcribe(audio=part, language="en", beam_size=5)
            for segment in segments:
                yield segment.text


ensure_isimplementation(KokoroTextToSpeechService, TextToSpeechServiceProto)
ensure_isimplementation(WhisperSpeechToTextService, SpeechToTextServiceProto)
