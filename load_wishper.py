"""
STT тест: faster-whisper (large-v3-turbo) через CTranslate2.

Запуск:
    python load_wishper.py [audio.mp3]

GPU (RTX 3080) используется автоматически, если доступен. На GPU нужен cuDNN 9 —
он ставится вместе с torch (nvidia-cudnn-cu12). Если CTranslate2 не находит cuDNN,
скрипт сам падает обратно на CPU (int8), который работает всегда.
"""

import sys
from pathlib import Path

from faster_whisper import WhisperModel

MODEL_SIZE = "large-v3-turbo"
DOWNLOAD_ROOT = "/home/fakhriddin3040/models/wishper-large-v3-turbo"
AUDIO = sys.argv[1] if len(sys.argv) > 1 else "audio.mp3"


def _pick_device() -> tuple[str, str]:
    """(device, compute_type): float16 на GPU, int8 на CPU."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"


def load_model() -> WhisperModel:
    device, compute_type = _pick_device()
    print(f"[INFO] Загружаем {MODEL_SIZE} на {device} ({compute_type})...")
    try:
        return WhisperModel(
            MODEL_SIZE,
            device=device,
            compute_type=compute_type,
            download_root=DOWNLOAD_ROOT,
        )
    except Exception as exc:  # cuDNN/cuBLAS не найдены → CPU
        if device == "cuda":
            print(f"[WARN] GPU недоступен для CTranslate2 ({exc}). Падаем на CPU int8.")
            return WhisperModel(
                MODEL_SIZE,
                device="cpu",
                compute_type="int8",
                download_root=DOWNLOAD_ROOT,
            )
        raise


def main() -> None:
    if not Path(AUDIO).exists():
        print(f"[ERROR] Аудиофайл не найден: {AUDIO}")
        print("        Передай путь: python load_wishper.py /path/to/audio.wav")
        sys.exit(1)

    model = load_model()

    segments, info = model.transcribe(AUDIO, beam_size=5, language="en")
    print(f"[INFO] Язык: {info.language} (p={info.language_probability:.2f})")
    for segment in segments:
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")


if __name__ == "__main__":
    main()
