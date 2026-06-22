"""
TTS тест: Kokoro-82M и XTTS-v2 (Coqui).
Модели сохраняются в /home/fakhriddin3040/models/<model_...>

Запуск:
    python load_tts.py          — обе модели
    python load_tts.py kokoro   — только Kokoro
    python load_tts.py xtts     — только XTTS

Системные зависимости (Arch):
    sudo pacman -S espeak-ng ffmpeg
"""

import os
import sys
import time
from pathlib import Path

# XTTS-v2 распространяется под Coqui Public Model License — соглашаемся заранее,
# иначе TTS() зависнет на интерактивном запросе.
os.environ.setdefault("COQUI_TOS_AGREED", "1")

# ─── Пути к моделям ───────────────────────────────────────────────────────────

MODELS_ROOT = Path("/home/fakhriddin3040/models")
KOKORO_DIR = MODELS_ROOT / "model_kokoro_82m"
XTTS_DIR = MODELS_ROOT / "model_xtts_v2"

KOKORO_DIR.mkdir(parents=True, exist_ok=True)
XTTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DIR = Path("./tts_output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Нейтральный текст для синтеза (проверка просодии и интонации).
TEST_TEXT = "Hello! This is a generated audio test text."

# Для XTTS — референсный аудиофайл для клонирования голоса (мин. 6 сек чистой речи).
REFERENCE_AUDIO = "./reference_voice.wav"


def _device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


# ─── Kokoro-82M ───────────────────────────────────────────────────────────────


def run_kokoro() -> None:
    print("\n" + "=" * 60)
    print("  KOKORO-82M")
    print("=" * 60)

    try:
        import numpy as np
        import soundfile as sf
        from kokoro import KPipeline
    except ImportError:
        print("[ERROR] Не установлены зависимости: uv add kokoro soundfile")
        return

    os.environ["HF_HOME"] = str(KOKORO_DIR)
    print(f"[INFO] Модель будет в: {KOKORO_DIR}")

    t0 = time.time()
    pipeline = KPipeline(lang_code="a")  # 'a' = American English
    print(f"[INFO] Pipeline загружен за {time.time() - t0:.1f}с")

    # af_* — женские, am_* — мужские голоса (American English).
    for voice in ["af_bella", "am_adam"]:
        print(f"\n[Kokoro] Голос: {voice}")
        t0 = time.time()

        chunks = [audio for _, _, audio in pipeline(TEST_TEXT, voice=voice, speed=1.0)]
        audio_full = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]

        out_path = OUTPUT_DIR / f"kokoro_{voice}.wav"
        sf.write(str(out_path), audio_full, 24000)

        elapsed, duration = time.time() - t0, len(audio_full) / 24000
        print(f"[Kokoro] Готово: {out_path}")
        print(
            f"[Kokoro] {elapsed:.2f}с | аудио {duration:.2f}с | RTF {elapsed / duration:.2f}x"
        )


# ─── XTTS-v2 (Coqui) ──────────────────────────────────────────────────────────


def run_xtts() -> None:
    print("\n" + "=" * 60)
    print("  XTTS-v2 (Coqui)")
    print("=" * 60)

    try:
        from TTS.api import TTS
    except ImportError:
        print("[ERROR] Не установлены зависимости: uv add coqui-tts")
        raise
        return

    use_cloning = Path(REFERENCE_AUDIO).exists()
    if use_cloning:
        print(f"[INFO] Референс для клонирования: {REFERENCE_AUDIO}")
    else:
        raise
        print(f"[WARN] Референс не найден ({REFERENCE_AUDIO}) — встроенный голос.")

    os.environ["COQUI_TTS_HOME"] = str(XTTS_DIR)
    device = _device()
    print(f"[INFO] Загружаем XTTS-v2 на {device} (первый раз — скачает ~2GB)...")

    t0 = time.time()
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    print(f"[INFO] Модель загружена за {time.time() - t0:.1f}с")

    out_path = OUTPUT_DIR / "xtts_v2_output.wav"
    print(
        f"\n[XTTS] Синтез {'с клонированием' if use_cloning else 'встроенным голосом'}..."
    )
    t0 = time.time()

    if use_cloning:
        tts.tts_to_file(
            text=TEST_TEXT,
            speaker_wav=REFERENCE_AUDIO,
            language="en",
            file_path=str(out_path),
        )
    else:
        raise
        speaker = tts.speakers[0] if tts.speakers else None
        tts.tts_to_file(
            text=TEST_TEXT,
            speaker=speaker,
            language="en",
            file_path=str(out_path),
        )

    print(f"[XTTS] Готово: {out_path} (за {time.time() - t0:.2f}с)")


# ─── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    args = sys.argv[1:]
    print("Директории моделей:")
    print(f"  Kokoro : {KOKORO_DIR}")
    print(f"  XTTS   : {XTTS_DIR}")
    print(f"Выходные файлы: {OUTPUT_DIR.resolve()}")

    if not args or "kokoro" in args:
        run_kokoro()
    if not args or "xtts" in args:
        run_xtts()

    print("\n✅ Готово. Файлы в ./tts_output/")


if __name__ == "__main__":
    main()
