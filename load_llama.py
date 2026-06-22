"""
Llama 3.1 (unsloth) — заготовка.

Пока только проверяет окружение: unsloth/torch импортируются, CUDA видна.
Реальная логика инференса и LoRA-дообучения — позже (см. TODO ниже).

Запуск:
    python load_llama.py
"""

MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
MODELS_DIR = "/home/fakhriddin3040/models/llama-3.1-8b"
MAX_SEQ_LEN = 2048


def check_env() -> None:
    import torch

    print(f"[INFO] torch {torch.__version__}")
    print(f"[INFO] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

    import unsloth  # noqa: F401  — тяжёлый импорт, проверяем что патчи встают

    print("[INFO] unsloth импортирован — окружение готово.")


def load_model():
    """TODO (инференс): загрузка 4-bit модели через unsloth.

    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
        cache_dir=MODELS_DIR,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer
    """
    raise NotImplementedError("Инференс ещё не реализован")


def finetune():
    """TODO (дообучение): LoRA через unsloth + trl SFTTrainer.

    model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=16, ...)
    trainer = SFTTrainer(model=model, train_dataset=..., ...)
    trainer.train()
    """
    raise NotImplementedError("Дообучение ещё не реализовано")


if __name__ == "__main__":
    check_env()
