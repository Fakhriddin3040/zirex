"""
Тест обученной модели: грузит базу + LoRA-адаптер, даёт интерактивный чат
в терминале, чтобы быстро проверить как модель усвоила стиль.

Запуск:
    python test_llama.py
"""

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

# ─────────────────────────────────────────
# Конфигурация — поправь под себя
# ─────────────────────────────────────────
MAX_SEQ_LENGTH = 10240

# ─────────────────────────────────────────
# Загрузка базы + адаптера одним вызовом
# Unsloth сам подхватит base model из adapter_config.json внутри папки адаптера
# ─────────────────────────────────────────
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/home/fakhriddin3040/models/llama-3.1-8b-bnb-4bit",
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)

tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")
FastLanguageModel.for_inference(model)

# ─────────────────────────────────────────
# Интерактивный чат-цикл
# ─────────────────────────────────────────
print("Тест обученной модели. Пустая строка или Ctrl+C — выход.\n")

history = []

while True:
    try:
        user_input = input("Ты: ").strip()
    except KeyboardInterrupt, EOFError:
        print("\nВыход.")
        break

    if not user_input:
        break

    history.append({"role": "user", "content": user_input})

    inputs = tokenizer.apply_chat_template(
        history,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    out = model.generate(
        input_ids=inputs,
        max_new_tokens=10240,
        temperature=1,
        do_sample=True,
    )

    response = tokenizer.decode(out[0][inputs.shape[1] :], skip_special_tokens=True)
    print(f"Модель: {response}\n")

    history.append({"role": "assistant", "content": response})
