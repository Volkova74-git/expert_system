import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Папка с исходными JSON (на уровень выше, чем папка benchmark)
STANDARDS_DIR = Path("standards/processed")
OUTPUT_DIR = Path("benchmark/embeddings")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Модели для сравнения
MODELS = {
    "e5-small": "intfloat/multilingual-e5-small",
    "e5-large": "intfloat/multilingual-e5-large",
    "rosberta": "ai-forever/ru-en-RoSBERTa",
}

def load_chunks_from_standards():
    """Проходит по всем JSON, собирает текст страниц, разбивает на чанки."""
    chunks = {}
    json_files = list(STANDARDS_DIR.glob("*.json"))
    print(f"Найдено {len(json_files)} JSON-файлов в {STANDARDS_DIR}")

    for json_path in tqdm(json_files, desc="Обработка файлов"):
        with open(json_path, 'r', encoding='utf-8') as f:
            doc = json.load(f)

        # Склеиваем содержимое всех страниц
        full_text = ""
        if "pages" in doc and isinstance(doc["pages"], list):
            for page in doc["pages"]:
                full_text += page.get("page_content", "") + "\n\n"
        else:
            # если структура неожиданная – пропускаем файл
            continue

        if not full_text.strip():
            continue

        # Разбиваем на чанки по двойному переносу строки (как в index_to_elasticsearch.py)
        raw_chunks = full_text.split('\n\n')
        doc_name = json_path.stem  # например, "ГОСТ 31937-2024"
        for i, chunk_text in enumerate(raw_chunks):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue
            chunk_id = f"{doc_name}_{i}"
            chunks[chunk_id] = chunk_text

    print(f"Всего получено чанков: {len(chunks)}")
    return chunks

def generate_embeddings_for_model(model_key, hf_name, chunks_dict):
    print(f"\n--- Генерация для модели: {model_key} ({hf_name}) ---")
    model = SentenceTransformer(hf_name)
    dim = model.get_sentence_embedding_dimension()
    print(f"Размерность: {dim}")

    embeddings = {}
    for chunk_id, text in tqdm(chunks_dict.items(), desc="Генерация эмбеддингов"):
        # Для E5 нужно добавить префикс passage:
        if "e5" in model_key:
            text = f"passage: {text}"
        embeddings[chunk_id] = model.encode(text)

    out_file = OUTPUT_DIR / f"chunk_embeddings_{model_key}.npy"
    np.save(out_file, embeddings)
    print(f"Сохранено в {out_file}")

def main():
    chunks = load_chunks_from_standards()
    if not chunks:
        print("Не найдено ни одного чанка. Проверьте структуру JSON-файлов в standards/processed/")
        return

    for model_key, hf_name in MODELS.items():
        generate_embeddings_for_model(model_key, hf_name, chunks)

    print("\nГотово! Теперь можно запускать сравнение: python benchmark/benchmark_models.py")

if __name__ == "__main__":
    main()