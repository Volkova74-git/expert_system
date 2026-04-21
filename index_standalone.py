import json, pickle, os, numpy as np
from gigachat import GigaChat
from gigachat.models import Embeddings
from sentence_splitter import SentenceSplitter
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# ========== НАСТРОЙКИ ==========
JSON_FILE = "standards/test_standard.json"   # ИЗМЕНИТЕ ПУТЬ, ЕСЛИ НУЖНО
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
# ===============================

def load_giga():
    """Инициализация клиента GigaChat"""
    return GigaChat(
        #credentials="",
        credentials=os.getenv("GIGACHAT_CREDENTIALS"),
        verify_ssl_certs=False,
        scope="GIGACHAT_API_PERS",
        model="GigaChat-Pro",
        auth_url="https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
        api_base="https://gigachat.devices.sberbank.ru/api/v1",
        timeout=60
    )

def split_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    splitter = SentenceSplitter(language="ru")
    sentences = splitter.split(text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= size:
            current_chunk += " " + sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            overlap_text = " ".join(current_chunk.split()[-overlap:]) if overlap else ""
            current_chunk = overlap_text + " " + sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    chunks = [ch for ch in chunks if len(ch) >= 50]
    return chunks

def main():
    print("1. Загрузка JSON...")
    with open(JSON_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Извлечение текста
    if 'pages' in data:
        full_text = "\n".join([page.get('page_content', '') for page in data['pages']])
    elif 'text' in data:
        full_text = data['text']
    elif 'content' in data:
        full_text = data['content']
    else:
        print(f"Ошибка: нет полей 'pages'/'text'/'content'. Доступны: {list(data.keys())}")
        return

    print(f"   Текст загружен, длина: {len(full_text)} символов")
    print("2. Разбиение на чанки...")
    chunks = split_text(full_text)
    print(f"   Получено {len(chunks)} чанков")
    print("3. Генерация эмбедингов...")
    giga = load_giga()
    all_embeddings = []
    for i, chunk in enumerate(tqdm(chunks, desc="Эмбеддинги")):
        try:
            # ПРАВИЛЬНЫЙ ВЫЗОВ ДЛЯ НОВОЙ ВЕРСИИ
            response = giga.embeddings([chunk])
            emb = np.array(response.data[0].embedding, dtype=np.float32)
            all_embeddings.append(emb)
        except Exception as e:
            print(f"Ошибка на чанке {i}: {e}")
            all_embeddings.append(np.zeros(1024, dtype=np.float32))

    print("4. Сохранение индекса...")
    index = np.array(all_embeddings)
    with open("standards_index.faiss", "wb") as f:
        np.save(f, index)
    with open("standards_texts.pkl", "wb") as f:
        pickle.dump(chunks, f)
    print("✅ Готово! Файлы standards_index.faiss и standards_texts.pkl созданы.")

if __name__ == "__main__":
    main()