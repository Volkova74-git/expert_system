import streamlit as st
import numpy as np
import pickle
import io
import os
from elasticsearch import Elasticsearch
from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from dotenv import load_dotenv

load_dotenv()  # загружаем переменные из .env

# ========== НАСТРОЙКИ ==========
ES_HOST = "http://127.0.0.1:9200"
INDEX_NAME = "construction_standards"
TOP_K = 5



# ---------- Функции для работы с GigaChat и Elasticsearch ----------
@st.cache_resource
def load_giga():
    """Инициализация клиента GigaChat (кэшируется)"""

    return GigaChat(
        credentials=os.getenv("GIGACHAT_CREDENTIALS", ""),
        verify_ssl_certs=False,
        scope="GIGACHAT_API_PERS",
        model="GigaChat-Pro",
        auth_url="https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
        api_base="https://gigachat.devices.sberbank.ru/api/v1",
        timeout=60
    )


@st.cache_resource
def load_elasticsearch():
    """Подключение к Elasticsearch (кэшируется)"""
    return Elasticsearch(ES_HOST)


@st.cache_resource
def get_available_documents(es, index_name):
    """Получить список уникальных названий нормативов (doc_name) для фильтрации"""
    body = {
        "size": 0,
        "aggs": {
            "unique_docs": {
                "terms": {"field": "doc_name.keyword", "size": 100}
            }
        }
    }
    resp = es.search(index=index_name, body=body)
    buckets = resp['aggregations']['unique_docs']['buckets']
    return [b['key'] for b in buckets]


def get_embedding(giga, text: str):
    """Получить эмбеддинг для текста через GigaChat"""
    response = giga.embeddings([text])
    return np.array(response.data[0].embedding, dtype=np.float32).tolist()


def find_similar(query_text: str, giga, es, index_name: str, top_k: int = TOP_K, doc_filter: str = None):
    """Поиск релевантных чанков через Elasticsearch с нормализованной оценкой"""
    query_vector = get_embedding(giga, query_text)

    # Базовый запрос (находит все документы)
    base_query = {"match_all": {}}
    if doc_filter and doc_filter != "Все":
        base_query = {"term": {"doc_name.keyword": doc_filter}}

    body = {
        "size": top_k,
        "query": {
            "script_score": {
                "query": base_query,
                "script": {
                    # Нормализуем оценку в диапазон [0, 1]
                    "source": "(cosineSimilarity(params.query_vector, 'vector') + 1.0) / 2.0",
                    "params": {"query_vector": query_vector}
                }
            }
        }
    }
    response = es.search(index=index_name, body=body)

    results = []
    for hit in response['hits']['hits']:
        results.append({
            "text": hit['_source']['chunk_text'],
            "score": hit['_score'],  # уже в диапазоне [0,1]
            "chunk_index": hit['_source'].get('chunk_index', int(hit['_id'])),
            "doc_name": hit['_source'].get('doc_name', 'Неизвестный документ')
        })
    return results


# ---------- Генерация PDF ----------
def generate_report(defect_description, chosen_chunks_with_names, analysis_text):
    """Генерирует PDF-отчет с результатами, включая названия нормативов"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
    styles = getSampleStyleSheet()

    style_title = ParagraphStyle('Title', parent=styles['Title'], fontSize=16, alignment=TA_LEFT, spaceAfter=12)
    style_heading = ParagraphStyle('Heading', parent=styles['Heading2'], fontSize=12, alignment=TA_LEFT, spaceAfter=6,
                                   spaceBefore=12)
    style_body = ParagraphStyle('Body', parent=styles['Normal'], fontSize=10, alignment=TA_JUSTIFY, leading=14,
                                spaceAfter=12)

    story = []
    story.append(Paragraph("Отчет по результатам анализа дефекта", style_title))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Описание дефекта:", style_heading))
    story.append(Paragraph(defect_description, style_body))

    story.append(Paragraph("Выбранные пункты нормативов:", style_heading))
    for i, (doc_name, chunk_text) in enumerate(chosen_chunks_with_names, 1):
        short_chunk = chunk_text[:500] + "..." if len(chunk_text) > 500 else chunk_text
        story.append(Paragraph(f"<b>{i}. {doc_name}</b><br/>{short_chunk}", style_body))

    story.append(Paragraph("Заключение и рекомендации:", style_heading))
    story.append(Paragraph(analysis_text.replace('\n', '<br/>'), style_body))

    doc.build(story)
    return buffer.getvalue()


# ---------- Интерфейс Streamlit ----------
st.set_page_config(page_title="Поиск по строительным нормативам", layout="wide")
st.title("🏗️ Поиск и анализ дефектов по строительным нормативам")

# Проверка подключения к Elasticsearch
try:
    es = load_elasticsearch()
    if not es.ping():
        st.error("❌ Не удалось подключиться к Elasticsearch. Убедитесь, что контейнер запущен (docker-compose up -d).")
        st.stop()
    if not es.indices.exists(index=INDEX_NAME):
        st.error(f"❌ Индекс '{INDEX_NAME}' не найден. Сначала запустите python index_to_elasticsearch.py")
        st.stop()
except Exception as e:
    st.error(f"Ошибка подключения к Elasticsearch: {e}")
    st.stop()

# Загрузка клиента GigaChat
try:
    giga = load_giga()
except Exception as e:
    st.error(f"Ошибка инициализации GigaChat: {e}")
    st.stop()

# Получение списка доступных нормативов для фильтрации
try:
    doc_list = get_available_documents(es, INDEX_NAME)
    doc_list.insert(0, "Все")  # добавляем опцию "Все"
except:
    doc_list = ["Все"]

# Инициализация состояния сессии
if 'selected_chunks' not in st.session_state:
    st.session_state.selected_chunks = []  # храним кортежи (doc_name, text)
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'defect_description' not in st.session_state:
    st.session_state.defect_description = ""

# Две колонки: фото и описание
col1, col2 = st.columns(2)
with col1:
    uploaded_files = st.file_uploader("📸 Загрузите фото дефектов", type=["jpg", "png", "jpeg"],
                                      accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        st.image(uploaded_file, width=200, caption=uploaded_file.name)

with col2:
    defect_description = st.text_area("📝 Опишите дефект", height=200,
                                      placeholder="Пример: На бетонной поверхности обнаружены трещины шириной до 2 мм...")
    # Выбор норматива (фильтр)
    selected_doc_filter = st.selectbox("📄 Фильтр по нормативу", doc_list, index=0)
    search_button = st.button("🔍 Найти пункты в нормативах")

# Поиск
if search_button and defect_description:
    st.session_state.defect_description = defect_description
    with st.spinner("Поиск релевантных пунктов через Elasticsearch..."):
        results = find_similar(defect_description, giga, es, INDEX_NAME, top_k=TOP_K, doc_filter=selected_doc_filter)

    if results:
        st.subheader("📚 Найденные пункты нормативов")
        st.session_state.search_results = results
        st.session_state.selected_chunks = []

        for i, res in enumerate(results):
            # Отображаем название документа и нормализованный процент
            with st.expander(f"**{res['doc_name']}** (Совпадение: {res['score']:.2%})"):
                st.write(res['text'])
                if st.button(f"Выбрать этот пункт", key=f"btn_{i}"):
                    # Сохраняем и название, и текст
                    if (res['doc_name'], res['text']) not in st.session_state.selected_chunks:
                        st.session_state.selected_chunks.append((res['doc_name'], res['text']))
                        st.success(f"Пункт из {res['doc_name']} добавлен для анализа")
    else:
        st.warning("По вашему запросу ничего не найдено. Попробуйте изменить описание.")

# Отображение выбранных пунктов
if st.session_state.selected_chunks:
    st.subheader(f"✅ Выбранные пункты для анализа ({len(st.session_state.selected_chunks)})")
    for i, (doc_name, chunk_text) in enumerate(st.session_state.selected_chunks):
        st.text_area(f"Пункт {i + 1} — {doc_name}", chunk_text[:300], height=100, key=f"selected_{i}")

    if st.button("🗑️ Очистить список"):
        st.session_state.selected_chunks = []
        st.rerun()

# Финальный анализ и отчёт
if st.session_state.selected_chunks and st.button("🤖 Проанализировать и создать отчёт"):
    with st.spinner("GigaChat анализирует дефекты и готовит рекомендации..."):
        # Формируем промпт с указанием названий нормативов
        normatives_text = "\n".join(
            [f"[{doc_name}] {chunk_text}" for doc_name, chunk_text in st.session_state.selected_chunks])
        prompt = f"""
        Ты — эксперт по строительным нормам и правилам.

        Задача: Проанализировать описание дефекта строительной конструкции и выбранные пункты нормативов.

        Описание дефекта:
        {st.session_state.defect_description}

        Выбранные пункты нормативов (с указанием документа):
        {normatives_text}

        Твой ответ должен содержать три четкие секции:
        1. **Анализ дефекта:** На основе нормативов опиши, почему данный дефект является нарушением.
        2. **Рекомендации по устранению:** Конкретные шаги для исправления дефекта, ссылаясь на нормативы.
        3. **Вывод о состоянии конструкции:** Общее заключение (например, "Конструкция находится в ограниченно работоспособном состоянии").

        Ответ должен быть подробным, структурированным и написанным профессиональным русским языком.
        """

        messages = [Messages(role=MessagesRole.USER, content=prompt)]
        chat_payload = Chat(messages=messages)
        response = giga.chat(chat_payload)
        analysis_text = response.choices[0].message.content

        st.session_state.analysis_text = analysis_text
        st.session_state.analysis_done = True

# Результат и скачивание PDF
if st.session_state.get('analysis_done', False):
    st.subheader("📊 Результат анализа")
    st.markdown(st.session_state.analysis_text)

    # Передаём в генератор PDF список кортежей (doc_name, chunk_text)
    pdf_data = generate_report(
        st.session_state.defect_description,
        st.session_state.selected_chunks,
        st.session_state.analysis_text
    )

    st.download_button(
        label="📥 Скачать отчёт в PDF",
        data=pdf_data,
        file_name="construction_defect_report.pdf",
        mime="application/pdf"
    )

    if st.button("🔄 Новый анализ"):
        st.session_state.selected_chunks = []
        st.session_state.analysis_done = False
        st.rerun()