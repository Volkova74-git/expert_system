import streamlit as st
import numpy as np
import io
import os
import base64
import tempfile
from elasticsearch import Elasticsearch
from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from dotenv import load_dotenv

load_dotenv()

# ========== НАСТРОЙКИ ==========
ES_HOST = "http://127.0.0.1:9200"
INDEX_NAME = "construction_standards"
TOP_K = 5


# ===============================

# ---------- Регистрация кириллического шрифта ----------
def register_russian_font():
    """Пытается зарегистрировать шрифт с поддержкой кириллицы."""
    possible_fonts = [
        "C:/Windows/Fonts/arial.ttf",  # Windows
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
        "/System/Library/Fonts/Arial.ttf",  # macOS
        os.path.join(os.path.dirname(__file__), "fonts", "DejaVuSerif.ttf")
    ]
    for font_path in possible_fonts:
        if os.path.exists(font_path):
            pdfmetrics.registerFont(TTFont('RussianFont', font_path))
            return 'RussianFont'
    # Если не нашли, возвращаем стандартный (кириллица не отобразится)
    return 'Helvetica'


RUSSIAN_FONT = register_russian_font()

# ---------- CSS стили ----------
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    text-align: center;
    color: #1565C0;
    margin-bottom: 2rem;
    padding: 1rem;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.section-header {
    font-size: 1.5rem;
    font-weight: 600;
    color: #0D47A1;
    margin-top: 1rem;
    margin-bottom: 1rem;
    border-left: 4px solid #1565C0;
    padding-left: 1rem;
}
.result-card {
    background-color: #f8f9fa;
    border-radius: 10px;
    padding: 15px;
    margin: 10px 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    border-left: 4px solid #1565C0;
}
.selected-card {
    background-color: #e8f5e9;
    border-left-color: #4caf50;
}
.image-container {
    display: flex;
    flex-wrap: wrap;
    gap: 15px;
}
.image-item {
    position: relative;
    width: calc(33% - 10px);
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}
.image-item img {
    width: 100%;
    height: 150px;
    object-fit: cover;
}
.defect-card {
    background: #f0f2f6;
    border-radius: 15px;
    padding: 15px;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)


# ---------- Функции для работы с GigaChat и Elasticsearch ----------
@st.cache_resource
def load_giga():
    return GigaChat(
        credentials=os.getenv("GIGACHAT_CREDENTIALS",
                              ""),
        verify_ssl_certs=False,
        scope="GIGACHAT_API_PERS",
        model="GigaChat-Pro",
        auth_url="https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
        api_base="https://gigachat.devices.sberbank.ru/api/v1",
        timeout=60
    )


@st.cache_resource
def load_elasticsearch():
    return Elasticsearch(ES_HOST)


@st.cache_resource
def get_available_documents(es, index_name):
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
    response = giga.embeddings([text])
    return np.array(response.data[0].embedding, dtype=np.float32).tolist()


def find_similar(query_text, giga, es, index_name, top_k=TOP_K, doc_filter=None):
    query_vector = get_embedding(giga, query_text)
    base_query = {"match_all": {}}
    if doc_filter and doc_filter != "Все":
        base_query = {"term": {"doc_name.keyword": doc_filter}}
    body = {
        "size": top_k,
        "query": {
            "script_score": {
                "query": base_query,
                "script": {
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
            "score": hit['_score'],
            "chunk_index": hit['_source'].get('chunk_index', 0),
            "doc_name": hit['_source'].get('doc_name', 'Неизвестный документ'),
            "doc_id": hit['_id']
        })
    return results


def generate_report(defects_data, analysis_text):
    """
    defects_data: список словарей, каждый содержит:
        - photo_filename: str (имя файла)
        - description: str
        - selected_chunks: list of (doc_name, chunk_text)
    analysis_text: строка с общим анализом от GigaChat
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
    styles = getSampleStyleSheet()

    style_title = ParagraphStyle('Title', parent=styles['Title'], fontSize=16, fontName=RUSSIAN_FONT, alignment=TA_LEFT,
                                 spaceAfter=12)
    style_heading = ParagraphStyle('Heading', parent=styles['Heading2'], fontSize=12, fontName=RUSSIAN_FONT,
                                   alignment=TA_LEFT, spaceAfter=6, spaceBefore=12)
    style_body = ParagraphStyle('Body', parent=styles['Normal'], fontSize=10, fontName=RUSSIAN_FONT,
                                alignment=TA_JUSTIFY, leading=14, spaceAfter=12)

    story = []
    story.append(Paragraph("Общий отчет по результатам анализа дефектов", style_title))
    story.append(Spacer(1, 12))

    for idx, defect in enumerate(defects_data, 1):
        story.append(Paragraph(f"Дефект №{idx}: {defect['photo_filename']}", style_heading))
        story.append(Paragraph(f"Описание: {defect['description']}", style_body))
        if defect['selected_chunks']:
            story.append(Paragraph("Выбранные пункты нормативов:", style_heading))
            for i, (doc_name, chunk_text) in enumerate(defect['selected_chunks'], 1):
                short_chunk = chunk_text[:300] + "..." if len(chunk_text) > 300 else chunk_text
                story.append(Paragraph(f"{i}. {doc_name}: {short_chunk}", style_body))
        else:
            story.append(Paragraph("Нормативные пункты не выбраны.", style_body))
        story.append(Spacer(1, 12))

    story.append(Paragraph("Общее заключение и рекомендации", style_heading))
    story.append(Paragraph(analysis_text.replace('\n', '<br/>'), style_body))

    doc.build(story)
    return buffer.getvalue()


# ---------- Интерфейс Streamlit ----------
st.markdown('<div class="main-header">🏗️ Поиск и анализ дефектов по строительным нормативам</div>',
            unsafe_allow_html=True)

# Подключение к Elasticsearch
try:
    es = load_elasticsearch()
    if not es.ping():
        st.error("❌ Не удалось подключиться к Elasticsearch. Убедитесь, что контейнер запущен (docker-compose up -d).")
        st.stop()
    if not es.indices.exists(index=INDEX_NAME):
        st.error(f"❌ Индекс '{INDEX_NAME}' не найден. Сначала запустите batch_index.py")
        st.stop()
    st.success("✅ Подключение к Elasticsearch успешно")
except Exception as e:
    st.error(f"Ошибка Elasticsearch: {e}")
    st.stop()

# Подключение к GigaChat
try:
    giga = load_giga()
    st.success("✅ Подключение к GigaChat успешно")
except Exception as e:
    st.error(f"Ошибка GigaChat: {e}")
    st.stop()

# Список нормативов для фильтра
try:
    doc_list = get_available_documents(es, INDEX_NAME)
    doc_list.insert(0, "Все")
except:
    doc_list = ["Все"]

# Инициализация сессии
if 'defects' not in st.session_state:
    st.session_state.defects = []  # список дефектов
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'final_analysis' not in st.session_state:
    st.session_state.final_analysis = ""

# Фильтр по нормативу (глобальный)
selected_doc_filter = st.selectbox("📄 Фильтр по нормативу", doc_list, index=0)

# Форма добавления нового дефекта
with st.form("add_defect_form"):
    uploaded_photo = st.file_uploader("Загрузите фото дефекта", type=["jpg", "png", "jpeg"], key="new_photo")
    defect_description = st.text_area("Описание дефекта", placeholder="Пример: трещина в бетонной стене...")
    add_button = st.form_submit_button("➕ Добавить дефект")

    if add_button and uploaded_photo and defect_description:
        # Сохраняем фото во временный файл
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(uploaded_photo.getvalue())
            tmp_path = tmp.name
        st.session_state.defects.append({
            "photo": tmp_path,
            "photo_filename": uploaded_photo.name,
            "description": defect_description,
            "selected_chunks": []  # список кортежей (doc_name, chunk_text)
        })
        st.success("✅ Дефект добавлен. Теперь для него можно выполнить поиск нормативов.")
        st.rerun()

# Отображение списка дефектов
if st.session_state.defects:
    st.markdown("### 📋 Список добавленных дефектов")
    for idx, defect in enumerate(st.session_state.defects):
        with st.expander(f"Дефект {idx + 1}: {defect['photo_filename']}"):
            # Показываем фото
            st.image(defect['photo'], width=200)
            st.write(f"**Описание:** {defect['description']}")

            # Поиск нормативов для этого дефекта
            if st.button(f"🔍 Найти нормативы для дефекта {idx + 1}", key=f"search_{idx}"):
                with st.spinner("Поиск..."):
                    results = find_similar(defect['description'], giga, es, INDEX_NAME, top_k=TOP_K,
                                           doc_filter=selected_doc_filter)
                if results:
                    st.session_state.current_search_results = results
                    st.session_state.current_defect_idx = idx
                    st.success("Найдены пункты. Выберите подходящие ниже.")
                else:
                    st.warning("Ничего не найдено. Попробуйте другое описание.")

            # Если есть результаты поиска для этого дефекта, показываем их
            if 'current_search_results' in st.session_state and st.session_state.current_defect_idx == idx:
                for i, res in enumerate(st.session_state.current_search_results):
                    with st.container():
                        st.markdown(f"**{res['doc_name']}** (Совпадение: {res['score']:.2%})")
                        st.write(res['text'][:300] + "...")
                        if st.button(f"✅ Выбрать этот пункт", key=f"select_{idx}_{i}"):
                            if (res['doc_name'], res['text']) not in defect['selected_chunks']:
                                defect['selected_chunks'].append((res['doc_name'], res['text']))
                                st.success("Пункт добавлен")
                                st.rerun()

            # Показываем уже выбранные пункты для этого дефекта
            if defect['selected_chunks']:
                st.write("**Выбранные пункты:**")
                for doc_name, chunk_text in defect['selected_chunks']:
                    st.write(f"- {doc_name}: {chunk_text[:100]}...")
                if st.button(f"🗑️ Очистить выбранные для дефекта {idx + 1}", key=f"clear_{idx}"):
                    defect['selected_chunks'] = []
                    st.rerun()

    # Кнопка формирования общего отчёта
    if st.button("📄 Сформировать общий отчёт по всем дефектам"):
        with st.spinner("GigaChat анализирует дефекты..."):
            # Формируем промпт
            defects_text = ""
            for i, defect in enumerate(st.session_state.defects, 1):
                defects_text += f"Дефект {i}: {defect['photo_filename']}\n"
                defects_text += f"Описание: {defect['description']}\n"
                if defect['selected_chunks']:
                    defects_text += "Соответствующие нормативные пункты:\n"
                    for doc_name, chunk_text in defect['selected_chunks']:
                        defects_text += f"- {doc_name}: {chunk_text[:200]}...\n"
                else:
                    defects_text += "Нормативные пункты не выбраны.\n"
                defects_text += "\n"
            prompt = f"""
            Ты — эксперт по строительным нормам и правилам.

            Задача: Проанализировать описанные ниже дефекты строительных конструкций и выбранные пункты нормативов.
            Для каждого дефекта опиши:
            - вероятную причину возникновения дефекта;
            - какие пункты нормативов нарушены (со ссылками);
            - как устранить дефект (рекомендации по ремонту).
            Затем дай общее заключение о состоянии конструкций.

            {defects_text}

            Ответ должен быть структурирован: для каждого дефекта отдельный блок, затем общий вывод.
            """
            messages = [Messages(role=MessagesRole.USER, content=prompt)]
            chat_payload = Chat(messages=messages)
            response = giga.chat(chat_payload)
            analysis_text = response.choices[0].message.content

            st.session_state.final_analysis = analysis_text
            st.session_state.analysis_done = True

# Отображение результатов анализа и кнопка скачивания PDF
if st.session_state.analysis_done:
    st.markdown("### 📊 Результаты анализа GigaChat")
    st.write(st.session_state.final_analysis)

    pdf_data = generate_report(st.session_state.defects, st.session_state.final_analysis)
    st.download_button(
        label="📥 Скачать отчёт в PDF",
        data=pdf_data,
        file_name="defect_report.pdf",
        mime="application/pdf"
    )
    if st.button("🔄 Начать новый анализ"):
        st.session_state.defects = []
        st.session_state.analysis_done = False
        st.session_state.final_analysis = ""
        st.rerun()