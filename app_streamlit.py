import streamlit as st
import numpy as np
import io
import os
import re
from elasticsearch import Elasticsearch
from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT, TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from dotenv import load_dotenv

load_dotenv()

# Настройки
ES_HOST = "http://127.0.0.1:9200"
INDEX_NAME = "construction_standards"
TOP_K = 5

# Регистрация кириллического шрифта для PDF
def register_russian_font():
    possible_fonts = [
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Arial.ttf",
        os.path.join(os.path.dirname(__file__), "fonts", "DejaVuSerif.ttf")
    ]
    for font_path in possible_fonts:
        if os.path.exists(font_path):
            pdfmetrics.registerFont(TTFont('RussianFont', font_path))
            return 'RussianFont'
    return 'Helvetica'

RUSSIAN_FONT = register_russian_font()

# Настройка широкой страницы
st.set_page_config(page_title="Поиск по строительным нормативам", layout="wide")

# CSS для расширения контента
st.markdown("""
<style>
    .reportview-container .main .block-container {
        max-width: 95%;
        padding-top: 2rem;
        padding-right: 2rem;
        padding-left: 2rem;
    }
    .stTextArea textarea {
        font-size: 16px;
    }
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
</style>
""", unsafe_allow_html=True)

# ---------- Функции для работы с GigaChat и Elasticsearch ----------
@st.cache_resource
def load_giga():
    return GigaChat(
        credentials=os.getenv("GIGACHAT_CREDENTIALS"),
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

def get_embedding(giga, text: str):
    response = giga.embeddings([text])
    return np.array(response.data[0].embedding, dtype=np.float32).tolist()

def find_similar(query_text, giga, es, index_name, top_k=TOP_K):
    query_vector = get_embedding(giga, query_text)
    body = {
        "size": top_k,
        "query": {
            "script_score": {
                "query": {"match_all": {}},
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

def clean_markdown(text: str) -> str:
    """Удаляет маркеры Markdown (#, *, **, ---) и лишние пробелы."""
    text = re.sub(r'#+\s*', '', text)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'[-*]{3,}', '', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    return text.strip()

def generate_report(defects_data, analysis_text):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
    styles = getSampleStyleSheet()

    style_title = ParagraphStyle('Title', parent=styles['Title'], fontName=RUSSIAN_FONT, fontSize=16,
                                 alignment=TA_CENTER, spaceAfter=12)
    style_heading = ParagraphStyle('Heading', parent=styles['Heading2'], fontName=RUSSIAN_FONT, fontSize=12,
                                   alignment=TA_LEFT, spaceAfter=6, spaceBefore=12)
    style_body = ParagraphStyle('Body', parent=styles['Normal'], fontName=RUSSIAN_FONT, fontSize=10,
                                alignment=TA_JUSTIFY, leading=14, spaceAfter=12)
    style_italic = ParagraphStyle('Italic', parent=styles['Normal'], fontName=RUSSIAN_FONT, fontSize=10,
                                  alignment=TA_JUSTIFY, leading=14, spaceAfter=6, fontStyle='italic')

    story = []
    story.append(Paragraph("Общий отчет по результатам анализа дефектов", style_title))
    story.append(Spacer(1, 12))

    for idx, defect in enumerate(defects_data, 1):
        story.append(Paragraph(f"Дефект №{idx}", style_heading))

        if 'image_bytes' in defect and defect['image_bytes']:
            try:
                img_buffer = io.BytesIO(defect['image_bytes'])
                img = Image(img_buffer, width=250, height=200, kind='proportional')
                story.append(img)
                story.append(Spacer(1, 6))
            except Exception as e:
                story.append(Paragraph(f"[Ошибка загрузки фото: {e}]", style_body))

        # Убрали строку с выводом имени файла
        story.append(Paragraph(f"Описание: {defect['description']}", style_body))

        if defect['selected_chunks']:
            story.append(Paragraph("Выбранные пункты нормативов:", style_italic))
            for i, (doc_name, chunk_text) in enumerate(defect['selected_chunks'], 1):
                short_chunk = chunk_text[:300] + "..." if len(chunk_text) > 300 else chunk_text
                story.append(Paragraph(f"{i}. {doc_name}: {short_chunk}", style_body))
        else:
            story.append(Paragraph("Нормативные пункты не выбраны.", style_italic))
        story.append(Spacer(1, 12))

    story.append(Paragraph("Общее заключение и рекомендации", style_heading))
    cleaned_analysis = clean_markdown(analysis_text)
    paragraphs = cleaned_analysis.split('\n\n')
    for para in paragraphs:
        if para.strip():
            story.append(Paragraph(para.replace('\n', ' ').strip(), style_body))

    doc.build(story)
    return buffer.getvalue()

# Интерфейс Streamlit
st.markdown('<div class="main-header">🏗️ Поиск и анализ дефектов по строительным нормативам</div>', unsafe_allow_html=True)

# Подключения
try:
    es = load_elasticsearch()
    if not es.ping():
        st.error("Не удалось подключиться к Elasticsearch. Убедитесь, что контейнер запущен (docker-compose up -d).")
        st.stop()
    if not es.indices.exists(index=INDEX_NAME):
        st.error(f"Индекс '{INDEX_NAME}' не найден. Сначала запустите batch_index.py")
        st.stop()
    st.success("Подключение к Elasticsearch установлено")
except Exception as e:
    st.error(f"Ошибка Elasticsearch: {e}")
    st.stop()

try:
    giga = load_giga()
    st.success("Подключение к GigaChat установлено")
except Exception as e:
    st.error(f"Ошибка GigaChat: {e}")
    st.stop()

# Инициализация сессии
if 'defects' not in st.session_state:
    st.session_state.defects = []
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'final_analysis' not in st.session_state:
    st.session_state.final_analysis = ""

# Счётчик версии формы для очистки после добавления
if 'form_version' not in st.session_state:
    st.session_state.form_version = 0

# Блок добавления дефекта с предпросмотром
st.markdown("Добавление нового дефекта")

current_version = st.session_state.form_version
uploaded_photo = st.file_uploader("Загрузите фото дефекта", type=["jpg", "png", "jpeg"], key=f"new_photo_upload_{current_version}")

# Временные переменные для предпросмотра
if 'temp_photo_bytes' not in st.session_state:
    st.session_state.temp_photo_bytes = None
if 'temp_photo_filename' not in st.session_state:
    st.session_state.temp_photo_filename = ""

if uploaded_photo is not None:
    st.image(uploaded_photo, width=300, caption=uploaded_photo.name)
    st.session_state.temp_photo_bytes = uploaded_photo.getvalue()
    st.session_state.temp_photo_filename = uploaded_photo.name
else:
    if st.session_state.temp_photo_bytes is not None:
        st.session_state.temp_photo_bytes = None
        st.session_state.temp_photo_filename = ""

defect_description = st.text_area("Описание дефекта", value="",
                                   placeholder="Например: вздутие кровли, трещина в стене...", height=150, key=f"desc_{current_version}")

if st.button("Добавить дефект"):
    if st.session_state.temp_photo_bytes and defect_description.strip():
        st.session_state.defects.append({
            "image_bytes": st.session_state.temp_photo_bytes,
            "photo_filename": st.session_state.temp_photo_filename,
            "description": defect_description.strip(),
            "selected_chunks": []
        })
        # Очищаем временные данные и увеличиваем версию формы
        st.session_state.temp_photo_bytes = None
        st.session_state.temp_photo_filename = ""
        st.session_state.form_version += 1
        st.success("Дефект добавлен")
        st.rerun()
    else:
        st.warning("Загрузите фото и введите описание")

# Отображение списка дефектов
if st.session_state.defects:
    st.markdown("### Список добавленных дефектов")
    for idx, defect in enumerate(st.session_state.defects):
        col_img, col_content = st.columns([1, 2])
        with col_img:
            st.image(defect['image_bytes'], width=300, caption=defect['photo_filename'])
            # Кнопка удаления дефекта под фото
            if st.button(f"🗑️ Удалить дефект {idx+1}", key=f"delete_{idx}"):
                st.session_state.defects.pop(idx)
                if not st.session_state.defects:
                    st.session_state.analysis_done = False
                    st.session_state.final_analysis = ""
                st.rerun()
        with col_content:
            st.write(f"**Дефект {idx + 1}**")
            st.write(f"**Описание:** {defect['description']}")
            if st.button(f"Найти нормативы для дефекта {idx + 1}", key=f"search_{idx}"):
                with st.spinner("Поиск..."):
                    results = find_similar(defect['description'], giga, es, INDEX_NAME, top_k=TOP_K)
                if results:
                    st.session_state.current_search_results = results
                    st.session_state.current_defect_idx = idx
                    st.success("Найдены пункты. Выберите подходящие ниже.")
                else:
                    st.warning("Ничего не найдено.")
            if 'current_search_results' in st.session_state and st.session_state.current_defect_idx == idx:
                remaining_results = [
                    res for res in st.session_state.current_search_results
                    if (res['doc_name'], res['text']) not in defect['selected_chunks']
                ]
                for i, res in enumerate(remaining_results):
                    with st.expander(f"{res['doc_name']} (ID: {res['doc_id']}) (совпадение {res['score']:.2%})", expanded=True):
                        st.write(res['text'])
                        if st.button(f"Выбрать", key=f"select_{idx}_{i}"):
                            defect['selected_chunks'].append((res['doc_name'], res['text']))
                            st.session_state.current_search_results.remove(res)
                            st.success("Пункт добавлен")
                            st.rerun()
                if not remaining_results:
                    del st.session_state.current_search_results
                    st.rerun()
            if defect['selected_chunks']:
                st.write("**Выбранные пункты:**")
                for doc_name, chunk_text in defect['selected_chunks']:
                    st.write(f"- {doc_name}: {chunk_text[:100]}...")
                if st.button(f"Очистить выбранные", key=f"clear_{idx}"):
                    defect['selected_chunks'] = []
                    st.rerun()
        st.divider()

    if st.button("🗑️ Очистить все дефекты", use_container_width=True):
        st.session_state.defects = []
        st.session_state.analysis_done = False
        st.session_state.final_analysis = ""
        st.rerun()

    if st.button("Сформировать общий отчёт по всем дефектам", use_container_width=True):
        if not st.session_state.defects:
            st.warning("Нет добавленных дефектов.")
        else:
            with st.spinner("GigaChat анализирует дефекты..."):
                defects_text = ""
                for i, defect in enumerate(st.session_state.defects, 1):
                    defects_text += f"Дефект {i}: {defect['photo_filename']}\n"
                    defects_text += f"Описание: {defect['description']}\n"
                    if defect['selected_chunks']:
                        defects_text += "Выбранные нормативные пункты:\n"
                        for doc_name, chunk_text in defect['selected_chunks']:
                            defects_text += f"- {doc_name}: {chunk_text[:200]}...\n"
                    else:
                        defects_text += "Нормативные пункты не выбраны.\n"
                    defects_text += "\n"
                prompt = f"""
                Ты — эксперт по строительным нормам и правилам.
                Проанализируй описанные ниже дефекты и выбранные пункты нормативов.
                Ответ должен содержать для каждого дефекта:
                - Вероятную причину возникновения дефекта.
                - Какие пункты нормативных документов нарушены (укажи конкретные ссылки).
                - Рекомендации по устранению дефекта.
                Затем дай общее заключение о состоянии конструкций.
                Не используй символы # и * в ответе. Пиши обычным текстом, разделяя разделы пустыми строками.

                {defects_text}
                """
                messages = [Messages(role=MessagesRole.USER, content=prompt)]
                chat_payload = Chat(messages=messages)
                response = giga.chat(chat_payload)
                analysis_text = response.choices[0].message.content
                st.session_state.final_analysis = analysis_text
                st.session_state.analysis_done = True

# Результат анализа и скачивание PDF
if st.session_state.analysis_done:
    st.markdown("### Результаты анализа GigaChat")
    st.write(st.session_state.final_analysis)
    pdf_data = generate_report(st.session_state.defects, st.session_state.final_analysis)
    st.download_button(
        label="Скачать отчёт в PDF",
        data=pdf_data,
        file_name="defect_report.pdf",
        mime="application/pdf",
        use_container_width=True
    )
    if st.button("Начать новый анализ", use_container_width=True):
        st.session_state.defects = []
        st.session_state.analysis_done = False
        st.session_state.final_analysis = ""
        st.rerun()