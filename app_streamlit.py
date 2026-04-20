import streamlit as st
import numpy as np
import pickle
import io
from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole
from sklearn.metrics.pairwise import cosine_similarity
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from dotenv import load_dotenv

credentials=os.getenv("GIGACHAT_CREDENTIALS"),


@st.cache_resource
def load_search_index():
    with open("standards_index.faiss", "rb") as f:
        index = np.load(f)
    with open("standards_texts.pkl", "rb") as f:
        texts = pickle.load(f)
    return index, texts

@st.cache_resource
def load_giga():
    return GigaChat(
        credentials=GIGA_CREDENTIALS,
        verify_ssl_certs=False,
        scope="GIGACHAT_API_PERS",
        model="GigaChat-Pro",
        auth_url="https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
        api_base="https://gigachat.devices.sberbank.ru/api/v1",
        timeout=60
    )

def find_similar(query_text, giga, index, texts, top_k=5):
    response = giga.embeddings([query_text])
    query_emb = np.array(response.data[0].embedding, dtype=np.float32).reshape(1, -1)
    similarities = cosine_similarity(query_emb, index)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    results = []
    for idx in top_indices:
        results.append({
            "text": texts[idx],
            "score": similarities[idx],
            "index": idx
        })
    return results

def generate_report(defect_description, chosen_chunks, analysis_text):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
    styles = getSampleStyleSheet()
    style_title = ParagraphStyle('Title', parent=styles['Title'], fontSize=16, alignment=TA_LEFT, spaceAfter=12)
    style_heading = ParagraphStyle('Heading', parent=styles['Heading2'], fontSize=12, alignment=TA_LEFT, spaceAfter=6, spaceBefore=12)
    style_body = ParagraphStyle('Body', parent=styles['Normal'], fontSize=10, alignment=TA_JUSTIFY, leading=14, spaceAfter=12)
    story = []
    story.append(Paragraph("Отчет по результатам анализа дефекта", style_title))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Описание дефекта:", style_heading))
    story.append(Paragraph(defect_description, style_body))
    story.append(Paragraph("Выбранные пункты нормативов:", style_heading))
    for i, chunk in enumerate(chosen_chunks, 1):
        short_chunk = chunk[:500] + "..." if len(chunk) > 500 else chunk
        story.append(Paragraph(f"<b>{i}. </b>{short_chunk}", style_body))
    story.append(Paragraph("Заключение и рекомендации:", style_heading))
    story.append(Paragraph(analysis_text.replace('\n', '<br/>'), style_body))
    doc.build(story)
    return buffer.getvalue()

st.set_page_config(page_title="Поиск по строительным нормативам", layout="wide")
st.title("🏗️ Поиск и анализ дефектов по строительным нормативам")

try:
    with st.spinner("Загрузка поискового индекса..."):
        index, texts = load_search_index()
    with st.spinner("Подключение к GigaChat..."):
        giga = load_giga()
except FileNotFoundError:
    st.error("Файлы индекса не найдены. Сначала запустите python index_standalone.py")
    st.stop()
except Exception as e:
    st.error(f"Ошибка загрузки: {e}")
    st.stop()

if 'selected_chunks' not in st.session_state:
    st.session_state.selected_chunks = []
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'defect_description' not in st.session_state:
    st.session_state.defect_description = ""

col1, col2 = st.columns(2)
with col1:
    uploaded_files = st.file_uploader("📸 Загрузите фото дефектов", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        st.image(uploaded_file, width=200, caption=uploaded_file.name)
with col2:
    defect_description = st.text_area("📝 Опишите дефект", height=200,
                                       placeholder="Пример: На бетонной поверхности обнаружены трещины...")
    search_button = st.button("🔍 Найти пункты в нормативах")

if search_button and defect_description:
    st.session_state.defect_description = defect_description
    with st.spinner("Поиск релевантных пунктов..."):
        results = find_similar(defect_description, giga, index, texts, top_k=5)
    if results:
        st.subheader("📚 Найденные пункты нормативов")
        st.session_state.search_results = results
        st.session_state.selected_chunks = []
        for i, res in enumerate(results):
            with st.expander(f"Результат {i+1} (Совпадение: {res['score']:.2%})"):
                st.write(res['text'])
                if st.button(f"Выбрать этот пункт", key=f"btn_{i}"):
                    if res['text'] not in st.session_state.selected_chunks:
                        st.session_state.selected_chunks.append(res['text'])
                        st.success(f"Пункт {i+1} добавлен")
    else:
        st.warning("Ничего не найдено. Попробуйте другое описание.")

if st.session_state.selected_chunks:
    st.subheader(f"✅ Выбрано: {len(st.session_state.selected_chunks)} пунктов")
    for i, chunk in enumerate(st.session_state.selected_chunks):
        st.text_area(f"Пункт {i+1}", chunk[:300], height=100, key=f"selected_{i}")
    if st.button("🗑️ Очистить"):
        st.session_state.selected_chunks = []
        st.rerun()

if st.session_state.selected_chunks and st.button("🤖 Проанализировать и создать отчёт"):
    with st.spinner("Анализ..."):
        prompt = f"""
        Ты — эксперт по строительным нормам.
        Описание дефекта: {st.session_state.defect_description}
        Выбранные пункты нормативов:
        {chr(10).join([f'- {chunk}' for chunk in st.session_state.selected_chunks])}
        Ответ должен содержать три секции:
        1. **Анализ дефекта** (почему нарушение)
        2. **Рекомендации по устранению**
        3. **Вывод о состоянии конструкции**
        """
        messages = [Messages(role=MessagesRole.USER, content=prompt)]
        response = giga.chat(Chat(messages=messages))
        analysis_text = response.choices[0].message.content
        st.session_state.analysis_text = analysis_text
        st.session_state.analysis_done = True

if st.session_state.get('analysis_done', False):
    st.subheader("📊 Результат анализа")
    st.markdown(st.session_state.analysis_text)
    pdf_data = generate_report(st.session_state.defect_description, st.session_state.selected_chunks, st.session_state.analysis_text)
    st.download_button("📥 Скачать отчёт PDF", pdf_data, "defect_report.pdf", "application/pdf")
    if st.button("🔄 Новый анализ"):
        st.session_state.selected_chunks = []
        st.session_state.analysis_done = False
        st.rerun()