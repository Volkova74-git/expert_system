import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Any, Optional

def normalize_table_line_breaks(text: str) -> str:
    pattern = re.compile(r"(\d{5}/)\s*\n\s+")
    return pattern.sub(r"\1 ", text)


def merge_short_chunks(chunks: List[str], min_len: int = 150) -> List[str]:
    """
    устраняет слишком короткие обрывки.
    """
    if not chunks:
        return chunks

    result: List[str] = []
    i = 0
    n = len(chunks)

    while i < n:
        ch = chunks[i].strip()
        if not ch:
            i += 1
            continue

        if i == n - 1:
            if result and len(ch) < min_len:
                result[-1] = result[-1] + " " + ch
            else:
                result.append(ch)
            break

        if len(ch) < min_len:
            merged = ch + " " + chunks[i + 1]
            chunks[i + 1] = merged
            i += 1
            continue

        # нормальный чанк
        result.append(ch)
        i += 1

    return result


def normalize_and_merge_chunks(
    chunks: List[str],
    max_chunk_len: int = 1000,
    min_chunk_len: int = 300,
) -> List[str]:
    """
      1. Склеивает "плохо разбитые" чанки:
         - ссылки на статьи: "… ст." + "14 Семейного…" одна фраза;
         - "… статьи 3" + "6 Федерального закона…" одна фраза;
         - "3." + "2 Формы…" воспринимаем как продолжение ("3.2 ...");

      2. Разрезает по заголовкам разделов
         - при этом НЕ режет внутри ссылок на статьи
           (ст. 14, статьи 51).

      4. склеивает слишком маленькие чанки с соседями,
         если суммарная длина не превышает max_chunk_len.

      5. Нормализует пробелы (один пробел между словами).
    """

    def should_merge(prev: str, cur: str) -> bool:
        """
        Эвристика: нужно ли склеивать текущий чанк с предыдущим.
        Работает по первой "содержательной" строке cur и хвосту prev.
        """
        prev = prev.rstrip()
        cur = cur.lstrip()
        if not prev or not cur:
            return False

        s = cur.lstrip()
        if not s:
            return False

        # "… ст." + "14 …"
        if re.search(r"\bст\.$", prev, re.IGNORECASE) and re.match(r"^\d+\b", s):
            return True

        # "… статьи" / "… статьи 3" + "6 …"
        if re.search(r"\bстатьи(?:\s+\d+)?$", prev, re.IGNORECASE) and re.match(r"^\d+\b", s):
            return True

        # "3." + "2 Формы…" считаем продолжением (для криво распознанных "3.2")
        if re.search(r"\b\d+\.$", prev) and re.match(r"^\d+\s+[А-ЯA-Z]", s):
            return True

        # если явно конец предложения - НЕ склеиваем
        if prev[-1] in ".!?;:\"»":
            return False

        first = s[0]

        # строка начинается с числа скорее всего продолжение (таблица, номер и т.п.),
        # НО не для заголовков вида "2. Текст"
        if re.match(r"^\d+\b", s) and not re.match(r"^\d+[.)]\s+[А-ЯA-Z]", s):
            return True

        # строка пунктуации или буллета
        if first in ".,;:)]" or first in "•№-–/":
            return True

        if first.isalpha() and first.islower():
            return True

        # "2. ..." / "3) ..." и т.п.
        if re.match(r"^\d+[.)]\s+", s):
            return True

        return False

    merged: List[str] = []
    for ch in chunks:
        ch = ch.strip()
        if not ch:
            continue

        if not merged:
            merged.append(ch)
            continue

        prev = merged[-1]
        if should_merge(prev, ch):
            merged[-1] = prev.rstrip() + " " + ch.lstrip()
        else:
            merged.append(ch)

    # Разрезка по заголовкам разделов
    section_pattern = re.compile(
        r"(?<!ст\.\s)(?<!статьи\s)"          # не после "ст. " / "статьи "
        r"(?:(?<=^)|(?<=\n)|(?<=[.!?;]\s))"  # начало, после \n или после конца предложения
        r"(\d+[.)]?\s+[А-ЯA-Z])"             # сам номер раздела
    )

    def split_by_sections_in_chunk(ch: str) -> List[str]:
        parts: List[str] = []
        last_idx = 0
        for m in section_pattern.finditer(ch):
            start = m.start(1)

            if start == 0 and last_idx == 0:
                continue

            prev_part = ch[last_idx:start].strip()
            if prev_part:
                parts.append(prev_part)
            last_idx = start

        tail = ch[last_idx:].strip()
        if tail:
            parts.append(tail)

        return parts or [ch]

    split_by_sections: List[str] = []
    for ch in merged:
        ch = ch.strip()
        if not ch:
            continue
        split_by_sections.extend(split_by_sections_in_chunk(ch))

    deduped: List[str] = []
    seen = set()
    for ch in split_by_sections:
        ch = ch.strip()
        if not ch:
            continue
        if ch in seen:
            continue
        seen.add(ch)
        deduped.append(ch)

    compact: List[str] = []
    for ch in deduped:
        ch = ch.strip()
        if not ch:
            continue

        if compact:
            prev = compact[-1]
            if (len(ch) < min_chunk_len or len(prev) < min_chunk_len) and \
               len(prev) + 1 + len(ch) <= max_chunk_len:
                compact[-1] = prev + " " + ch
                continue

        compact.append(ch)

    # Нормализация пробелов
    normalized: List[str] = []
    for ch in compact:
        ch_norm = re.sub(r"\s+", " ", ch).strip()
        if ch_norm:
            normalized.append(ch_norm)

    normalized = merge_short_chunks(normalized, min_len=150)

    return normalized


def build_search_chunks(text: str) -> List[str]:
    """
    Пайплайн разбиения текста на чанки для поиска / эмбеддингов.
    """
    text = normalize_table_line_breaks(text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
    )

    raw_chunks = splitter.split_text(text)

    fixed_chunks = normalize_and_merge_chunks(
        raw_chunks,
        max_chunk_len=1000,
        min_chunk_len=300,
    )
    return fixed_chunks


def clean_page_numbers(text: str) -> str:
    lines = text.splitlines()
    out_lines = []
    for line in lines:
        stripped = line.strip()
        if re.fullmatch(r"\d{1,4}", stripped):
            continue
        out_lines.append(line)
    return "\n".join(out_lines)


def split_by_sections(one_line: str) -> str:
    pattern = re.compile(r'(?<!\w)(\d+(?:\.\d+)*[.)]?\s+(?=[А-ЯA-Z]))')

    sections: List[str] = []
    last_idx = 0

    for m in pattern.finditer(one_line):
        start = m.start(1)
        if start == 0:
            continue
        chunk = one_line[last_idx:start].strip()
        if chunk:
            sections.append(chunk)
        last_idx = start

    tail = one_line[last_idx:].strip()
    if tail:
        sections.append(tail)

    return "\n".join(sections)


def normalize_block(text: str) -> str:
    """
    Приводит блок к аккуратному виду:
    - склеивает номера пунктов (2.1 / 1. / 1) / 3.2.4) с текстом;
    - склеивает '... -\\nстрока';
    - склеивает разорванные предложения;
    - склеивает строки, начинающиеся с '. ; , : ) ] • № - –' с предыдущей;
    - в конце разбивает по номерам разделов и ставит \\n только там.
    """
    lines = [l.rstrip() for l in text.splitlines()]
    result: List[str] = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # пропускаем пустые строки
        if line == "":
            i += 1
            continue

        # если строка начинается с "подвисшего" знака или буллета,
        # считаем её продолжением предыдущей
        if result:
            # первая значимая (не пробельная) буква/символ
            stripped_leading = line.lstrip()
            if stripped_leading:
                first = stripped_leading[0]
                # строка начинается с . , ; : ) ] или с буллета/№/дефиса
                if first in ".,;:)]" or first in "•№-–":
                    prev = result[-1]
                    # для чистой пунктуации (.,;:)] клеим без доп. пробела
                    if first in ".,;:)]":
                        result[-1] = prev + stripped_leading
                    else:
                        # для буллетов/№/дефиса через пробел
                        result[-1] = prev + " " + stripped_leading
                    i += 1
                    continue

        # Номер пункта отдельно: "2.1", "1.", "1)", "3.2.4"
        marker_pattern = r"^\d+(?:\.\d+)*[.)]?$"
        if re.fullmatch(marker_pattern, line) and i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            if next_line:
                merged = f"{line} {next_line}"
                result.append(merged)
                i += 2
                continue
            else:
                result.append(line)
                i += 1
                continue

        # Склейка '... -\\nстрока'
        if result and re.search(r"\s-$", result[-1]):
            prev = result[-1]
            merged = re.sub(r"\s-$", "", prev) + " - " + line
            result[-1] = merged
            i += 1
            continue

        # Склейка разорванных предложений по маленькой букве.
        if result:
            prev = result[-1]
            if prev and not re.search(r"[.:;!?]$", prev):
                first_alpha = None
                for ch in line:
                    if ch.isalpha():
                        first_alpha = ch
                        break
                if first_alpha is not None and first_alpha.islower():
                    result[-1] = prev + " " + line
                    i += 1
                    continue

        # обычная строка
        result.append(line)
        i += 1

    # превращаем всё в одну строку
    one_line = " ".join(result)
    one_line = re.sub(r"\s+", " ", one_line).strip()

    # разбиваем по разделам (1, 1.1, 2.3.4 ...) ставим \n только там
    return split_by_sections(one_line)


def filter_document(text: str) -> Dict[str, Optional[str]]:
    """
    - режем по 'ПРИКАЗЫВАЮ', сохраняем этот блок;
    - убираем раздел 'Приложения' в конце;
    - выделяем 'Используемые понятия' и 'Общие положения';
    - чистим номера страниц;
    - нормализуем блоки и разбиваем их по разделам.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    order_block: Optional[str] = None
    terms_block: Optional[str] = None

    # ПРИКАЗЫВАЮ
    m_order = re.search(r"приказываю", text, re.IGNORECASE)
    if m_order:
        after_order = text[m_order.start():]
        parts = re.split(r"\n\s*\n", after_order, maxsplit=1)
        order_block = parts[0].strip()
        if len(parts) > 1:
            remaining = parts[1].lstrip("\n")
        else:
            remaining = ""
    else:
        remaining = text

    pattern_app_header = re.compile(
        r"^[ \t]*приложени[ея]\b(?!\s+к\b).*$",
        re.IGNORECASE | re.MULTILINE,
    )
    m_app_all = pattern_app_header.search(remaining)
    if m_app_all:
        work = remaining[:m_app_all.start()].rstrip()
    else:
        work = remaining

    pattern_terms = re.compile(
        r"^[ \t]*(\d+[\.\)]\s+)?используемые\s+понятия\b(?!.*\.{5,}).*$",
        re.IGNORECASE | re.MULTILINE,
    )
    pattern_general = re.compile(
        r"^[ \t]*(\d+[\.\)]\s+)?(\d+\s+)?общие\s+положени[ея]\b(?!.*\.{5,}).*$",
        re.IGNORECASE | re.MULTILINE,
    )

    m_terms = pattern_terms.search(work)
    m_general = pattern_general.search(work)

    if m_terms and m_general and m_terms.start() < m_general.start():
        terms_block_raw = work[m_terms.start():m_general.start()].strip()
        body_text_raw = work[m_general.start():].strip()
    elif m_general and (not m_terms or m_general.start() < m_terms.start()):
        terms_block_raw = None
        body_text_raw = work[m_general.start():].strip()
    elif m_terms and not m_general:
        terms_block_raw = work[m_terms.start():].strip()
        body_text_raw = ""
    else:
        terms_block_raw = None
        body_text_raw = work.strip()

    # чистим номера страниц и нормализуем + разбиваем по разделам
    if order_block:
        order_block = normalize_block(clean_page_numbers(order_block))
    if terms_block_raw:
        terms_block = normalize_block(clean_page_numbers(terms_block_raw))
    else:
        terms_block = None
    body_text = normalize_block(clean_page_numbers(body_text_raw))

    return {
        "order_block": order_block or None,
        "terms_block": terms_block or None,
        "body": body_text,
    }