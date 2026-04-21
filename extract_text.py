import json

# 1. Загружаем JSON
with open("data/ГОСТ 31937-2024.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 2. Склеиваем всё содержимое страниц
all_text = []
for page in data["pages"]:
    all_text.append(page["page_content"])

full_text = "\n\n".join(all_text)   # разделяем страницы двойным переводом строки

# 3. Сохраняем в текстовый файл
with open("gost_31937_full.txt", "w", encoding="utf-8") as out:
    out.write(full_text)

print(f"Создан файл gost_31937_full.txt, {len(full_text)} символов.")