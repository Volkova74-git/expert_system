# Умный поиск по строительным нормативам

Прототип системы семантического поиска по строительным нормам и правилам (СП, ГОСТ, СНиП). Позволяет загружать фото дефектов, описывать их естественным языком, находить релевантные пункты нормативов, формировать PDF-отчёт с анализом от GigaChat.

## Используемые данные

Датасет строительных нормативов: Russian Construction Standards Dataset (https://github.com/Akumsk/russian-construction-standards). В проекте используются JSON-файлы с текстами нормативов (СП 17.13330.2017, СП 60.13330.2020 и др.). Данные распространяются под лицензией CC BY 4.0.

## Архитектура

Хранение и поиск: Elasticsearch (векторное хранилище, косинусное сходство). Генерация эмбеддингов и анализ: GigaChat API. Интерфейс: Streamlit.

## Запуск проекта

1. Установите Python 3.9+, Docker Desktop, Git.

2. Клонируйте репозиторий:
   git clone https://github.com/Volkova74-git/expert_system.git
   cd expert_system

3. Создайте и активируйте виртуальное окружение:
   python -m venv .venv
   source .venv/bin/activate   (Linux/macOS)
   .venv\Scripts\activate      (Windows)

4. Установите зависимости:
   pip install -r requirements.txt

5. Получите API-ключ GigaChat в личном кабинете разработчика. Создайте в корне проекта файл .env со строкой:
   GIGACHAT_CREDENTIALS=ваш_ключ

6. Запустите Elasticsearch через Docker Compose:
   docker-compose up -d

7. Выполните индексацию нормативных документов (один раз):
   python batch_index.py

8. Запустите веб-приложение:
   streamlit run app_streamlit.py

   Откройте в браузере http://localhost:8501.

## Оценка качества поиска

Для расчёта метрик (Precision@5, Recall@5, MRR, NDCG@5) выполните:
python evaluate_metrics.py

Результаты сохраняются в evaluation_results.json и выводятся в консоль.

## Добавление новых нормативов

Поместите JSON-файл в папку standards/ и повторно запустите python batch_index.py. Новые документы добавятся к существующим.

## Устранение неполадок

- Если Elasticsearch не отвечает, проверьте, что Docker Desktop запущен, и выполните docker-compose restart.
- При ошибке 401 от GigaChat проверьте ключ в файле .env.
- Для очистки всех данных Elasticsearch (переиндексация): docker-compose down -v, затем docker-compose up -d и python batch_index.py.

## Ссылки

Репозиторий проекта: https://github.com/Volkova74-git/expert_system
Датасет: https://github.com/Akumsk/russian-construction-standards
Документация GigaChat: https://developers.sber.ru/docs/ru/gigachat/overview
