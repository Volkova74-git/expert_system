# Dockerfile
FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Порт, на котором работает uvicorn
EXPOSE 8080

# Запуск FastAPI-приложения
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]