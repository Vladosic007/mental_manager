# Ментал — веб-приложение для деплоя на Hugging Face Spaces (Docker SDK).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache \
    PORT=7860

# Непривилегированный пользователь (рекомендация HF Spaces)
RUN useradd -m -u 1000 user

WORKDIR /app

# torch — отдельно, CPU-сборка (без CUDA: легче и достаточно для инференса).
# Версию не фиксируем: берём актуальное CPU-колесо, совместимое с sentence-transformers.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Остальные зависимости веб-приложения
COPY requirements-web.txt .
RUN pip install --no-cache-dir -r requirements-web.txt

# Код приложения
COPY . .

# Заранее кладём модель семантики в образ — чтобы старт был быстрым,
# и отдаём права пользователю (он пишет JSON-файлы данных в /app на рантайме)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')" \
    && chown -R user:user /app

USER user
EXPOSE 7860
CMD ["python", "webapp/server.py"]
