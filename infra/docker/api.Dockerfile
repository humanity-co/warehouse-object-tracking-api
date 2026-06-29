FROM python:3.11-slim

WORKDIR /app

COPY platform /app/platform
COPY services/api /app/services/api

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e /app/platform

ENV PYTHONPATH=/app/platform/src

WORKDIR /app/services/api

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

