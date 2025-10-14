FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

COPY operation-server/requirements.txt /app/operation-server/requirements.txt
RUN pip install --no-cache-dir -r /app/operation-server/requirements.txt

COPY shared /app/shared
COPY operation-server /app/operation-server

ENV PYTHONPATH=/app:/app/operation-server
WORKDIR /app/operation-server

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
