FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    vim \
    wget \
    curl \
    git \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY analysis-server/requirements.txt /app/analysis-server/requirements.txt
RUN pip3 install --no-cache-dir -r /app/analysis-server/requirements.txt

COPY shared /app/shared
COPY analysis-server /app/analysis-server

ENV PYTHONPATH=/app:/app/analysis-server
WORKDIR /app/analysis-server

CMD ["celery", "-A", "tasks", "worker", "--loglevel=info", "--concurrency=2", "--queue=inference"]
