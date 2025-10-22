FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY victoria-consumer/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY victoria-consumer/ ./victoria-consumer/
COPY shared/ ./shared/

WORKDIR /app/victoria-consumer

CMD ["python", "consumer.py"]
