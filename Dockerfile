FROM python:3.10

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app

WORKDIR /app/app

ENV PORT=8080
ENV PYTHONUNBUFFERED=1

CMD exec gunicorn \
    --bind :$PORT \
    --workers 1 \
    --threads 8 \
    --timeout 300 \
    --graceful-timeout 300 \
    --log-level info \
    main:app