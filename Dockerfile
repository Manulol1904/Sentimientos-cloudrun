FROM python:3.10

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app
WORKDIR /app/app

ENV PORT=8080

CMD exec gunicorn --bind :$PORT main:app