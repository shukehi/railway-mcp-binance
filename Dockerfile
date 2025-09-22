FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY app_fastapi.py /app/

ENV PORT=8000
CMD ["/bin/sh", "-c", "uvicorn app_fastapi:api --host 0.0.0.0 --port ${PORT:-8000}"]
