FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/

# Railway 会注入 PORT；我们绑定 0.0.0.0:$PORT
ENV PORT=8000
CMD ["/bin/sh", "-c", "uvicorn app_fastapi:api --host 0.0.0.0 --port ${PORT:-8000}"]
