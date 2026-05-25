FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV DATA_DIR=/data
ENV DB_PATH=/data/results.db
ENV CORS_ORIGINS=http://localhost:5173
ENV REQUEST_TIMEOUT=120
ENV PYTHONUNBUFFERED=1

RUN mkdir -p /data/extractedFrames/save

EXPOSE 8000

WORKDIR /app/backend

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
