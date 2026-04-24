FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# For transcription
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \ 
    build-essential \
    && \
    # Clean up the cache to keep the image small
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt


COPY ./src ./src

EXPOSE 8000

CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "src.main:app", "-b", "0.0.0.0:8000"]