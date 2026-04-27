FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \ 
    build-essential \
    git \
    && \
    # Clean up the cache to keep the image small
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --default-timeout=1000  torch torchaudio xformers --index-url https://download.pytorch.org/whl/cu124

COPY requirements.txt .
RUN sed -i '/^torch/d' requirements.txt && \
    sed -i '/^torchvision/d' requirements.txt && \
    sed -i '/^torchaudio/d' requirements.txt && \
    sed -i '/^xformers/d' requirements.txt && \
    sed -i '/^unsloth/d' requirements.txt && \
    sed -i '/^transformers/d' requirements.txt && \
    sed -i '/^peft/d' requirements.txt

RUN pip install "unsloth @ git+https://github.com/unslothai/unsloth.git"

RUN pip install -r requirements.txt

RUN pip install --default-timeout=1000 transformers==4.56.2 \
    peft \
    trl \
    accelerate \
    bitsandbytes

COPY ./src ./src

EXPOSE 8000

CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "src.main:app", "-b", "0.0.0.0:8000"]