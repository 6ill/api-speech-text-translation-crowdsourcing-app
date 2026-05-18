# Speech-to-Text & Translation App with Continual Learning

## Background
The primary objective of this project is to build an end-to-end crowdsourcing application capable of Automatic Speech Recognition (ASR) and Machine Translation (MT). To ensure the AI models adapt to new vocabulary and domain-specific context over time without losing previously acquired knowledge, the system implements an automated **Continual Learning (CL)** pipeline. This pipeline utilizes **LoRA (Low-Rank Adaptation)** and **QLoRA (4-bit Quantization)** to fine-tune OpenAI's Whisper (ASR) and Qwen3 (MT) efficiently on consumer-grade hardware (e.g., RTX 4080 16GB).

## System Architecture & Services

The application is fully containerized using Docker Compose and consists of several interconnected microservices:

1. **`postgres` (Database):** A PostgreSQL 15 instance that stores application data (users, audio metadata, text corrections) and serves as the backend store for MLflow tracking.
2. **`redis` (Message Broker):** In-memory data store used by Celery to queue and route asynchronous tasks.
3. **`object-storage` (RustFS):** An S3-compatible object storage service handling raw audio files, static evaluation datasets, and MLflow model artifacts (adapters).
4. **`mlflow` (Model Registry):** Tracks fine-tuning experiments, metrics (WER/BLEU), and acts as the central registry for production-ready LoRA adapters.
5. **`backend` (FastAPI):** The main REST API handling client requests, audio uploads, data retrieval, and routing inference requests to workers.
6. **`worker_inference` (Celery):** A dedicated GPU worker that handles real-time inference tasks. It dynamically fetches the latest production LoRA adapters from MLflow and attaches them to the base models.
7. **`worker_pipeline` (Celery):** A dedicated GPU worker handling heavy ETL and ML training tasks. It extracts user corrections, trains new adapters, evaluates them, and pushes successful models to MLflow.
8. **`celery_beat` (Scheduler):** Periodically checks the database for pipeline configurations and triggers the `worker_pipeline` based on CRON schedules.

## Continual Learning Flow

The intelligence of the system relies on a continuous feedback loop:

1. **Crowdsourcing & Inference:** A user uploads an audio file or text. The `worker_inference` processes it using the latest active model and returns the result. The user can submit a **correction** if the AI made a mistake.
2. **Triggering the Pipeline:** `celery_beat` runs periodically. If the accumulated corrections exceed a defined threshold (e.g., 50 samples), it triggers the CL pipeline.
3. **Dataset Preparation:** The system gathers the new corrections and mixes them with a **20% replay ratio** of historical data. This acts as a safeguard against catastrophic forgetting.
4. **Adapter Fine-Tuning:**
    * **Cold Start:** If no previous adapter exists, the system initializes a fresh LoRA adapter on the base model (Whisper or Qwen).
    * **Continual Learning:** If a previous adapter exists in MLflow, it is downloaded and "unlocked" (`is_trainable=True`) to resume training on the new data batch.
5. **Comparative Evaluation:** The newly trained adapter is evaluated against a static test dataset. Its score (WER for ASR, BLEU for MT) is compared against the current production baseline.
6. **Promotion & Deployment:** If the new model surpasses the improvement threshold, it is registered in MLflow and tagged as `staging`.

## Setup & Installation Instructions

### Prerequisites
* Linux/WSL2 environment
* Docker & Docker Compose plugin
* NVIDIA GPU (e.g., RTX 4080) with `nvidia-container-toolkit` installed.

### 1. Environment Configuration
Create a `.env` file in the root directory based on `.env.example` (or configure it manually). Ensure your storage and MLflow URLs are set up correctly for internal Docker routing:

```env
# Internal Docker URLs
DATABASE_URL="postgresql+asyncpg://postgres:postgres@postgres:5432/stt_app"
CELERY_BROKER_URL="redis://redis:6379/0"
STORAGE_ENDPOINT_URL="http://object-storage:9000"
MLFLOW_S3_ENDPOINT_URL="http://object-storage:9000"

# External URL (For frontend Presigned URLs)
STORAGE_EXTERNAL_URL="http://localhost:9100"

# Hugging Face Token (Required for accessing model from huggingface repo)
HF_TOKEN="your_hf_token_here"
```

### 2. Start Infrastructure Services First
Before building the heavy ML workers, start the core infrastructure to set up databases and buckets.

```bash
docker compose up -d postgres redis object-storage
```

### 3. Initialize Databases & Storage Buckets
#### A. Databases & Migrations:
Access the Postgres container or use a DB client to ensure both `stt_app` and `mlflow_db` databases are created. Then, apply the Alembic migrations to build the tables and seed the initial pipeline configurations:

```bash
alembic upgrade head
```
#### B. Storage Buckets:
Open the RustFS web console (http://localhost:9101) and manually create the following three buckets:
- `mlflow-artifacts` (For MLflow model registry)
- `audio-files` (For user audio uploads)
- `test-dataset `(For static evaluation datasets)

### 4. Prepare & Upload Test Datasets
The Continual Learning pipeline requires static test datasets to evaluate model improvements.
- ASR Test Dataset: We recommend using the [GigaSpeech 2 dataset](https://huggingface.co/datasets/speechcolab/gigaspeech2). Format the audio and transcripts into a Hugging Face Dataset format, zip the directory, and upload it to the `test-dataset` bucket with the exact key: `asr/asr_test_set.zip`.
- MT Test Dataset: We recommend using the [OpenSubtitles (id-en)](https://opus.nlpl.eu/datasets/OpenSubtitles?pair=id&en) dataset. Format the pairs into a JSONL file with source_text and target_text keys, and upload it to the `test-dataset` bucket with the exact key: `mt/mt_test_set.jsonl`.

*(Note: These keys must match the ones seeded in your database by Alembic).*

### 5. Build and Run the Full Application
Once the infrastructure and data are ready, build and start the remaining services (FastAPI, MLflow, and Celery Workers):
```bash
docker compose build --no-cache
docker compose up -d
```

### 6. Verify the Services
Check the logs to ensure the backend and workers have started successfully and connected to the GPU:
```bash
docker compose logs -f backend
docker compose logs -f worker_inference
```
### 7. Access the Interfaces
- FastAPI Swagger Docs: http://localhost:8000/docs
- MLflow UI: http://localhost:5000
- RustFS S3 Console: http://localhost:9101

### 8. Manual Adapter Registration (Optional)
If you have a pre-trained adapter from a Kaggle notebook, you can bypass the pipeline and manually register it to MLflow to initialize the Continual Learning chain:

```bash
docker exec -it <backend_container_name> bash
python scripts/register_adapter_to_mlflow.py
```