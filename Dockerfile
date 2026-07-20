# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

# Copy dependency file first (better layer caching)
COPY requirements.txt ./requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application source. NOTE: never COPY .env or any GCP service-account
# JSON into the image — credentials are provided at runtime via env vars
# (ANTHROPIC_API_KEY, OPENFDA_API_KEY, GOOGLE_CLOUD_PROJECT) and, for
# BigQuery, a service-account key mounted at the path in
# GOOGLE_APPLICATION_CREDENTIALS (see render.yaml).
COPY server.py pipeline.py ./
COPY systems/ ./systems/
COPY utils/ ./utils/

# Expose the port uvicorn will listen on
EXPOSE 8000

# Start the server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
