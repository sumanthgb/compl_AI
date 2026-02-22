# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Set working directory to where server.py and all modules live
WORKDIR /app

# Copy dependency file first (better layer caching)
COPY compl_ai/requirements.txt ./requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire compl_ai package
COPY compl_ai/ .

# Expose the port uvicorn will listen on
EXPOSE 8000

# Start the server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
