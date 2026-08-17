FROM python:3.10-slim

WORKDIR /app

# Install dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model artifacts
COPY app/ ./app/
COPY artifacts/ ./artifacts/

EXPOSE 8000

# Railway (and most cloud platforms) inject a $PORT env var at runtime.
# Shell form (not exec/array form) is required so $PORT gets substituted.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
