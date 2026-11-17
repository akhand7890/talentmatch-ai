# Multi-stage production Dockerfile for TalentMatch AI Agent
FROM python:3.11-slim as builder

WORKDIR /build
COPY requirements.txt .

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && pip install --no-cache-dir --user -r requirements.txt

# Final Runtime Image
FROM python:3.11-slim as runner

WORKDIR /app

# Install runtime tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Copy application source code
COPY . /app

# Ensure data directories exist
RUN mkdir -p /app/data/index

EXPOSE 8000
ENV PORT=8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
