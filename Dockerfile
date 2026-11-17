FROM python:3.11-slim

WORKDIR /app

# Install curl for diagnostics
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies directly to global site-packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Ensure data directories exist
RUN mkdir -p /app/data/index

ENV PYTHONUNBUFFERED=1
ENV PORT=8000
EXPOSE 8000

# Shell-form CMD automatically expands Render's dynamic $PORT
CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT
