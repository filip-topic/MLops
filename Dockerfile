FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create model directory if it doesn't exist
RUN mkdir -p model

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Command to run Prefect ML pipeline
CMD ["python", "src/flow.py"]
CMD ["pytest", "pre_deployment_tests/", "-v", "--cov=.", "--cov-report=xml:coverage.xml"]