# Dockerfile
FROM python:3.9-slim

# Install system deps
RUN apt-get update && apt-get install -y gcc

# Set workdir
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of the code
COPY . .

# Prefect needs a home directory set (optional)
ENV PREFECT_HOME=/app/.prefect

# Run the full ML flow on container start
CMD ["python", "flows/training_flow.py"]
