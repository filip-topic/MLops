FROM python:3.10-slim

# install Docker CLI so subprocess(["docker", ...]) will work
RUN apt-get update \
 && apt-get install -y --no-install-recommends docker.io \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Prefect + plugins only

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy orchestration code (keep it small)
#COPY flows/ /app/flows
COPY . .
WORKDIR /app/flows
CMD ["python", "training_flow.py"]
