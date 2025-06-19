FROM python:3.11-slim
WORKDIR /app
COPY compute_kl_divergence.py .
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
ENTRYPOINT ["python", "compute_kl_divergence.py"]