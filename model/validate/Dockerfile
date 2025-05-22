# validate_model Dockerfile

FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expect dataset + mlruns volume
CMD ["python", "validate_robustness.py"]
