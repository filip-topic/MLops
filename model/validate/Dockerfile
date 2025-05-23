FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Runs the robustness checks against your MLflow artifacts
CMD ["python", "validate_robustness.py"]
