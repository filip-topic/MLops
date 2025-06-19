FROM python:3.11-slim
WORKDIR /app
COPY run_predictions.py .
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
CMD ["python", "run_predictions.py"]