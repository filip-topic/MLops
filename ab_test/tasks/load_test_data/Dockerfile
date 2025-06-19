FROM python:3.11-slim
WORKDIR /app
COPY load_test_data.py .
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
CMD ["python", "load_test_data.py"]