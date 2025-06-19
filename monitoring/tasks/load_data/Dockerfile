FROM python:3.11-slim
WORKDIR /app
COPY load_data.py .
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
ENTRYPOINT ["python", "load_data.py"]