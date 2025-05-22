# run_data_tests Dockerfile

FROM python:3.10-slim

WORKDIR /app

# Dependencies first for layer-caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Then the code
COPY . .

CMD ["python", "main.py"]