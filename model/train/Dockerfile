# train_model Dockerfile

FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expect the dataset and (optionally) an ./mlruns volume to be mounted at runtime
CMD ["python", "train.py"]
