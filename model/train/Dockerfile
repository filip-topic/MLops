FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip

RUN pip install --no-cache-dir --timeout=180 -r requirements.txt

COPY . .

# When run, this will execute your training script
CMD ["python", "train.py"]

