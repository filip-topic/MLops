FROM python:3.11-slim
WORKDIR /app
COPY get_latest_two_run_ids.py .
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY get_latest_two_run_ids.py .
CMD ["python", "get_latest_two_run_ids.py"]