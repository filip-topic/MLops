# Use a slim Python base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy only requirements first for better cache reuse
COPY requirements.txt .

# Install the step-specific dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the tests code
COPY . .

# Default command to run the tests
CMD ["python", "main.py"]
