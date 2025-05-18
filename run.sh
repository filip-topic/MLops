#!/bin/bash

echo "Building Docker image..."
docker build -t data-quality-check .

echo "Running Docker container..."
docker run data-quality-check