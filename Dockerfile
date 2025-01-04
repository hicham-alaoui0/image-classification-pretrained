# docker/Dockerfile

# Use official Python image as base
FROM python:3.8-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy project
COPY . .

# Expose port for FastAPI
EXPOSE 8000

# Run the inference server
CMD ["python", "src/inference.py"]
