FROM python:3.11-slim

# Install system dependencies required by PaddleOCR and OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose port (default for uvicorn)
EXPOSE 8000

# Run the application, using Railway's PORT environment variable or default to 8000
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}