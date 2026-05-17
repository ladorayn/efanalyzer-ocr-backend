FROM python:3.10-slim

# Install system dependencies PaddleOCR needs (must be done as root)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set up a new user named "user" with UID 1000 for Hugging Face Spaces compatibility
RUN useradd -m -u 1000 user

# Switch to the new user
USER user

# Set up environment variables
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory
WORKDIR $HOME/app

# Copy requirements and install dependencies as the 'user'
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Download PaddleOCR models at build time so they're baked into the image under the user's home directory
RUN python -c "from paddleocr import PaddleOCR; PaddleOCR(use_angle_cls=True, lang='en', show_log=False)"

# Copy the rest of the application files with correct ownership
COPY --chown=user . .

# Expose Hugging Face's default port
EXPOSE 7860

# Run the application, defaulting to port 7860
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860}"]
