# --- Stage 1: Builder for Data Processing and Model Indexing ---
FROM python:3.9-slim as builder

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    libssl-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files (excluding those in .dockerignore)
COPY . .

# --- CRITICAL FIX: Copy the pre-downloaded model weights to the correct cache path ---
# This bypasses the unreliable download step during the build.
RUN mkdir -p /root/.cache/torch/hub/checkpoints/
COPY efficientnet_b0_rwightman-3dd342df.pth /root/.cache/torch/hub/checkpoints/efficientnet_b0_rwightman-3dd342df.pth

# Ensure setup.sh is executable and run the setup process
RUN chmod +x setup.sh
RUN ./setup.sh

# --- Stage 2: Final Production Image (minimalist) ---
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy Python packages from the builder stage
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages

# Copy only the necessary files and pre-built artifacts from the builder stage
COPY --from=builder /app/app.py .
COPY --from=builder /app/config.py .
COPY --from=builder /app/models ./models
COPY --from=builder /app/utils ./utils
COPY --from=builder /app/data ./data
COPY --from=builder /root/.cache/torch/hub/checkpoints/efficientnet_b0_rwightman-3dd342df.pth /root/.cache/torch/hub/checkpoints/efficientnet_b0_rwightman-3dd342df.pth
COPY --from=builder /app/yolov8n.pt .

# --- CRITICAL FIX: Create 'templates' directory and copy index.html into it ---
RUN mkdir -p templates
COPY --from=builder /app/index.html /app/templates/index.html

# Create a directory for user uploads
RUN mkdir -p uploads

# Expose the port the Flask app runs on
EXPOSE 5000

# Set the entrypoint command for the application
CMD ["python", "app.py"]
