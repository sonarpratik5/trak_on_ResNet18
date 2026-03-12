# 1. Your proven base image
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# 2. Where everything lives inside the container
WORKDIR /app

# Tell the OS to never pause and ask for human input
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# 3. Copy and install Python dependencies first
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# 4. Copy the rest of the project
COPY . .

# 5. Fix the Python Path (removes that warning you saw earlier)
ENV PYTHONPATH="/app"

# 6. Create a directory for checkpoints so it doesn't crash on save
RUN mkdir -p checkpoints

# 7. Create a non-root user and give permissions (Great for HTCondor!)
RUN useradd -m app && chown -R app:app /app
USER app

# 8. Run your new script structure
ENTRYPOINT ["python", "/app/main.py"]
CMD ["--mode", "all"]