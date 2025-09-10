# OpenCV-compatible container with full GL support
FROM python:3.10-slim AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DISPLAY=:99

# Install OpenGL and OpenCV dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgthread-2.0-0 \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    libjpeg62-turbo \
    libpng16-16 \
    libtiff6 \
    wget \
    curl \
    git \
    build-essential \
    mesa-utils \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements_docker_headless.txt .

# Upgrade pip first
RUN pip install --upgrade pip

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_docker_headless.txt

# Copy the entire project
COPY . .

# Create necessary directories
RUN mkdir -p outputs/ground_truth outputs/predictions outputs/mapping

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Start virtual display and run app
CMD ["sh", "-c", "Xvfb :99 -screen 0 1024x768x24 & python -m streamlit run src/webapp/main.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true"]
