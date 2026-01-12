# Manus Sandbox - Python 3.11 with development tools
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    wget \
    git \
    vim \
    nano \
    htop \
    tree \
    jq \
    unzip \
    ca-certificates \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 20 LTS
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Create workspace directory
RUN mkdir -p /workspace

# Set working directory
WORKDIR /workspace

# Install common Python packages
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
    requests \
    beautifulsoup4 \
    lxml \
    httpx \
    pyyaml

# Create non-root user for security
RUN useradd -m -s /bin/bash agent \
    && chown -R agent:agent /workspace

# Keep container running
CMD ["tail", "-f", "/dev/null"]
