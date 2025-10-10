# Use NVIDIA CUDA base image with development tools
FROM nvidia/cuda:13.0.1-devel-ubuntu24.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CPM_SOURCE_CACHE=/root/.cache/CPM \
    PATH=/root/.local/bin:$PATH

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Build tools
    build-essential \
    libssl-dev \
    ninja-build \
    make \
    g++ \
    git \
    # Python
    python3.12 \
    python3.12-dev \
    python3-pip \
    # Testing
    libgtest-dev \
    # Utilities
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install cmake
ARG CMAKE_VERSION=4.1.2
WORKDIR /tmp
RUN wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz
RUN tar -xzvf cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz
RUN mv cmake-${CMAKE_VERSION}-linux-x86_64 /opt/cmake
RUN ln -sf /opt/cmake/bin/* /usr/local/bin/
RUN rm -rf cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz

# Install uv package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Set working directory
WORKDIR /workspace

# Install Python dependencies with uv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

# Copy project files
COPY cmake/ ./cmake/
COPY CMakeLists.txt Makefile ./
COPY . .

# Build the project
RUN make build-release

# Set default command to run tests
CMD ["make", "test"]

