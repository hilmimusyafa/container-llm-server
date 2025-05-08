# Dockerfile for llama.cpp with CUDA support
FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04 

# Install dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    cmake \
    curl \
    libopenblas-dev \
    pkg-config \
    python3 \
    python3-pip \
    libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Clone llama.cpp
WORKDIR /opt
RUN git clone https://github.com/ggerganov/llama.cpp.git
WORKDIR /opt/llama.cpp

# First build without CUDA to ensure we get the executable
RUN cmake -B build
RUN cmake --build build --config Release -j4
RUN build/bin/llama-cli --completion-bash > ~/.llama-completion.bash
RUN source ~/.llama-completion.bash
RUN echo "source ~/.llama-completion.bash" >> ~/.bashrc

EXPOSE 4040

# Add the build/bin directory to PATH
ENV PATH="/opt/llama.cpp/build/bin:$PATH"

# Create model directory
RUN mkdir -p /models
WORKDIR /models
COPY ./models/current-model.gguf /models/current-model.gguf

# For GPU inference, we'll use the nvidia-container-runtime
# This will only work when running with --gpus all
CMD ["./"]