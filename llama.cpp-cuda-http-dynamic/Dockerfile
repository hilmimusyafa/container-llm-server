FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04 

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

WORKDIR /opt
RUN git clone https://github.com/ggerganov/llama.cpp.git
WORKDIR /opt/llama.cpp

RUN cmake -B build
RUN cmake --build build --config Release -j4

EXPOSE 4040

ENV PATH="/opt/llama.cpp/build/bin:$PATH"

RUN mkdir -p /models
WORKDIR /models

CMD ["llama-server", "-m", "/models/current-model.gguf", "--port", "4040", "-c", "16384", "-np", "4", "--n-gpu-layers", "35", "--host", "0.0.0.0", "--nowebserver"]