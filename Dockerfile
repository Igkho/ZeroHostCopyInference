# Use NVIDIA's official image with TensorRT and CUDA pre-installed
FROM ghcr.io/igkho/tensorrt:24.07-py3

# 1. Install System Dependencies (FFMpeg, CMake)
RUN apt-get update && apt-get install -y \
    pkg-config \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    cmake \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# 2. Copy Project Files
WORKDIR /app
COPY . .

# 3. Build The Project
RUN mkdir build && cd build \
    && cmake .. -DCMAKE_BUILD_TYPE=Release -DFETCH_TEST_DATA=OFF \
    && make -j$(nproc)

# 4. Set Entrypoint
ENTRYPOINT ["./build/ZeroCopyInference"]
CMD ["--help"]
