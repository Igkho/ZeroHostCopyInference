# Use NVIDIA's official image with TensorRT and CUDA pre-installed
FROM nvcr.io/nvidia/tensorrt:23.08-py3

# 1. Install System Dependencies (FFMpeg, CMake)
RUN apt-get update && apt-get install -y \
    pkg-config \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# 2. Copy Project Files
WORKDIR /app
COPY . .

# 3. Build The Project
RUN mkdir build && cd build \
    && cmake .. -DCMAKE_BUILD_TYPE=Release \
    && make -j$(nproc)

# 4. Set Entrypoint
ENTRYPOINT ["./build/ZeroCopyInference"]
CMD ["--help"]