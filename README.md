# High-Performance Zero-Host-Copy Inference Pipeline (C++/CUDA)

![Status](https://img.shields.io/badge/Status-Active_Development-yellow)
![Platform](https://img.shields.io/badge/Platform-Linux_x64-blue)
![Porting](https://img.shields.io/badge/Porting-Windows_%7C_Jetson_Orin-yellow)
![Language](https://img.shields.io/badge/Language-C%2B%2B17_%7C_CUDA-green)

**Author:** Igor Khozhanov

**Contact:** khozhanov@gmail.com

**Copyright:** © 2026 Igor Khozhanov. All Rights Reserved.

---

## ⚠️ Current Development Status: Phase 3 (Integration)

**Note for Reviewers:** This repository is currently under active development. The pipeline is being implemented in stages to ensure memory safety and zero-host-copy verification.

| Module / Stage | Status | Notes |
| :--- | :--- | :--- |
| **FFMpeg Source** | ✅ **Stable** | Handles stream connection and packet extraction. |
|**Stub Detector** | ✅ **Stable** | Pass-through module, validated for pipeline latency profiling. |
| **Output / NVJpeg** | ✅ **Stable** | Saves frames from GPU memory to disk as separate *.jpg images. |
| **Inference Pipeline** | ✅ **Stable** | Connects all the stages together. |
| **ONNX Detector** | 🛠**Integration** | Implemented with `Ort::MemoryInfo` for Zero-Copy input. |
| **TensorRT Detector** | 🛠**Integration** | Engine builder & `enqueueV3` implemented; Dynamic shapes supported. |
| **Object Tracker** | 🚧 **WIP** | Kernels for position prediction & IOU matching. |
| **Post-Processing** | 🚧 **WIP** | Custom CUDA kernels for YOLOv8 decoding & NMS. |

---

## Project Overview
This project implements a high-performance video inference pipeline designed to minimize CPU-GPU bandwidth usage. Unlike standard OpenCV implementations, this pipeline keeps data entirely on the VRAM (Zero-Host-Copy) from decoding to inference.

## How to Build & Test (Current Version)

The current build verifies the **Decoding, Memory Allocation and Data Saving** stages.

## Compatibility

### Supported Platforms
* ✅ Linux x64 (Verified on Ubuntu 24.04 / RTX 3060 Ti)
* 🚧 Windows 10/11 (Build scripts implemented, pending validation)
* 🚧 Nvidia Jetson Orin (CMake configuration ready, pending hardware tests)

Note: The CMakeLists.txt contains specific logic for vcpkg (Windows) and aarch64 (Jetson), but these targets are currently experimental.

## Dependencies

### Build Time
* CMake 3.19+
* CUDA Toolkit (12.x)
* TensorRT 10.x+
* **FFmpeg**: Required.
    * *Linux Users:* Install via package manager or build from source with `--enable-shared`.

### Runtime Requirements
* **NVIDIA cuDNN**: Required by ONNX Runtime CUDA provider. 
    * *Note: 
    Ensure `libcudnn.so` is in your `LD_LIBRARY_PATH` or installed system-wide.*
    
## Compilation & Run

### Build

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Run pipeline

```bash
./ZeroCopyInference -i ../video/Moving.mp4 --backend stub -b 16 -o Moving
```
or
```bash
docker run --rm --gpus all \
  -v $(pwd)/video:/app/video \
  ghcr.io/igkho/zerohostcopyinference:main \
  -i video/Moving.mp4 --backend stub -b 16 -o video/output
```

### Run tests

```bash
./ZeroCopyInferenceTests
```
or
```bash
docker run --rm --gpus all \
  --entrypoint ./build/ZeroCopyInferenceTests \
  ghcr.io/igkho/zerohostcopyinference:main
```

## 🚀 Performance Benchmarks

Infrastructure overhead measured on **NVIDIA RTX 3060 Ti** (1440p Video):

| Metric | Result | Notes |
| :--- | :--- | :--- |
| **Max Pipeline Capacity** | **~300 FPS (No Model)** | Measured with Stub/Pass-through Detector. Represents the I/O ceiling (Decode → GPU Memory → Encode) before adding model latency |
| **I/O Latency** | **~3.3 ms** | Time spent on non-inference tasks, leaving **13ms+** (at 60FPS) purely for AI models. |
| **CPU Usage** | **Low** | Zero-Host-Copy ensures CPU only handles orchestration, not pixels. |

## ⚖️ License

The source code of this project is licensed under the **MIT License**. You are free to use, modify, and distribute this infrastructure code for any purpose, including commercial applications.

### ⚠️ Important Note on Model Licensing
While the C++ pipeline code is MIT-licensed, the **models** you run on it may have their own restrictive licenses.

* **Example:** If you use **YOLOv8** (Ultralytics) with this pipeline, be aware that YOLOv8 is licensed under **AGPL-3.0**.
* **Implication:** Integrating an AGPL-3.0 model may legally require your entire combined application to comply with AGPL-3.0 terms (i.e., open-sourcing your entire project).

**User Responsibility:** This repository provides the *execution engine* only. No models are bundled. You are responsible for verifying and complying with the license of any specific ONNX/TensorRT model you choose to load.