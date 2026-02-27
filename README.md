# High-Performance Zero-Host-Copy Inference Pipeline (C++/CUDA)

![Status](https://img.shields.io/badge/Status-Active_Development-yellow)
![Platform](https://img.shields.io/badge/Platform-Linux_x64-blue)
![Porting](https://img.shields.io/badge/Porting-Windows_%7C_Jetson_Orin-yellow)
![Language](https://img.shields.io/badge/Language-C%2B%2B17_%7C_CUDA-green)

**Author:** Igor Khozhanov

**Contact:** khozhanov@gmail.com

**Copyright:** © 2026 Igor Khozhanov. All Rights Reserved.

---

## 🎬 Real-Time Output
*Processing 1440p Video Stream @ ~165 FPS on RTX 3060 Ti.*

![Crop & Weed Detection Demo](video/Moving_annotated.gif)

---

## ⚠️ Current Development Status: Phase 5 (Porting)

The previous phases **Phase 3 (Integration)** and **Phase 4 (Functional Inference)** completed. The pipeline now supports full end-to-end detection and tracking using TensorRT and ONNX backends with mathematically verified kernels.

**Note for Reviewers:** This repository is currently under active development. The pipeline is being implemented in stages to ensure memory safety and zero-host-copy verification.

| Module / Stage | Status | Notes |
| :--- | :--- | :--- |
| **FFMpeg Source** | ✅ **Stable** | Handles stream connection and packet extraction. |
|**Stub Detector** | ✅ **Stable** | Pass-through module, validated for pipeline latency profiling. |
| **Output / NVJpeg** | ✅ **Stable** | Saves frames from GPU memory to disk as separate *.jpg images. |
| **Inference Pipeline** | ✅ **Stable** | Connects all the stages together. |
| **ONNX Detector** | ✅ **Stable** | Implemented with Zero-Copy input. |
| **TensorRT Detector** | ✅ **Stable** | Engine builder & `enqueueV3` implemented. |
| **Object Tracker** | ✅ **Stable** | Kernels for position prediction, IOU matching, velocity filtering. |
| **Post-Processing** | ✅ **Stable** | Custom CUDA kernels for YOLOv8 output decoding & NMS. |
| **Windows Port** | 🚧 **WIP** | Adapting CMake & CUDA |
| **Jetson Port** | 🚧 **WIP** | ARM64 optimization. |

---

## Project Overview
This project implements a high-performance video inference pipeline designed to minimize CPU-GPU bandwidth usage. Unlike standard OpenCV implementations, this pipeline keeps data entirely on the VRAM (Zero-Host-Copy) from decoding to inference.

## 📥 Model Setup (Required)

This repository contains the **Inference Engine** (MIT Licensed). It does **not** include pre-trained model weights.

To reproduce the demo results (Crop & Weed Detection), you must download pre-trained YOLOv8 model separately.

### Download the Model
The model is hosted in the research repository (AGPL-3.0):

* **Download:** [`best_int8.onnx`](https://github.com/Igkho/CropAndWeedDetection/releases) 
* **License:** AGPL-3.0 (Derived from Ultralytics YOLOv8)

## How to Build & Test (Current Version)

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

### Build & Run (Native)

```bash
git clone https://github.com/Igkho/ZeroHostCopyInference.git
cd ZeroHostCopyInference

mkdir build
mv ~/Downloads/best_int8.onnx ./build/

cd build
cmake ..
make -j$(nproc)
```

### Run pipeline

```bash
./ZeroCopyInference -i ../video/Moving.mp4 --backend trt --model best_int8.onnx -b 16 -o Moving
```

### Run tests

```bash
./ZeroCopyInferenceTests
```

### Quick Start (Docker)
No C++ compilation required. Requires NVIDIA Container Toolkit.

### Run pipeline

```bash

git clone https://github.com/Igkho/ZeroHostCopyInference.git
cd ZeroHostCopyInference

mkdir models
mv ~/Downloads/best_int8.onnx ./models/

docker run --rm --gpus all \
  -v $(pwd)/video:/app/video \
  -v $(pwd)/models:/app/models \
  ghcr.io/igkho/zerohostcopyinference:main \
  -i video/Moving.mp4 \
  --backend trt \
  --model /app/models/best_int8.onnx \
  -b 16 \
  -o video/output
```

### Run tests

```bash

docker run --rm --gpus all \
  --entrypoint ./build/ZeroCopyInferenceTests \
  ghcr.io/igkho/zerohostcopyinference:main
```

## 🚀 Performance Benchmarks

Benchmarks performed on **NVIDIA RTX 3060 Ti**.
**Input:** 1440p Video Stream.
**Model:** YOLOv8 Medium (YOLOv8m) @ 1024x1024 Resolution.

### 1. Infrastructure Ceiling (Stub Mode)
To measure the raw overhead of the pipeline architecture (I/O latency), a pass-through (Stub) detector should be used.

| Metric | Result | Notes |
| :--- | :--- | :--- |
| **Throughput** | **~300 FPS** | Maximum theoretical speed without AI model. |
| **Latency** | **3.3 ms** | Combined Decoding + Memory Management overhead. |

### 2. Real-World Inference (TensorRT INT8 Mode)
Running **YOLOv8m** (Explicitly Quantized INT8) with full object tracking and NVJpeg output.

| Metric | Result | Notes |
| :--- | :--- | :--- |
| **Total Throughput** | **~165 FPS** | Wall time (End-to-End). **>2.5x Real-Time**. |
| **Bottleneck Shift** | **Decoding** | Inference is now so fast (3.36ms) that Video Decoding (4.83ms) has become the primary bottleneck. |

**Workload Distribution (Active Work):**
* **Decoding (Source):** ~4.83 ms/frame (46.91% load)
* **Inference (Detector):** ~3.36 ms/frame (32.66% load)
* **Storage (Sink):** ~2.10 ms/frame (20.43% load)


### 3. Backend & Precision Comparison
Both **TensorRT** (Highly Optimized) and **ONNX Runtime** (Generic Compatibility) are supported.

**Scenario:** 1024x1024 Input Resolution on RTX 3060 Ti.

| Backend / Precision | FPS | Latency (Inference) | Speedup Factor | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **TensorRT (INT8)** | **~164.8** | **~3.36 ms** | **1.4x** | Maximum performance. **Recommended.** |
| **TensorRT (FP16)** | **~118.1** | **~5.58 ms** | **1.0x (Ref)** | Baseline hardware acceleration. |
| **ONNX Runtime** | **~10.5** | **~94.8 ms** | **0.08x** | Generic execution. Useful for testing new models. |


## ⚖️ License

The source code of this project is licensed under the **MIT License**. You are free to use, modify, and distribute this infrastructure code for any purpose, including commercial applications.

### 🛑 Asset & Model Licensing Exceptions

While the code is MIT-licensed, the **assets and models** used in this repository are subject to different terms. Please review them carefully before redistributing:

#### 1. Video Assets (Non-Commercial Only)
* **Files:** Content located in the `video/` directory (e.g., `Moving.mp4`, `Moving_annotated.gif`).
* **Source:** Generated using **KlingAI (Free Tier)**.
* **Terms:** These assets are provided for **demonstration and educational purposes only**. They are **strictly non-commercial**. You may not use these specific video files in any commercial product or service.
* **Attribution:** The watermarks on these videos must remain intact as per the platform's Terms of Service.

### 2. Model Licensing

* **Example:** If you use **YOLOv8** (Ultralytics) with this pipeline, be aware that YOLOv8 is licensed under **AGPL-3.0**.
* **Implication:** Integrating an AGPL-3.0 model may legally require your entire combined application to comply with AGPL-3.0 terms (i.e., open-sourcing your entire project).

**User Responsibility:** This repository provides the *execution engine* only. No models are bundled. You are responsible for verifying and complying with the license of any specific ONNX/TensorRT model you choose to load.