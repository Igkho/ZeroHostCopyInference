#include "StubDetector.h"
#include "DetectionRaw.h"
#include <iostream>

namespace cropandweed {

StubDetector::StubDetector(Token) {}

CudaError StubDetector::Init() {
    CUDA_TRY(CudaStream::Create(cuda_stream_, cudaStreamNonBlocking));

    pool_.resize(POOL_SIZE);

    // Pre-allocate memory for the pool to ensure zero-allocation during runtime.
    // We assume a max batch size of 8 for pre-allocation purposes.
    int maxBatch = 8;
    size_t totalBytes = maxBatch * BatchDetections::MAX_DETECTIONS_PER_FRAME * sizeof(DetectionRaw);

    for (auto& bd : pool_) {
        // Initialize the BatchDetections struct (creates readyEvent)
        CUDA_TRY(bd.Init(maxBatch));

        // Reserve maximum capacity so resize() calls in Detect() don't trigger malloc
        CUDA_TRY(bd.data.reserve(totalBytes));
        CUDA_TRY(bd.counts.reserve(maxBatch));
    }

    std::cout << "[StubDetector] Initialized with pool size " << POOL_SIZE << std::endl;
    return CudaError();
}

std::pair<size_t, size_t> StubDetector::GetInputSize() const {
    // Stub accepts any size, but we return a standard resolution
    return {1024, 1024};
}

CudaError StubDetector::Detect(const BatchData& input, BatchDetections& output) {
    BatchDetections& res = pool_[poolIndex_];
    poolIndex_ = (poolIndex_ + 1) % POOL_SIZE;

    // 1. Wait for Input Data
    if (input.readyEvent) {
        CUDA_TRY(cudaStreamWaitEvent(*cuda_stream_, *input.readyEvent, 0));
    }

    // 2. Resize Buffers
    // Since we reserved memory in Init, this is a metadata-only change (Zero Allocation)
    size_t bytesNeeded = input.batchSize * BatchDetections::MAX_DETECTIONS_PER_FRAME * sizeof(DetectionRaw);
    CUDA_TRY(res.data.resize(bytesNeeded));
    CUDA_TRY(res.counts.resize(input.batchSize));

    // Ensure Event exists (in case the swapped-in structure didn't have one)
    if (!res.readyEvent) {
        CUDA_TRY(CudaEvent::Create(res.readyEvent));
    }

    // 3. "Run" Stub Inference
    // Just zero out the memory to simulate processing and ensure clean state
    CUDA_TRY(cudaMemsetAsync(res.data.data(), 0, bytesNeeded, *cuda_stream_));
    CUDA_TRY(cudaMemsetAsync(res.counts.data(), 0, input.batchSize * sizeof(int), *cuda_stream_));

    // 4. Record Completion
    CUDA_TRY(cudaEventRecord(*res.readyEvent, *cuda_stream_));

    // 5. Swap to Output
    // 'output' (from the pipeline) is swapped into 'res' (the pool).
    // The pipeline gets the valid buffer, and the pool gets the old recycled buffer.
    std::swap(output, res);

    return CudaError();
}

} // namespace cropandweed
