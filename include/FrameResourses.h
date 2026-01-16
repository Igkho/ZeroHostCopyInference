#pragma once
#include <memory>
#include "helpers.h"
#include "Block.h"
#include "BatchDetections.h"

namespace cropandweed {

struct FrameResources {
    // Direct fields: Metadata (size, ptr) lives inside FrameResources memory
    Block<float> rawOutput;
    Block<uint8_t> candidates;
    Block<int> candidateCount;
    
    // Even nested structs can hold direct Blocks
    BatchDetections result;

    Block<uint8_t> nmsMask;
    
    // CudaEvent must be Move-Only (or wrapped in unique_ptr if it has no default ctor)
    // Assuming you updated CudaEvent to be like Block (Move-Only wrapper):
    std::unique_ptr<CudaEvent> readyEvent; 

    // Default Ctor (Fast, no-op)
    FrameResources() = default;

    // Unified Initialization
    CudaError Init(int width, int height) {
        // Resize direct blocks. Returns error on OOM.
        CUDA_TRY(rawOutput.resize(width * height));
        CUDA_TRY(candidates.resize(1000));
        CUDA_TRY(candidateCount.resize(1));
        
        // Init nested items
        CUDA_TRY(result.data.resize(500));
        CUDA_TRY(nmsMask.resize(1));

        // Init Event
        CUDA_TRY(CudaEvent::Create(readyEvent, cudaStreamNonBlocking));

        return CudaError();
    }

    // Static Factory for the whole bundle
    static CudaError Create(std::shared_ptr<FrameResources>& out, int w, int h) {
        // 1. Single Allocation (using make_shared for efficiency)
        auto res = std::make_shared<FrameResources>();

        // 2. Error Propagating Init
        CUDA_TRY(res->Init(w, h));

        // 3. Success
        out = std::move(res);
        return CudaError();
    }
};

}
