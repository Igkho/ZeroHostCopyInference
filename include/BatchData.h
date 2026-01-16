#pragma once
#include <memory>
#include "helpers.h"
#include "Block.h"

namespace cropandweed {

struct BatchData {
    int batchId = 0;
    size_t batchSize = 0;
    size_t width = 0;
    size_t height = 0;
    static constexpr size_t MAX_BATCH_SIZE = 20;
    static constexpr size_t OPTIMUM_BATCH_SIZE = 16; // for TRT engine optimization

    // 1. Direct Member: Metadata lives here, GPU data lives in VRAM
    Block<float> deviceData;

    // 2. Standard Vector (CPU side)
    std::vector<std::string> sourceIdentifiers;

    // 3. Event Wrapper (Move-only, managed by unique_ptr for lazy init)
    std::unique_ptr<CudaEvent> readyEvent;

    // --- Unified Initialization ---
    CudaError Init(size_t bSize, size_t w, size_t h) {
        batchSize = bSize;
        width = w;
        height = h;

        // One resize call, error propagates automatically
        CUDA_TRY(deviceData.resize(batchSize * width * height * 3)); // Assuming 3 channels (RGB/Planar)

        // Init Event
        CUDA_TRY(CudaEvent::Create(readyEvent, cudaStreamNonBlocking));

        // Reserve CPU strings (optional optimization)
        sourceIdentifiers.resize(batchSize);

        return CudaError();
    }

    // --- Static Factory ---
    static CudaError Create(std::shared_ptr<BatchData>& out, int id, size_t bSize, size_t w, size_t h) {
        auto res = std::make_shared<BatchData>();
        res->batchId = id;
        CUDA_TRY(res->Init(bSize, w, h));
        out = std::move(res);
        return CudaError();
    }
};

}
