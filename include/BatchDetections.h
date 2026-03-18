#pragma once
#include <memory>
#include "helpers.h"
#include "Block.h"
#include "DetectionRaw.h"

namespace cropandweed {

struct BatchDetections {
    static constexpr int MAX_DETECTIONS_PER_FRAME = 1000;

    // 1. Direct Block Members
    BoundaryTypedBlock<DetectionRaw> data;
    BoundaryBlock<int> counts;

    // 2. Synchronization
    std::unique_ptr<CudaEvent> readyEvent;

    // --- Unified Initialization ---
    CudaError Init(size_t batchSize, cudaStream_t stream = 0) {
        // Allocate Raw Data buffer
        // Size calculation handled here cleanly
//        size_t totalBytes = batchSize * MAX_DETECTIONS_PER_FRAME * sizeof(DetectionRaw); // Assuming DetectionRaw is defined
//        CUDA_TRY(data.resize(totalBytes, stream));
        size_t totalElements = batchSize * MAX_DETECTIONS_PER_FRAME;
        CUDA_TRY(data.resize(totalElements, stream));

        // Allocate Counts buffer
        CUDA_TRY(counts.resize(batchSize, stream));

        // Init Event
        CUDA_TRY(CudaEvent::Create(readyEvent, cudaStreamNonBlocking));

        return CudaError();
    }

    // --- Static Factory ---
    static CudaError Create(std::shared_ptr<BatchDetections>& out, size_t batchSize,
                            cudaStream_t stream = 0) {
        auto res = std::make_shared<BatchDetections>();
        CUDA_TRY(res->Init(batchSize, stream));
        out = std::move(res);
        return CudaError();
    }
};

}
