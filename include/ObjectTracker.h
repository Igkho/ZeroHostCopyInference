#pragma once
#include "helpers.h"
#include "Block.h"
#include "ObjectTrackerKernels.h"
#include <memory>
//#include <type_traits> // for is_shared_ptr_v logic in helpers.h

namespace cropandweed {

class ObjectTracker {
private:
    // Passkey Idiom: Prevents direct construction via new/make_unique outside of Create()
    struct Token {};

public:
    // Constructor requires Token
    ObjectTracker(Token) {}
    ~ObjectTracker() {}

    /**
     * @brief Factory method to create and initialize an ObjectTracker.
     * Handles memory allocation and error propagation.
     */
    template <typename SmartPtr>
    static CudaError Create(SmartPtr& out, int maxTracks = 2000) {
        if constexpr (is_shared_ptr_v<SmartPtr>) {
            auto ptr = std::make_shared<ObjectTracker>(Token{});
            CUDA_TRY(ptr->Init(maxTracks));
            out = std::move(ptr);
        } else {
            auto ptr = std::make_unique<ObjectTracker>(Token{});
            CUDA_TRY(ptr->Init(maxTracks));
            out = std::move(ptr);
        }
        return CudaError();
    }

    // Updates state based on detections
    CudaError ProcessBatch(int batchIndex, 
                           DetectionRaw* detections, 
                           int* countBuffer, 
                           int maxDetections,
                           cudaStream_t stream);

    // Draws the current state (boxes + IDs) onto the image
    CudaError Annotate(float* imageBatch, 
                       int batchSize, 
                       int width, 
                       int height, 
                       const DetectionRaw* detections, 
                       const int* counts,
                       cudaStream_t stream);

private:
    // Initialization is now internal to the Factory
    CudaError Init(int maxTracks);

    Block<uint8_t> tracks_;
    Block<int> trackCount_;
    Block<int> nextTrackId_;
    Block<int> detectionMatches_; 

    int maxTracks_ = 0;
};

} // namespace cropandweed
