#pragma once
#include "helpers.h"
#include "Block.h"
#include "ObjectTrackerKernels.h"
#include <memory>

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
    static CudaError Create(std::unique_ptr<ObjectTracker>& out,
                            int numClasses,
                            cudaStream_t stream,
                            int maxTracks = TRACKER_MAX_TRACKS) {
        auto ptr = std::make_unique<ObjectTracker>(Token{});
        CUDA_TRY(ptr->Init(maxTracks, numClasses, stream));
        out = std::move(ptr);
        return CudaError();
    }

    // Updates state based on detections
    CudaError ProcessBatch(int batchIndex,
                           BoundaryTypedBlock<DetectionRaw>& detections,
                           BoundaryBlock<int>& countBuffer,
                           int maxDetections,
                           int width,
                           int height,
                           cudaStream_t stream);

    CudaError Compact(cudaStream_t stream);

    // Draws the current state (boxes + IDs) onto the image
    CudaError Annotate(Block<float>& imageBatch,
                       int batchSize, 
                       int width, 
                       int height,
                       const BoundaryTypedBlock<DetectionRaw>& detections,
                       const BoundaryBlock<int>& counts,
                       cudaStream_t stream);

private:
    // Initialization is now internal to the Factory
    CudaError Init(int maxTracks, int numClasses, cudaStream_t stream);

    TypedBlock<TrackState> tracks_;
//    Block<uint8_t> tracks_;
    Block<int> trackCount_;
    Block<int> nextTrackId_;
    Block<int> detectionMatches_; 

    int maxTracks_ = 0;
    int numClasses_ = 0;
};

} // namespace cropandweed
