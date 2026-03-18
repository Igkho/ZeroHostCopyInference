#include "ObjectTracker.h"
#include <iostream>
#include "BatchDetections.h"

namespace cropandweed {

CudaError ObjectTracker::Init(int maxTracks, int numClasses, cudaStream_t stream) {
    // Store class count
    numClasses_ = (numClasses > TRACKER_MAX_CLASSES) ? TRACKER_MAX_CLASSES : numClasses;
    if (numClasses_ <= 0) numClasses_ = 1;
    // Enforce safe capacity
    maxTracks_ = (maxTracks < TRACKER_MAX_TRACKS) ? TRACKER_MAX_TRACKS : maxTracks;

    CUDA_TRY(tracks_.resize(maxTracks_, stream));

    CUDA_TRY(trackCount_.resize(1, stream));
    CUDA_TRY(trackCount_.fill(0, stream));

    CUDA_TRY(nextTrackId_.assign({1}, stream));

    // Buffer for matches needs to fit largest possible detections in a batch
    size_t matchesSize = BatchDetections::MAX_DETECTIONS_PER_FRAME; // * sizeof(int);
    CUDA_TRY(detectionMatches_.resize(matchesSize, stream));

    return CudaError();
}

CudaError ObjectTracker::ProcessBatch(int batchIndex,
                                      BoundaryTypedBlock<DetectionRaw> &detections,
                                      BoundaryBlock<int> &countBuffer,
                                      int maxDetectionsStride,
                                      int width,
                                      int height,
                                      cudaStream_t stream)
{
    CUDA_TRY(TrackBatch(batchIndex,
                        detections,
                        countBuffer,
                        tracks_,
                        trackCount_,
                        nextTrackId_,
                        detectionMatches_,
                        maxDetectionsStride,
                        maxTracks_,
                        numClasses_,
                        TRACKER_ALPHA,
                        width,
                        height,
                        stream
                       ));
    return CudaError();
}

CudaError ObjectTracker::Compact(cudaStream_t stream) {
    return CompactTracks(tracks_, trackCount_, maxTracks_, stream);
}

CudaError ObjectTracker::Annotate(Block<float>& imageBatch,
                                  int batchSize,
                                  int width,
                                  int height,
                                  const BoundaryTypedBlock<DetectionRaw> &detections,
                                  const BoundaryBlock<int> &counts,
                                  cudaStream_t stream)
{
    return DrawDetections(
        imageBatch,
        batchSize,
        width,
        height,
        detections,
        counts,
        stream
        );
}

} // namespace cropandweed
