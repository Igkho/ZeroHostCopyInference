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

//    CUDA_TRY(tracks_.resize(maxTracks_ * sizeof(TrackState), stream));
    CUDA_TRY(tracks_.resize(maxTracks_, stream));
//    CUDA_TRY(cudaMemsetAsync(tracks_.data(), 0, tracks_.byte_size(), stream)));

    CUDA_TRY(trackCount_.resize(1, stream));
    CUDA_TRY(cudaMemsetAsync(trackCount_.data(), 0, trackCount_.byte_size(), stream));

    nextTrackId_.assign({1}, stream);

//    CUDA_TRY(nextTrackId_.resize(1, stream));
//    int startId = 1;
//    CUDA_TRY(cudaMemcpyAsync(nextTrackId_.data(), &startId, sizeof(int), cudaMemcpyHostToDevice));

    // Buffer for matches needs to fit largest possible detections in a batch
    size_t matchesSize = BatchDetections::MAX_DETECTIONS_PER_FRAME; // * sizeof(int);
    CUDA_TRY(detectionMatches_.resize(matchesSize, stream));
//    CUDA_TRY(cudaMemsetAsync(detectionMatches_.data(), 0, detectionMatches_.byte_size(), stream));


    return CudaError();
}

CudaError ObjectTracker::ProcessBatch(int batchIndex,
                                      DetectionRaw* detections,
                                      int* countBuffer,
                                      int maxDetectionsStride,
                                      int width,
                                      int height,
                                      cudaStream_t stream)
{
    // auto* trackStatePtr = reinterpret_cast<TrackState*>(tracks_.data());
    // auto* matchesPtr = reinterpret_cast<int*>(detectionMatches_.data());
    auto* trackStatePtr = tracks_.data();
    auto* matchesPtr = detectionMatches_.data();

    CUDA_TRY(TrackBatch(batchIndex,
                        detections,
                        countBuffer,
                        trackStatePtr,
                        trackCount_.data(),
                        nextTrackId_.data(),
                        matchesPtr,
                        maxDetectionsStride,
                        maxTracks_,
                        numClasses_,
                        TRACKER_ALPHA,
                        width,
                        height,
                        stream
                       ));
    return CudaError(); //CompactTracks(tracks_.data(), trackCount_.data(), stream);
}

CudaError ObjectTracker::Compact(cudaStream_t stream) {
    return CompactTracks(tracks_.data(), trackCount_.data(), stream);
}

CudaError ObjectTracker::Annotate(float* imageBatch,
                                  int batchSize,
                                  int width,
                                  int height,
                                  const DetectionRaw* detections,
                                  const int* counts,
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
