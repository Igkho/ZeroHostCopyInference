#include "ObjectTracker.h"
#include <iostream>

namespace cropandweed {

CudaError ObjectTracker::Init(int maxTracks) {
    maxTracks_ = maxTracks;
    CUDA_TRY(tracks_.resize(maxTracks * sizeof(TrackState)));
    CUDA_TRY(trackCount_.resize(1));
    CUDA_TRY(nextTrackId_.resize(1));
    CUDA_TRY(detectionMatches_.resize(10000)); 

    CUDA_TRY(cudaMemset(trackCount_.data(), 0, sizeof(int)));
    int startId = 1;
    CUDA_TRY(cudaMemcpy(nextTrackId_.data(), &startId, sizeof(int), cudaMemcpyHostToDevice));

    return CudaError();
}

CudaError ObjectTracker::ProcessBatch(int batchIndex, 
                                      DetectionRaw* detections, 
                                      int* countBuffer, 
                                      int maxDetections,
                                      cudaStream_t stream) 
{
    auto* trackStatePtr = reinterpret_cast<TrackState*>(tracks_.data());

    return TrackBatch(
        batchIndex,
        detections,
        countBuffer,
        trackStatePtr,
        trackCount_.data(),
        nextTrackId_.data(),
        detectionMatches_.data(),
        maxDetections,
        maxTracks_,
        stream
    );
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
