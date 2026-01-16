#pragma once
#include <cuda_runtime.h>
#include "DetectionRaw.h"
#include "helpers.h"

namespace cropandweed {

// Device-side state structure
// Defined here so both the Class (for allocation) and Kernels (for access) can see it.
struct TrackState {
    int id;
    int age;
    int timeSinceUpdate;     
    int missedFrames;        
    float x, y, w, h;
    float vx, vy;
    float classProbs[6]; 
};

// --- Kernel Wrappers ---

/**
 * @brief Executes the tracking logic (Predict, Match, Update, Ghost).
 */
CudaError TrackBatch(int batchIndex,
                     DetectionRaw* detections,
                     int* countBuffer,
                     TrackState* tracks,
                     int* trackCount,
                     int* nextTrackId,
                     int* detectionMatches,
                     int maxDetections,
                     int maxTracks,
                     cudaStream_t stream);

/**
 * @brief Draws bounding boxes and Track IDs onto the image buffer.
 */
CudaError DrawDetections(float* imageBatch,
                         int batchSize,
                         int width,
                         int height,
                         const DetectionRaw* detections,
                         const int* counts,
                         cudaStream_t stream);

} // namespace cropandweed
