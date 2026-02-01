#pragma once
#include <cuda_runtime.h>
#include "DetectionRaw.h"
#include "helpers.h"

namespace cropandweed {

// Safe compile-time max. We will iterate only up to 'numClasses' at runtime.
constexpr int TRACKER_MAX_CLASSES = 128;

// Set to o 1024 to fit in a single CUDA block for shared memory reduction
constexpr int TRACKER_MAX_TRACKS = 1024;

// Ghost / Prediction Heuristics
// --------------------------------------------------------

// Tracks must exist this long before attempted to become "Ghost" (predict over occlusion).
// Prevents visualizing noise/flicker.
constexpr int GHOST_MIN_AGE_THRESHOLD = 10;

// Maximum frames to keep drawing a track that has no detector match.
// After this many frames of signal loss, rendering the prediction is stopped
// (though may be kept it in memory longer via MAX_MISSED_FRAMES).
constexpr int GHOST_MAX_STALE_FRAMES = 30;

// Maximum track velocity deviation from the mean velocity to exclude it from the living tracks
constexpr float MEAN_VELOCITY_DEVIATION_MARGIN = 1.f;

// --------------------------------------------------------

// Learning rate for class probabilities
constexpr float TRACKER_ALPHA = 0.1f;

// Time before death (tombstone)
constexpr int TRACKER_MAX_MISSED_FRAMES = 60;

// The margin in pixels to remove ghost track immediately
constexpr float GHOST_TRACK_EXIT_MARGIN = 30;

// The ratio for the tracks velocity filter
constexpr float TRACK_VELOCITY_FILTER_RATIO = 0.2;


struct TrackState {
    int id;
    int age;
    int timeSinceUpdate;
    int missedFrames;
    float x, y, w, h;
    float vx, vy;
    float classProbs[TRACKER_MAX_CLASSES];
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
                     int stride,
                     int maxTracks,
                     int activeClasses,
                     float alpha,
                     int imageWidth,
                     int imageHeight,
                     cudaStream_t stream);

CudaError CompactTracks(TrackState *tracksBuffer,
                        int* countBuffer,
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
