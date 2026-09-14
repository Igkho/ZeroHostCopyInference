#include "ObjectTrackerKernels.h"
#include <cstdio>
#include "BatchDetections.h"
//#include <thrust/remove.h>
//#include <thrust/execution_policy.h>
//#include <thrust/device_ptr.h>
//#include <cub/device/device_select.cuh>

namespace cropandweed {

namespace {

// =================================================================================
//                               CONSTANTS
// =================================================================================

// 7-Segment style digit bitmap (3x5 pixels)
__device__ const uint16_t DIGIT_MAP[10] = {
    0x7B6F, 0x2492, 0x73E7, 0x73CF, 0x5BC9,
    0x79CF, 0x79EF, 0x7249, 0x7BEF, 0x7BC9
};

// Procedural Color Generation (Golden Angle)
// Generates distinct colors for any Class ID without fixed arrays
__device__ void GetProceduralColor(int cls, float& r, float& g, float& b) {
    unsigned int hash = cls * 1664525u + 1013904223u;
    r = (float)((hash >> 0) & 0xFF) / 255.0f;
    g = (float)((hash >> 8) & 0xFF) / 255.0f;
    b = (float)((hash >> 16) & 0xFF) / 255.0f;
    // Boost saturation
    float maxVal = fmaxf(r, fmaxf(g, b));
    if (maxVal > 0.1f) { r /= maxVal; g /= maxVal; b /= maxVal; }
}

// =================================================================================
//                            TRACKING KERNELS
// =================================================================================

__device__ float CalculateIOU(const DetectionRaw& d, const TrackState& t) {
    float x1 = max(d.x - d.w/2.0f, t.x - t.w/2.0f);
    float y1 = max(d.y - d.h/2.0f, t.y - t.h/2.0f);
    float x2 = min(d.x + d.w/2.0f, t.x + t.w/2.0f);
    float y2 = min(d.y + d.h/2.0f, t.y + t.h/2.0f);

    float w = max(0.0f, x2 - x1);
    float h = max(0.0f, y2 - y1);
    float interArea = w * h;
    float unionArea = (d.w * d.h) + (t.w * t.h) - interArea;

    return (unionArea > 1e-6) ? interArea / unionArea : 0.0f;
}

__global__ void PredictTracksKernel(TrackState* __restrict__ tracks,
                                    int* __restrict__ numTracks) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= *numTracks) return;

    TrackState& t = tracks[idx];
    t.x += t.vx;
    t.y += t.vy;
    t.timeSinceUpdate++;
    t.missedFrames++;
}

// OPTIMIZED: Assumes 'detections' pointer is already offset to the current batch slice
__global__ void MatchDetectionsKernel(DetectionRaw* __restrict__ detections,
                                      int numDetections,
                                      TrackState* __restrict__ tracks,
                                      int numTracks,
                                      int* __restrict__ matches)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numDetections) return;

    // Direct access, no batch loop needed
    DetectionRaw myDet = detections[idx];
    matches[idx] = -1;

    float bestIOU = 0.3f;
    int bestTrackIdx = -1;

    for (int t = 0; t < numTracks; ++t) {
        TrackState& track = tracks[t];

        // Skip dead tracks
        if (track.age == -999) {
            continue;
        }

        float iou = CalculateIOU(myDet, track);

        float requiredIOU = (track.missedFrames > 0) ? 0.45f : 0.30f;

        if (iou > requiredIOU) {
            if (iou > bestIOU) {
                bestIOU = iou;
                bestTrackIdx = t;
            }
        }
    }

    if (bestTrackIdx != -1) {
        matches[idx] = bestTrackIdx;
    }
}

__global__ void UpdateTracksKernel(DetectionRaw* __restrict__ detections,
                                   int numDetections,
                                   int* __restrict__ matches,
                                   TrackState* tracks,
                                   int* __restrict__ numTracks,
                                   int maxTracks,
                                   int* __restrict__ nextTrackId,
                                   int activeClasses,
                                   float alpha,
                                   int width,
                                   int height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numDetections) return;

    int trackIdx = matches[idx];
    DetectionRaw& det = detections[idx];

    if (trackIdx != -1) {
        TrackState& t = tracks[trackIdx];

        float widthF = (float)width;
        float heightF = (float)height;
        float margin = 2.0f; // Pixel margin to consider "touching"

        // --- X Axis Analysis ---
        float raw_vx;

        bool touchingLeft = (det.x - det.w * 0.5f) <= margin;
        bool touchingRight = (det.x + det.w * 0.5f) >= (widthF - margin);

        if (touchingLeft && !touchingRight) {
            // Exiting/Entering Left: Left Edge is clipped/unreliable. Center is skewed.
            // TRUST RIGHT EDGE
            float det_x_r = det.x + det.w * 0.5f;
            float t_x_r   = t.x + t.w * 0.5f;
            raw_vx = det_x_r - t_x_r + t.vx;
        }
        else if (touchingRight && !touchingLeft) {
            // Exiting/Entering Right: Right Edge is clipped.
            // TRUST LEFT EDGE
            float det_x_l = det.x - det.w * 0.5f;
            float t_x_l   = t.x - t.w * 0.5f;
            raw_vx = det_x_l - t_x_l + t.vx;
        }
        else {
            // Center of screen (or spanning both borders):
            // Center mass is most stable for plants.
            raw_vx = det.x - t.x + t.vx;
        }

        // --- Y Axis Analysis ---
        float raw_vy;

        bool touchingTop = (det.y - det.h * 0.5f) <= margin;
        bool touchingBottom = (det.y + det.h * 0.5f) >= (heightF - margin);

        if (touchingTop && !touchingBottom) {
            // TRUST BOTTOM EDGE
            float det_y_b = det.y + det.h * 0.5f;
            float t_y_b   = t.y + t.h * 0.5f;
            raw_vy = det_y_b - t_y_b + t.vy;
        }
        else if (touchingBottom && !touchingTop) {
            // TRUST TOP EDGE
            float det_y_t = det.y - det.h * 0.5f;
            float t_y_t   = t.y - t.h * 0.5f;
            raw_vy = det_y_t - t_y_t + t.vy;
        }
        else {
            // TRUST CENTER
            raw_vy = det.y - t.y + t.vy;
        }

        // Apply Filter
        t.vx += (t.vx == 0 ? raw_vx : TRACK_VELOCITY_FILTER_RATIO * (raw_vx - t.vx));
        t.vy += (t.vy == 0 ? raw_vy : TRACK_VELOCITY_FILTER_RATIO * (raw_vy - t.vy));

        // --- End Adaptive Velocity Logic ---

        // Update Position State
        t.x = det.x;
        t.y = det.y;
        t.w = det.w;
        t.h = det.h;

        t.timeSinceUpdate = 0;
        t.missedFrames = 0;
        t.age++;

        // Bayesian Update using actual model class count
        int cls = (int)det.class_id;
        if (cls >= 0 && cls < activeClasses && cls < TRACKER_MAX_CLASSES) {
            for (int i = 0; i < activeClasses; ++i) {
                float measure = (i == cls) ? det.score : 0.0f;
                t.classProbs[i] = t.classProbs[i] + alpha * (measure - t.classProbs[i]);
            }
        }

        int bestC = 0;
        float maxP = 0.0f;
        for (int i = 0; i < activeClasses; ++i) {
            if (t.classProbs[i] > maxP) {
                maxP = t.classProbs[i];
                bestC = i;
            }
        }

        det.track_id = (float)t.id;
        det.class_id = (float)bestC;
        det.score = maxP;
//        det.score = abs(t.vx); //sqrtf(t.vx * t.vx + t.vy * t.vy) * 10;

    } else {
        // New Track Logic
        int newIdx = atomicAdd(numTracks, 1);
        if (newIdx < maxTracks) {
            TrackState& t = tracks[newIdx];
            t.id = atomicAdd(nextTrackId, 1);
            t.x = det.x; t.y = det.y;
            t.w = det.w; t.h = det.h;
            t.vx = 0; t.vy = 0;
            t.age = 1;
            t.timeSinceUpdate = 0;
            t.missedFrames = 0;

            // Initialize Probs
            for (int i = 0; i < activeClasses; ++i) {
                t.classProbs[i] = 0.0f;
            }

            int cls = (int)det.class_id;
            if (cls >=0 && cls < activeClasses && cls < TRACKER_MAX_CLASSES) {
                t.classProbs[cls] = det.score;
            }
            det.track_id = (float)t.id;
        }
    }
}

__inline__ __device__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int s = 16; s >= 1; s >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, s);
    }
    return val;
}

constexpr int MAX_TRACK_LOOP_REPEAT_COUNT = 1000;

__global__ void GhostAndCleanupKernel(TrackState* __restrict__ tracks,
                                      int* __restrict__ numTracks,
                                      DetectionRaw* __restrict__ detections,
                                      int* __restrict__ detCount,
                                      int maxDetections,
                                      float batchIndex,
                                      int activeClasses,
                                      int width,
                                      int height)
{
    // Layout: [0]=SumVx, [1]=SumVy, [2]=Count
    extern __shared__ float s_reduce[];

    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid & 0x1F;
    int wid = tid >> 5;
    int totalTracks = *numTracks;
    float svx = 0, svy = 0, cnt = 0;

    // The grid sride loop
    int sidx = idx;
    int failsafe_1 = 0; // Universal Failsafe
    while (sidx < totalTracks && failsafe_1 < MAX_TRACK_LOOP_REPEAT_COUNT) {
        failsafe_1++;
        TrackState& t = tracks[sidx];
        if (t.age != -999 && t.age >= 5) {
            svx += t.vx;
            svy += t.vy;
            cnt ++;
        }
        sidx += gridDim.x * blockDim.x;
    }

    // Reduce over warps and store the per-warp result to the shared memory
    svx = warp_reduce_sum(svx);
    svy = warp_reduce_sum(svy);
    cnt = warp_reduce_sum(cnt);
    if (lane == 0) {
        s_reduce[wid * 3] = svx;
        s_reduce[wid * 3 + 1] = svy;
        s_reduce[wid * 3 + 2] = cnt;
    }
    __syncthreads();

    // Reduce the shared mem results on the warp 0, 1 and 2
    // Then put the result in the beginning of shared mem
    if (wid == 0) {
        int num_warps = blockDim.x >> 5;
        float vx = (lane < num_warps ? s_reduce[lane * 3] : 0);
        vx = warp_reduce_sum(vx);
        float vy = (lane < num_warps ? s_reduce[lane * 3 + 1] : 0);
        vy = warp_reduce_sum(vy);
        float vz = (lane < num_warps ? s_reduce[lane * 3 + 2] : 0);
        vz = warp_reduce_sum(vz);
        if (lane == 0) {
            s_reduce[0] = vx;
            s_reduce[1] = vy;
            s_reduce[2] = vz;
        }
    }
    __syncthreads();

    // Spread the result over all threads of the block
    bool s_valid = s_reduce[2] > 2.f; // At least 3 tracks are required for the meaningful average
    if (s_valid) {
        svx = s_reduce[0] / s_reduce[2];
        svy = s_reduce[1] / s_reduce[2];
    }

    // Filter Outliers (All threads)
    sidx = idx;
    int failsafe_2 = 0; // Universal Failsafe
    while (sidx < totalTracks && failsafe_2 < MAX_TRACK_LOOP_REPEAT_COUNT) {
        failsafe_2++;
        TrackState& t = tracks[sidx];

        // Check if we should filter this track
        // Note: s_valid check prevents filtering if the scene is empty
        if (s_valid && t.age != -999 && t.age >= 5) {
            float devX = fabsf(t.vx - svx);
            float devY = fabsf(t.vy - svy);

            if (devX > MEAN_VELOCITY_DEVIATION_MARGIN ||
                devY > MEAN_VELOCITY_DEVIATION_MARGIN) {
                t.age = -999; // Mark as dead
            }
        }
        // Marking old for compaction
        if (t.age != -999 && t.missedFrames > TRACKER_MAX_MISSED_FRAMES) {
            t.age = -999;
        }

        // If the center of the track has left the frame, kill it immediately.
        if (t.age != -999 && (t.x < 0 || t.y < 0 || t.x >= width || t.y >= height)) {
            t.age = -999;
        }

        if (t.timeSinceUpdate > 0 &&
            t.timeSinceUpdate < GHOST_MAX_STALE_FRAMES &&
            t.age > GHOST_MIN_AGE_THRESHOLD) {
            int slot = atomicAdd(detCount, 1);
            if (slot < maxDetections) {
                DetectionRaw& g = detections[slot];
                g.x = t.x; g.y = t.y; g.w = t.w; g.h = t.h;
                g.batch_index = batchIndex;
                g.track_id = (float)t.id;

                int bestC = 0; float maxP = 0.0f;
                for (int i = 0; i < activeClasses; ++i) {
                    if (t.classProbs[i] > maxP) {
                        maxP = t.classProbs[i];
                        bestC = i;
                    }
                }
                g.class_id = (float)bestC;
                g.score = maxP * 0.5f;
            }
        }
        sidx += gridDim.x * blockDim.x;
    }
}

// =================================================================================
//                            ANNOTATOR KERNELS
// =================================================================================

__device__ void DrawFilledRect(float* __restrict__ img, int w, int h, int x, int y, int rw, int rh, float r, float g, float b) {
    int planeSize = w * h;
    for (int row = 0; row < rh; ++row) {
        for (int col = 0; col < rw; ++col) {
            int px = x + col;
            int py = y + row;
            // Clipping check
            if (px >= 0 && px < w && py >= 0 && py < h) {
                int idx = py * w + px;
                img[idx] = r;
                img[idx + planeSize] = g;
                img[idx + 2 * planeSize] = b;
            }
        }
    }
}

__device__ void DrawDigit(float* __restrict__ img, int w, int h, int x0, int y0, int digit, float r, float g, float b) {
    if ((digit < 0) || (digit > 9)) return;
    uint16_t map = DIGIT_MAP[digit];
    for (int row = 0; row < 5; ++row) {
        for (int col = 0; col < 3; ++col) {
            int bitIndex = (4 - row) * 3 + (2 - col);
            if (map & (1 << bitIndex)) {
                int px = x0 + col;
                int py = y0 + row;
                if (px >= 0 && px < w && py >= 0 && py < h) {
                    int idx = py * w + px;
                    int planeSize = w * h;
                    img[idx] = r; img[idx + planeSize] = g; img[idx + 2 * planeSize] = b;
                }
            }
        }
    }
}

__device__ void DrawNumber(float* __restrict__ img, int w, int h, int x, int y, int number, float r, float g, float b) {
    int temp = number;
    int numDigits = (number == 0) ? 1 : 0;
    while (temp > 0) { temp /= 10; numDigits++; }

    temp = number;
    for (int i = 0; i < numDigits; ++i) {
        int d = temp % 10;
        temp /= 10;
        DrawDigit(img, w, h, x + (numDigits - 1 - i) * 4, y, d, r, g, b);
    }
}

__global__ void DrawBoxesKernel(float* __restrict__ imageBatch,
                                int width, int height,
                                const DetectionRaw* __restrict__ detections,
                                const int* __restrict__ counts,
                                int batchSize,
                                int stride) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int batchId = idx / stride;
    int boxId = idx % stride;

    if (batchId >= batchSize) return;
    int count = counts[batchId];
    if (boxId >= count) return;

    DetectionRaw det = detections[batchId * stride + boxId];

    // if (isnan(det.x) || isnan(det.y) || isnan(det.w) || isnan(det.h) ||
    //     det.w <= 16.0f || det.h <= 16.0f || det.w > width || det.h > height) {
    //     return;
    // }

    int planeSize = width * height;
    float* img = imageBatch + (batchId * planeSize * 3);

    int x1 = (int)(det.x - det.w / 2.0f);
    int y1 = (int)(det.y - det.h / 2.0f);
    int x2 = (int)(det.x + det.w / 2.0f);
    int y2 = (int)(det.y + det.h / 2.0f);

    x1 = max(0, min(x1, width - 1));
    x2 = max(0, min(x2, width - 1));
    y1 = max(0, min(y1, height - 1));
    y2 = max(0, min(y2, height - 1));

    // Skip invalid boxes
   if (x1 >= x2 || y1 >= y2) return;

    int cls = (int)det.class_id;
    float r, g, b;
    if (cls < 0) {
        // Ghost Track Color: Light Grey
        r = 0.6f;
        g = 0.6f;
        b = 0.6f;
    } else {
        GetProceduralColor(cls, r, g, b);
    }

    int thickness = 4;

    if (cls < 0) thickness = 2;

    int t_w = min(thickness, (x2 - x1) / 2); // Clamp X thickness
    int t_h = min(thickness, (y2 - y1) / 2); // Clamp Y thickness

    // Draw Box
    for (int t = 0; t < t_h; ++t) {
        for (int x = x1; x <= x2; ++x) {
            int py_top = min(y1 + t, height - 1);
            int py_bot = max(y2 - t, 0);
            int idxT = py_top * width + x;
            int idxB = py_bot * width + x;
            img[idxT] = r; img[idxT + planeSize] = g; img[idxT + 2*planeSize] = b;
            img[idxB] = r; img[idxB + planeSize] = g; img[idxB + 2*planeSize] = b;
        }
    }
    for (int t = 0; t < t_w; ++t) {
        for (int y = y1; y <= y2; ++y) {
            int px_left = min(x1 + t, width - 1);
            int px_right = max(x2 - t, 0);
            int idxL = y * width + px_left;
            int idxR = y * width + px_right;
            img[idxL] = r; img[idxL + planeSize] = g; img[idxL + 2*planeSize] = b;
            img[idxR] = r; img[idxR + planeSize] = g; img[idxR + 2*planeSize] = b;
        }
    }

    // Draw ID
    int trkId = (int)det.track_id;
    if (trkId > 0) {

        int padding = 3;
        int idX = x1 + padding;
        int idY = (y1 < 20) ? y2 - 10 : y1 - 8;
        // Calculate text dimensions
        int temp = trkId;
        int numDigits = (trkId == 0) ? 1 : 0;
        while (temp > 0) { temp /= 10; numDigits++; }

        // Total text width: (digits * 3) + (digits-1 * 1) = digits * 4 - 1
        // We approximate stride as 4 per digit.
        int textW = numDigits * 4;
        int textH = 5;

        // Background Box Dimensions
        int bgW = textW + (padding * 2);
        int bgH = textH + (padding * 2);
        int bgX = idX - padding;
        int bgY = idY - padding;

        // Draw Background (Bounding Box Color)
        DrawFilledRect(img, width, height, bgX, bgY, bgW, bgH, r, g, b);
        // Draw Text (White)
       DrawNumber(img, width, height, idX, idY, trkId, 1.0f, 1.0f, 1.0f);
        // DrawNumber(img, width, height, idX, idY, velocityVal, 1.0f, 1.0f, 1.0f);
    }
}

// struct IsDeadTrack {
//     __host__ __device__
//         bool operator()(const TrackState& t) { return t.age == -999; }
// };

} // anonymous namespace


// =================================================================================
//                            HOST WRAPPERS
// =================================================================================

CudaError TrackBatch(int batchIndex,
                     BoundaryTypedBlock<DetectionRaw> &detections,
                     BoundaryBlock<int> &countBuffer,
                     std::vector<int> &countBufferHost,
                     TypedBlock<TrackState> &tracks,
                     Block<int> &trackCount,
                     std::vector<int> &trackCountHost,
                     Block<int> &nextTrackId,
                     Block<int> &detectionMatches,
                     int stride,
                     int maxTracks,
                     int activeClasses,
                     float alpha,
                     int imageWidth,
                     int imageHeight,
                     cudaStream_t stream)
{
    if (batchIndex < 0 || detections.empty() || countBuffer.empty() || tracks.empty() ||
        trackCount.empty() || nextTrackId.empty() || detectionMatches.empty() ||
        stride <= 0 || maxTracks < 0 || activeClasses < 0 || imageWidth <= 0 || imageHeight <= 0) {
        return CudaError(ERROR_SOURCE, "Invalid input parameters in TrackBatch");
    }

    // Predict
    KernelGrid gridPredict(maxTracks);
    PredictTracksKernel<<<gridPredict.gsize(), gridPredict.bsize(), 0, stream>>>(tracks.data(),
                                                                                 trackCount.data());
    CUDA_CHECK_KERNEL(stream);

    // Get Count for THIS batch (Async)
    // std::vector<int> currentDetCounts;
    // std::vector<int> currentTrackCounts;

    // CUDA_TRY(countBuffer.to_vector(currentDetCounts, stream));
    CUDA_TRY(countBuffer.to_vector(countBufferHost, stream));

    // CUDA_TRY(trackCount.to_vector(currentTrackCounts, stream));
    CUDA_TRY(trackCount.to_vector(trackCountHost, stream));

    // Clamp count to stride (capacity)
    int currentDetCount = std::min(countBufferHost[batchIndex], stride);
    int currentTrackCount = std::min(trackCountHost[0], maxTracks);

    // Calculate slice pointer
    DetectionRaw* batchDets = detections.data() + (batchIndex * stride);

    if (currentDetCount > 0) {
        // Match (Pass the slice, not the whole buffer)
        KernelGrid gridMatch(currentDetCount);
        MatchDetectionsKernel<<<gridMatch.gsize(), gridMatch.bsize(), 0, stream>>>(
            batchDets, currentDetCount, tracks.data(), currentTrackCount, detectionMatches.data()
        );
        CUDA_CHECK_KERNEL(stream);
        // Update
        UpdateTracksKernel<<<gridMatch.gsize(), gridMatch.bsize(), 0, stream>>>(
            batchDets, currentDetCount, detectionMatches.data(), tracks.data(), trackCount.data(),
            maxTracks, nextTrackId.data(), activeClasses, alpha, imageWidth, imageHeight
        );
        CUDA_CHECK_KERNEL(stream);
    }

    // Ghosts
    if (currentTrackCount > 0) {
        int validCount = std::min(currentTrackCount, TRACKER_MAX_TRACKS);
        // Round up to the nearest warp (32 threads)
        int threadsPerBlock = ((validCount + 31) / 32) * 32;
        // Clamp between 32 (1 warp) and 1024 (max block size)
        threadsPerBlock = std::max(32, std::min(1024, threadsPerBlock));
        KernelGrid gridGhost(validCount, threadsPerBlock);
        // Define how the shared memory scales with the block size.
        // We need 3 floats per warp (threads >> 5).
        auto shmem_calc_gac = [](int threads) {
            return (threads >> 5) * sizeof(float) * 3;
        };
        GhostAndCleanupKernel<<<gridGhost.gsize(), gridGhost.bsize(),
                                shmem_calc_gac(gridGhost.bsize().x), stream>>>(
            tracks.data(), trackCount.data(), batchDets, countBuffer.data() + batchIndex, stride,
            (float)batchIndex, activeClasses, imageWidth, imageHeight
        );
        CUDA_CHECK_KERNEL(stream);
    }

    return CudaError();
}

// struct IsAliveTrack {
//     __host__ __device__
//         bool operator()(const TrackState& t) const { return t.age != -999; }
// };

__global__ void CompactTracksKernel(const TrackState* __restrict__ src,
                                    TrackState* __restrict__ dst,
                                    int* __restrict__ newCount,
                                    int currentCount) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= currentCount) return;

    // Read the track into registers
    TrackState t = src[idx];

    // If it's alive, claim a slot in the destination buffer and write it
    if (t.age != -999) {
        int slot = atomicAdd(newCount, 1);
        dst[slot] = t;
    }
}

// Tracks compaction implementation
CudaError CompactTracks(TypedBlock<TrackState> &tracksBuffer,
                        TypedBlock<TrackState>& tempTracksBuffer,
                        Block<int> &countBuffer,
                        int maxTracks,
                        cudaStream_t stream) {
    if (tracksBuffer.empty() || countBuffer.empty() || maxTracks <= 0) {
        return CudaError(ERROR_SOURCE, "Invalid input parameters in CompactTracks");
    }
    std::vector<int> currentCounts;

    CUDA_TRY(countBuffer.to_vector(currentCounts, stream));
    int currentCount = std::min(currentCounts[0], maxTracks);

    if (currentCount == 0) {
        return CudaError();
    }

    // Reset the count buffer to 0 so the kernel can use it as an atomic counter
    CUDA_TRY(countBuffer.fill(0, stream));

    // Launch our custom, zero-workspace compaction kernel
    KernelGrid grid(currentCount);
    CompactTracksKernel<<<grid.gsize(), grid.bsize(), 0, stream>>>(
        tracksBuffer.data(), tempTracksBuffer.data(), countBuffer.data(), currentCount
        );
    CUDA_CHECK_KERNEL(stream);

    // O(1) Pointer Swap! (Zero D2D memory copy)
    tracksBuffer.raw().swap(tempTracksBuffer.raw());

    return CudaError();
}

CudaError DrawDetections(Block<float> &imageBatch,
                         int batchSize,
                         int width,
                         int height,
                         const BoundaryTypedBlock<DetectionRaw> &detections,
                         const BoundaryBlock<int> &counts,
                         cudaStream_t stream) {
    if (imageBatch.empty() || batchSize <= 0 || width <= 0 || height <= 0 ||
         detections.empty() || counts.empty()) {
        return CudaError(ERROR_SOURCE, "Invalid input parameters in DrawDetections");
    }

    int stride = BatchDetections::MAX_DETECTIONS_PER_FRAME;
    int totalSlots = batchSize * stride;
    KernelGrid grid(totalSlots);

    DrawBoxesKernel<<<grid.gsize(), grid.bsize(), 0, stream>>>(
        imageBatch.data(), width, height, detections.data(), counts.data(), batchSize, stride
        );
    CUDA_CHECK_KERNEL(stream);

    return CudaError();
}

} // namespace cropandweed
