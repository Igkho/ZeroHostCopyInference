#include "ObjectTrackerKernels.h"
//#include "device_launch_parameters.h"
#include <cstdio>
//#include <cmath>

namespace cropandweed {

namespace {

// =================================================================================
//                               CONSTANTS
// =================================================================================

__constant__ float ALPHA = 0.1f; 
__constant__ int MAX_MISSED_FRAMES = 60; 

// 7-Segment style digit bitmap (3x5 pixels)
__device__ const uint16_t DIGIT_MAP[10] = {
    0x7B6F, 0x2492, 0x73E7, 0x73CF, 0x5BC9, 
    0x79CF, 0x79EF, 0x7249, 0x7BEF, 0x7BC9
};

// Colors (RGB normalized)
__device__ const float CLASS_COLORS[6][3] = {
    {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.5f, 1.0f},
    {1.0f, 1.0f, 0.0f}, {1.0f, 0.0f, 1.0f}, {0.0f, 1.0f, 1.0f}
};

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

__global__ void MatchDetectionsKernel(const DetectionRaw* __restrict__ detections,
                                      int numDetections,
                                      TrackState* __restrict__ tracks,
                                      int numTracks,
                                      int* __restrict__ matches,
                                      int targetBatchIndex) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numDetections) return;

    DetectionRaw myDet = detections[idx];
    matches[idx] = -1; 
    
    if ((int)myDet.batch_index != targetBatchIndex) return;

    float bestIOU = 0.3f; 
    int bestTrackIdx = -1;

    for (int t = 0; t < numTracks; ++t) {
        float iou = CalculateIOU(myDet, tracks[t]);
        if (iou > bestIOU) {
            bestIOU = iou;
            bestTrackIdx = t;
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
                                   int targetBatchIndex) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numDetections) return;

    if ((int)detections[idx].batch_index != targetBatchIndex) return;

    int trackIdx = matches[idx];
    DetectionRaw& det = detections[idx];

    if (trackIdx != -1) {
        TrackState& t = tracks[trackIdx];
        
        float vx = det.x - t.x; 
        float vy = det.y - t.y;
        t.vx = 0.7f * t.vx + 0.3f * vx;
        t.vy = 0.7f * t.vy + 0.3f * vy;
        
        t.x = det.x; t.y = det.y;
        t.w = det.w; t.h = det.h;
        t.timeSinceUpdate = 0;
        t.missedFrames = 0;
        t.age++;
        
        int cls = (int)det.class_id;
        if (cls >= 0 && cls < 6) {
            for(int i=0; i<6; ++i) {
                float measure = (i == cls) ? det.score : 0.0f;
                t.classProbs[i] = t.classProbs[i] + ALPHA * (measure - t.classProbs[i]);
            }
        }

        int bestC = 0; float maxP = 0.0f;
        for(int i=0; i<6; ++i) { if(t.classProbs[i] > maxP) { maxP=t.classProbs[i]; bestC=i; } }
        
        det.track_id = (float)t.id;
        det.class_id = (float)bestC;
        det.score = maxP;

    } else {
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
            
            for(int i=0; i<6; ++i) t.classProbs[i] = 0.0f;
            int cls = (int)det.class_id;
            if(cls >=0 && cls < 6) t.classProbs[cls] = det.score;

            det.track_id = (float)t.id;
        }
    }
}

__global__ void GhostAndCleanupKernel(TrackState* __restrict__ tracks,
                                      int* __restrict__ numTracks,
                                      DetectionRaw* __restrict__ detections,
                                      int* __restrict__ detCount,
                                      int maxDetections,
                                      float batchIndex) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= *numTracks) return;

    TrackState& t = tracks[idx];

    if (t.missedFrames > MAX_MISSED_FRAMES) {
        t.x = -10000; 
        return; 
    }

    if (t.timeSinceUpdate > 0 && t.timeSinceUpdate < 30 && t.age > 10) {
        float vMag = t.vx*t.vx + t.vy*t.vy;
        if (vMag > 0.5f) { 
            int slot = atomicAdd(detCount, 1);
            if (slot < maxDetections) {
                DetectionRaw& g = detections[slot];
                g.x = t.x; g.y = t.y; g.w = t.w; g.h = t.h;
                g.batch_index = batchIndex;
                g.track_id = (float)t.id;
                
                int bestC = 0; float maxP = 0.0f;
                for(int i=0; i<6; ++i) { if(t.classProbs[i]>maxP) { maxP=t.classProbs[i]; bestC=i; } }
                g.class_id = (float)bestC;
                g.score = maxP * 0.5f; 
            }
        }
    }
}

// =================================================================================
//                            ANNOTATOR KERNELS
// =================================================================================

__device__ void DrawDigit(float* __restrict__ img, int w, int h, int x0, int y0, int digit, float r, float g, float b) {
    if (digit < 0 || digit > 9) return;
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
                                int batchSize) {
    int totalBoxesPerBatch = 1000; 
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    int batchId = idx / totalBoxesPerBatch;
    int boxId = idx % totalBoxesPerBatch;

    if (batchId >= batchSize) return;
    int count = counts[batchId]; 
    if (boxId >= count) return;

    DetectionRaw det = detections[batchId * totalBoxesPerBatch + boxId];

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

    int cls = (int)det.class_id % 6;
    float r = CLASS_COLORS[cls][0];
    float g = CLASS_COLORS[cls][1];
    float b = CLASS_COLORS[cls][2];

    int thickness = 2;

    // Draw Box
    for (int t = 0; t < thickness; ++t) {
        for (int x = x1; x <= x2; ++x) {
            int py_top = min(y1 + t, height - 1);
            int py_bot = max(y2 - t, 0);
            int idxT = py_top * width + x;
            int idxB = py_bot * width + x;
            img[idxT] = r; img[idxT + planeSize] = g; img[idxT + 2*planeSize] = b;
            img[idxB] = r; img[idxB + planeSize] = g; img[idxB + 2*planeSize] = b;
        }
    }
    for (int t = 0; t < thickness; ++t) {
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
    int idX = x1;
    int idY = (y1 < 20) ? y2 - 10 : y1 - 8; 
    int trkId = (int)det.track_id; 
    if (trkId > 0) {
        DrawNumber(img, width, height, idX, idY, trkId, 1.0f, 1.0f, 1.0f);
    }
}

} // anonymous namespace


// =================================================================================
//                            HOST WRAPPERS
// =================================================================================

CudaError TrackBatch(int batchIndex,
                     DetectionRaw* detections,
                     int* countBuffer,
                     TrackState* tracks,
                     int* trackCount,
                     int* nextTrackId,
                     int* detectionMatches,
                     int maxDetections,
                     int maxTracks,
                     cudaStream_t stream)
{
    if (batchIndex < 0 || !detections || !countBuffer || !tracks || !trackCount ||
        !nextTrackId || !detectionMatches || maxDetections < 0 || maxTracks < 0) {
        return CudaError(ERROR_SOURCE, "Invalid input parameters in TrackBatch");
    }

    // 1. Predict
    KernelGrid gridPredict(maxTracks);
    PredictTracksKernel<<<gridPredict.gsize(), gridPredict.bsize(), 0, stream>>>(tracks, trackCount);

    // 2. Get Sizes (Sync required)
    int currentDetCount = 0;
    int currentTrackCount = 0;
    CUDA_TRY(cudaMemcpyAsync(&currentDetCount, countBuffer, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_TRY(cudaMemcpyAsync(&currentTrackCount, trackCount, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_TRY(cudaStreamSynchronize(stream)); 

    if (currentDetCount > 0) {
        // 3. Match
        KernelGrid gridMatch(currentDetCount);
        MatchDetectionsKernel<<<gridMatch.gsize(), gridMatch.bsize(), 0, stream>>>(
            detections, currentDetCount, tracks, currentTrackCount, detectionMatches, batchIndex
        );

        // 4. Update
        UpdateTracksKernel<<<gridMatch.gsize(), gridMatch.bsize(), 0, stream>>>(
            detections, currentDetCount, detectionMatches, tracks, trackCount, maxTracks, nextTrackId, batchIndex
        );
    }

    // 5. Ghosts
    if (currentTrackCount > 0) {
        KernelGrid gridGhost(currentTrackCount);
        GhostAndCleanupKernel<<<gridGhost.gsize(), gridGhost.bsize(), 0, stream>>>(
            tracks, trackCount, detections, countBuffer, maxDetections, (float)batchIndex
        );
    }

    return CudaError(ERROR_SOURCE, cudaGetLastError());
}

CudaError DrawDetections(float* imageBatch,
                         int batchSize,
                         int width,
                         int height,
                         const DetectionRaw* detections,
                         const int* counts,
                         cudaStream_t stream) {
    if (!imageBatch || batchSize <= 0 || width <0 || height <= 0 || !detections || !counts) {
        return CudaError(ERROR_SOURCE, "Invalid input parameters in DrawDetections");
    }
    
    int totalSlots = batchSize * 1000;
    KernelGrid grid(totalSlots);

    DrawBoxesKernel<<<grid.gsize(), grid.bsize(), 0, stream>>>(
        imageBatch, width, height, detections, counts, batchSize
    );

    return CudaError(ERROR_SOURCE, cudaGetLastError());
}

} // namespace cropandweed
