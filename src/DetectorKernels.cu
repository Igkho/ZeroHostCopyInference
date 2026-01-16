#include "DetectorKernels.h"
#include "helpers.h"
#include "DetectionRaw.h"

#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/remove.h>

namespace cropandweed {

namespace {

__global__ void decode_kernel(const float* __restrict__ outputTensor,
                                   DetectionRaw* __restrict__ outputBuffer,
                                   int* __restrict__ countBuffer,
                                   int maxOut,
                                   int numAnchors,
                                   int numClasses,
                                   int batchSize,
                                   float confThreshold) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = numAnchors * batchSize;

    if (idx >= totalElements) return;

    int batchId = idx / numAnchors;
    int anchorId = idx % numAnchors;

    int rowSize = 4 + numClasses;
    int itemsPerBatch = rowSize * numAnchors;

    // Calculate pointer to this batch's data
    const float* data = outputTensor + (batchId * itemsPerBatch);

    float maxScore = 0.0f;
    int maxClassId = -1;

    for (int c = 0; c < numClasses; ++c) {
        float score = data[(4 + c) * numAnchors + anchorId];
        if (score > maxScore) {
            maxScore = score;
            maxClassId = c;
        }
    }

    if (maxScore > confThreshold) {
        int slot = atomicAdd(countBuffer, 1);
        if (slot >= maxOut) return;

        // Normalized 0..1 coordinates
        float cx = data[0 * numAnchors + anchorId];
        float cy = data[1 * numAnchors + anchorId];
        float w  = data[2 * numAnchors + anchorId];
        float h  = data[3 * numAnchors + anchorId];

        outputBuffer[slot].x = cx;
        outputBuffer[slot].y = cy;
        outputBuffer[slot].w = w;
        outputBuffer[slot].h = h;
        outputBuffer[slot].score = maxScore;
        outputBuffer[slot].class_id = (float)maxClassId;
        outputBuffer[slot].batch_index = (float)batchId;
    }
}

}

CudaError DecodeAndFilter(const float* d_output,
                          uint8_t * candidateBuffer,
                          int candidateBufferSize,
                          int *countBuffer,
                          int batchSize,
                          int numAnchors,
                          int numClasses,
                          float confThreshold,
                          cudaStream_t stream) {
    if (d_output == nullptr || candidateBuffer == nullptr || candidateBufferSize <= 0 ||
        countBuffer == nullptr || batchSize <= 0 ||
        numAnchors <= 0 || numClasses <= 0 ||
        (confThreshold < 0) || (confThreshold > 1)) {
            return CudaError(ERROR_SOURCE, "DecodeAndFilter invalid input");
    }

    CUDA_TRY(cudaMemsetAsync(countBuffer, 0, sizeof(int), stream));

    int maxOut = candidateBufferSize / sizeof(DetectionRaw);
    auto* rawPtr = reinterpret_cast<DetectionRaw*>(candidateBuffer);

    int totalThreads = numAnchors * batchSize;
    KernelGrid grid(totalThreads);

    decode_kernel<<<grid.gsize(), grid.bsize(), 0, stream>>>(
        d_output,
        rawPtr,
        countBuffer,
        maxOut,
        numAnchors,
        numClasses,
        batchSize,
        confThreshold
        );
    return CudaError(ERROR_SOURCE, cudaGetLastError());
}

namespace {

struct DetectComparator {
    __host__ __device__
        bool operator()(const DetectionRaw& a, const DetectionRaw& b) const {
        if (a.batch_index != b.batch_index)
            return a.batch_index < b.batch_index;
        return a.score > b.score;
    }
};

__global__ void nms_kernel(DetectionRaw* __restrict__ boxes,
                           int count,
                           float threshold,
                           bool* __restrict__ kept) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    bool keep = true;
    DetectionRaw myBox = boxes[idx];

    // Greedy NMS against sorted previous boxes
    for (int prev = idx - 1; prev >= 0; --prev) {
        DetectionRaw other = boxes[prev];
        if (other.batch_index != myBox.batch_index) break;

        float x1 = max(myBox.x - myBox.w/2, other.x - other.w/2);
        float y1 = max(myBox.y - myBox.h/2, other.y - other.h/2);
        float x2 = min(myBox.x + myBox.w/2, other.x + other.w/2);
        float y2 = min(myBox.y + myBox.h/2, other.y + other.h/2);

        float interW = max(0.0f, x2 - x1);
        float interH = max(0.0f, y2 - y1);
        float interArea = interW * interH;
        float unionArea = (myBox.w * myBox.h) + (other.w * other.h) - interArea;

        if ((interArea / unionArea) > threshold) {
            keep = false;
            break;
        }
    }
    kept[idx] = keep;
}

}

CudaError RunNMS_GPU(uint8_t *candidateBuffer,
                     int candidateBufferSize,
                     int *countBuffer,
                     uint8_t *finalOutputBuffer,
                     int finalOutputBufferSize,
                     int *finalOutputCount,
                     Block<uint8_t> &maskBuffer,
                     float nmsThreshold,
                     cudaStream_t stream) {
    if (candidateBuffer == nullptr || candidateBufferSize <= 0 || countBuffer == nullptr ||
        finalOutputBuffer == nullptr || finalOutputBufferSize <= 0 || finalOutputCount == nullptr ||
        (nmsThreshold < 0) || (nmsThreshold > 1)) {
            return CudaError(ERROR_SOURCE, "RunNMS_GPU invalid input");
    }

    // 1. Get Count (Async copy)
    int count = 0;
    CUDA_TRY(cudaMemcpyAsync(&count, countBuffer, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_TRY(cudaStreamSynchronize(stream));

    int maxCandidates = candidateBufferSize / sizeof(DetectionRaw);
    count = std::min(count, maxCandidates);

    if (count == 0) {
        CUDA_TRY(cudaMemsetAsync(finalOutputCount, 0, sizeof(int), stream));
        return CudaError();
    }

    auto* rawPtr = reinterpret_cast<DetectionRaw*>(candidateBuffer);
    auto* outPtr = reinterpret_cast<DetectionRaw*>(finalOutputBuffer);

    // 2. Sort Candidates (Async on Stream)
    thrust::device_ptr<DetectionRaw> ptr(rawPtr);
    thrust::sort(thrust::cuda::par.on(stream), ptr, ptr + count, DetectComparator());

    // 3. Run NMS Kernel
    // Temporary mask buffer (should ideally be passed in or cached)
    maskBuffer.resize(count);
    auto* maskPtr = reinterpret_cast<bool*>(maskBuffer.data());

    KernelGrid grid(count);
    nms_kernel<<<grid.gsize(), grid.bsize(), 0, stream>>>(rawPtr, count, nmsThreshold, maskPtr);

    // 4. Compact Results
    thrust::device_ptr<bool> maskDev(maskPtr);
    thrust::device_ptr<DetectionRaw> outDev(outPtr);

    auto endPtr = thrust::copy_if(thrust::cuda::par.on(stream),
                                  ptr, ptr + count, maskDev, outDev,
                                  thrust::identity<bool>());

    // 5. Write Final Count
    int keptCount = (int)(endPtr - outDev);
    CUDA_TRY(cudaMemcpyAsync(finalOutputCount, &keptCount, sizeof(int), cudaMemcpyHostToDevice, stream));
    return CudaError();
}

// CPU NMS IMPLEMENTATION
CudaError RunNMS_CPU(uint8_t *candidateBuffer,
                     int candidateBufferSize,
                     int *countBuffer,
                     int batchSize,
                     float nmsThreshold,
                     std::vector<std::vector<Detection>> &output) {

    // 1. Download Count
    int count = 0;
    CUDA_TRY(cudaMemcpy(&count, countBuffer, sizeof(int), cudaMemcpyDeviceToHost));

    // Safety clamp
    int maxStored = candidateBufferSize / sizeof(DetectionRaw);
    count = std::min(count, maxStored);

    if (count == 0) {
        output = std::vector<std::vector<Detection>>(batchSize);
        return CudaError();
    }

    // 2. Download Candidates
    std::vector<DetectionRaw> candidates(count);
    CUDA_TRY(cudaMemcpy(candidates.data(), candidateBuffer,
                         count * sizeof(DetectionRaw), cudaMemcpyDeviceToHost));

    // 3. Sort & NMS (CPU implementation)
    std::vector<std::vector<Detection>> results(batchSize);
    std::vector<std::vector<DetectionRaw>> batchCandidates(batchSize);

    // Group by Batch
    for (const auto& c : candidates) {
        int b = (int)c.batch_index;
        if (b >= 0 && b < batchSize) {
            batchCandidates[b].push_back(c);
        }
    }

    // Process each batch
    for (int b = 0; b < batchSize; ++b) {
        auto& dets = batchCandidates[b];

        // Sort by score descending
        std::sort(dets.begin(), dets.end(), [](const DetectionRaw& a, const DetectionRaw& b) {
            return a.score > b.score;
        });

        // Simple IOU Loop
        for (size_t i = 0; i < dets.size(); ++i) {
            if (dets[i].score == 0.0f) continue; // Suppressed

            // Add to final results
            Detection det;
            det.x = dets[i].x; det.y = dets[i].y;
            det.w = dets[i].w; det.h = dets[i].h;
            det.score = dets[i].score;
            det.classId = (int)dets[i].class_id;
            results[b].push_back(det);

            // Calculate corners for the current box (Box A)
            // (x,y) is center, w,h is full width/height
            float ax1 = dets[i].x - dets[i].w * 0.5f;
            float ay1 = dets[i].y - dets[i].h * 0.5f;
            float ax2 = dets[i].x + dets[i].w * 0.5f;
            float ay2 = dets[i].y + dets[i].h * 0.5f;
            float areaA = dets[i].w * dets[i].h;

            // Suppress neighbors
            for (size_t j = i + 1; j < dets.size(); ++j) {
                if (dets[j].score == 0.0f) continue;

                // Calculate corners for the neighbor box (Box B)
                float bx1 = dets[j].x - dets[j].w * 0.5f;
                float by1 = dets[j].y - dets[j].h * 0.5f;
                float bx2 = dets[j].x + dets[j].w * 0.5f;
                float by2 = dets[j].y + dets[j].h * 0.5f;
                float areaB = dets[j].w * dets[j].h;

                // Intersection Rectangle
                float xx1 = std::max(ax1, bx1);
                float yy1 = std::max(ay1, by1);
                float xx2 = std::min(ax2, bx2);
                float yy2 = std::min(ay2, by2);

                float w = std::max(0.0f, xx2 - xx1);
                float h = std::max(0.0f, yy2 - yy1);
                float interArea = w * h;

                // Union Area
                float unionArea = areaA + areaB - interArea;

                // IOU check
                if (unionArea > 0.0f) {
                    float iou = interArea / unionArea;
                    if (iou > nmsThreshold) {
                        dets[j].score = 0.0f; // Suppress by zeroing score
                    }
                }
            }
        }
    }
    output = std::move(results);
    return CudaError();
}

} // namespace cropandweed
