#include "DetectorKernels.h"
#include "helpers.h"
#include "DetectionRaw.h"

#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>

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
    if (idx < totalElements) {

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

__global__ void unpack_kernel(const DetectionRaw* __restrict__ sortedCandidates,
                              const bool* __restrict__ nmsMask,
                              int numCandidates,
                              DetectionRaw* __restrict__ stridedOutput,
                              int* __restrict__ batchCounts,
                              int maxDetsPerFrame)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numCandidates) return;

    // Only process if NMS kept this box
    if (nmsMask[idx]) {
        DetectionRaw d = sortedCandidates[idx];
        int b = (int)d.batch_index;

        // Get writing slot for this specific batch
        int slot = atomicAdd(&batchCounts[b], 1);

        if (slot < maxDetsPerFrame) {
            // Write to fixed-stride location: [Batch * Stride + Slot]
            int outIdx = b * maxDetsPerFrame + slot;
            stridedOutput[outIdx] = d;
        }
    }
}

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
    if (idx < count) {

        bool keep = true;
        DetectionRaw myBox = boxes[idx];

        // Greedy NMS against sorted previous boxes (within same batch)
        // Since boxes are sorted by Batch then Score, we only look backwards
        for (int prev = idx - 1; prev >= 0; --prev) {
            DetectionRaw other = boxes[prev];

            // Optimization: Stop if we hit a different batch
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

}

CudaError DecodeAndFilter(const Block<float>& d_output,
                          TypedBlock<DetectionRaw>& candidateBuffer,
                          BoundaryBlock<int> &countBuffer,
                          int batchSize,
                          int numAnchors,
                          int numClasses,
                          float confThreshold,
                          cudaStream_t stream) {
    if (d_output.empty() || candidateBuffer.empty() || countBuffer.empty() ||
        batchSize <= 0 || numAnchors <= 0 || numClasses <= 0 ||
        (confThreshold < 0) || (confThreshold > 1)) {
        return CudaError(ERROR_SOURCE, "DecodeAndFilter invalid input");
    }

    CUDA_TRY(countBuffer.fill(0, stream));

    int maxOut = candidateBuffer.size();
    auto* rawPtr = candidateBuffer.data();
    int totalThreads = numAnchors * batchSize;
    KernelGrid grid(totalThreads);

    decode_kernel<<<grid.gsize(), grid.bsize(), 0, stream>>>(
        d_output.data(),
        rawPtr,
        countBuffer.data(),
        maxOut,
        numAnchors,
        numClasses,
        batchSize,
        confThreshold
        );
    CUDA_CHECK_KERNEL(stream);
    return CudaError();
}

CudaError RunNMS(TypedBlock<DetectionRaw>& candidateBuffer,
                 BoundaryBlock<int> &candidateCountBuffer,
                 BoundaryTypedBlock<DetectionRaw> &finalOutputBuffer,
                 BoundaryBlock<int> &finalOutputCounts,
                 Block<uint8_t>& maskBuffer,
                 float nmsThreshold,
                 int maxOutputPerBatch,
                 int batchSize,
                 cudaStream_t stream) {
    if (candidateBuffer.empty() || candidateCountBuffer.empty() ||
        finalOutputBuffer.empty() || finalOutputCounts.empty() ||
        (nmsThreshold < 0) || (nmsThreshold > 1)) {
            return CudaError(ERROR_SOURCE, "RunNMS invalid input");
    }

    // 1. Get Count (Async copy)
    std::vector<int> count_v;
    CUDA_TRY(candidateCountBuffer.to_vector(count_v, stream));

    int maxCandidates = candidateBuffer.size();
    int count = std::min(count_v[0], maxCandidates);

    // Reset the final counts for all batches to 0
    CUDA_TRY(finalOutputCounts.fill(0, stream));
    if (count == 0) {
        return CudaError();
    }

    auto* rawPtr = candidateBuffer.data();
    auto* outPtr = finalOutputBuffer.data();

    // 2. Sort Candidates (Async on Stream)
    thrust::device_ptr<DetectionRaw> ptr(rawPtr);
    thrust::sort(thrust::cuda::par.on(stream), ptr, ptr + count, DetectComparator());
    CUDA_CHECK_KERNEL(stream);

    // 3. Run NMS Kernel
    // Temporary mask buffer (should ideally be passed in or cached)
    maskBuffer.resize(count, stream);
    auto* maskPtr = reinterpret_cast<bool*>(maskBuffer.data());

    KernelGrid grid(count);
    nms_kernel<<<grid.gsize(), grid.bsize(), 0, stream>>>(rawPtr, count, nmsThreshold, maskPtr);
    CUDA_CHECK_KERNEL(stream);

    // 4. Unpack / Scatter
    // Reads masked candidates and scatters them into [BatchID * Stride + Slot]
    unpack_kernel<<<grid.gsize(), grid.bsize(), 0, stream>>>(
        rawPtr,
        maskPtr,
        count,
        outPtr,
        finalOutputCounts.data(),
        maxOutputPerBatch
    );
    CUDA_CHECK_KERNEL(stream);

    return CudaError();
}

} // namespace cropandweed
