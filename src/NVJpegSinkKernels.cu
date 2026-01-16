#include "NVJpegSinkKernels.h"
#include "helpers.h"

namespace fs = std::filesystem;

namespace cropandweed {

namespace {

// Simple cast kernel. Works for Planar OR Interleaved (treats data as 1D array)
__global__ void FloatToUint8Kernel(const float* __restrict__ src,
                                   uint8_t* __restrict__ dst,
                                   int totalElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < totalElements) {
        float val = src[idx];
        // Clamp and scale 0.0-1.0 to 0-255
        val = (val < 0.0f) ? 0.0f : (val > 1.0f) ? 1.0f : val;
        dst[idx] = static_cast<uint8_t>(val * 255.0f + 0.5f);
    }
}

}

CudaError FloatToUint8(const float* src,
                         uint8_t* dst,
                         int totalElements) {
    if (src == nullptr || dst == nullptr || totalElements <= 0) {
        return CudaError(ERROR_SOURCE, "FloatToUint8 invalid input: Null pointer or zero size");
    }
    KernelGrid grid(totalElements);
    FloatToUint8Kernel<<<grid.gsize(), grid.bsize()>>>(src, dst, totalElements);
    return CudaError(ERROR_SOURCE, cudaGetLastError());
}

} // namespace cropandweed
