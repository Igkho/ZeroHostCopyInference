#include "SinkKernels.h"
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
                         int totalElements,
                         cudaStream_t stream
                       ) {
    if (src == nullptr || dst == nullptr || totalElements <= 0) {
        return CudaError(ERROR_SOURCE, "FloatToUint8 invalid input: Null pointer or zero size");
    }
    KernelGrid grid(totalElements);
    FloatToUint8Kernel<<<grid.gsize(), grid.bsize(), 0, stream>>>(src, dst, totalElements);
    CUDA_CHECK_KERNEL(stream);
    return CudaError();
}

namespace {

// YUV colorspace conversion for MMAPI hardware
__global__ void RGBPlanarToNV12Kernel(const float* __restrict__ src_rgb,
                                      uint8_t* __restrict__ dst_y,
                                      uint8_t* __restrict__ dst_uv,
                                      int pitch, int batch_offset,
                                      int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int frame_pixels = width * height;
    int src_idx = batch_offset * (frame_pixels * 3) + (y * width + x);

    // Read RGB (Assuming Planar Float [0, 1] range)
    float r = src_rgb[src_idx] * 255.0f;
    float g = src_rgb[src_idx + frame_pixels] * 255.0f;
    float b = src_rgb[src_idx + 2 * frame_pixels] * 255.0f;

    // Standard BT.601 YUV conversion
    float Y =  0.299f * r + 0.587f * g + 0.114f * b;
    float U = -0.169f * r - 0.331f * g + 0.500f * b + 128.0f;
    float V =  0.500f * r - 0.419f * g - 0.081f * b + 128.0f;

    // Write Y plane (Pitch Linear)
    dst_y[y * pitch + x] = (uint8_t)fminf(fmaxf(Y, 0.0f), 255.0f);

    // NV12 Subsampling: 1 UV pair per 2x2 block
    if (x % 2 == 0 && y % 2 == 0) {
        int uv_idx = (y / 2) * pitch + x;
        dst_uv[uv_idx]     = (uint8_t)fminf(fmaxf(U, 0.0f), 255.0f);
        dst_uv[uv_idx + 1] = (uint8_t)fminf(fmaxf(V, 0.0f), 255.0f);
    }
}

}

CudaError RGBPlanarToNV12(const float* src_rgb, uint8_t* dst_y, uint8_t* dst_uv,
                          int pitch, int batchIndex, int width, int height, cudaStream_t stream) {
    KernelGrid grid({(unsigned int)width, (unsigned int)height}, {16, 16});
    RGBPlanarToNV12Kernel<<<grid.gsize(), grid.bsize(), 0, stream>>>(src_rgb, dst_y, dst_uv, pitch, batchIndex, width, height);
    CUDA_CHECK_KERNEL(stream);
    return CudaError();
}

} // namespace cropandweed
