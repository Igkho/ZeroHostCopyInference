#include "SourceKernels.h"
#include "helpers.h"
#include <algorithm>
#include <cstring>

namespace cropandweed {

namespace {

// Normalized offsets for YUV (BT.601)
// 16 / 255.0f ~= 0.062745f
// 128 / 255.0f ~= 0.501961f
__constant__ float OFFSET_Y_NORM = 0.062745f;
__constant__ float OFFSET_UV_NORM = 0.501961f;


__device__ inline void YUVToRGB_Normalized(float y, float u, float v, float& r, float& g, float& b) {
    // 1. Subtract offsets (in normalized 0..1 space)
    float y_s = y - OFFSET_Y_NORM;
    float u_s = u - OFFSET_UV_NORM;
    float v_s = v - OFFSET_UV_NORM;

    // 2. Standard BT.601 conversion
    // Coefficients remain the same, they simply scale the normalized range now
    r = 1.164f * y_s + 1.596f * v_s;
    g = 1.164f * y_s - 0.813f * v_s - 0.391f * u_s;
    b = 1.164f * y_s + 2.018f * u_s;

    // 3. Clamp (0.0 to 1.0)
    r = fminf(fmaxf(r, 0.0f), 1.0f);
    g = fminf(fmaxf(g, 0.0f), 1.0f);
    b = fminf(fmaxf(b, 0.0f), 1.0f);
}

__global__ void NV12ToRGBPlanarTextureKernel(
    cudaTextureObject_t texY,
    cudaTextureObject_t texUV,
    float* __restrict__ dstBase,
    int batchOffsetPixels,
    int dstWidth, int dstHeight)
{
    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (dst_x >= dstWidth || dst_y >= dstHeight) return;

    float u = (dst_x + 0.5f) / dstWidth;
    float v = (dst_y + 0.5f) / dstHeight;

    // Hardware Interpolation with cudaReadModeNormalizedFloat
    // Returns 0.0 to 1.0 directly
    float y_val = tex2D<float>(texY, u, v);

    float2 uv_val = tex2D<float2>(texUV, u, v);
    float u_val = uv_val.x;
    float v_val = uv_val.y;

    // Convert using Normalized math
    float r, g, b;
    YUVToRGB_Normalized(y_val, u_val, v_val, r, g, b);

    int planePixels = dstWidth * dstHeight;
    float* imgPtr = dstBase + batchOffsetPixels;
    imgPtr[0 * planePixels + dst_y * dstWidth + dst_x] = r;
    imgPtr[1 * planePixels + dst_y * dstWidth + dst_x] = g;
    imgPtr[2 * planePixels + dst_y * dstWidth + dst_x] = b;
}

// Device helper: Manual Bilinear Interpolate
// Returns a normalized float [0.0, 1.0]
__device__ inline float interpolate_pixel_norm(
    const uint8_t* __restrict__ src,
    int pitch,
    int w, int h,
    float x, float y
    ) {
    int x1 = (int)x;
    int y1 = (int)y;
    int x2 = x1 + 1;
    int y2 = y1 + 1;

    if (x2 >= w) x2 = w - 1;
    if (y2 >= h) y2 = h - 1;

    float dx = x - x1;
    float dy = y - y1;

    // Convert to normalized float immediately
    float a = (float)src[y1 * pitch + x1] / 255.0f;
    float b = (float)src[y1 * pitch + x2] / 255.0f;
    float c = (float)src[y2 * pitch + x1] / 255.0f;
    float d = (float)src[y2 * pitch + x2] / 255.0f;

    return (1.0f - dx) * (1.0f - dy) * a +
           dx * (1.0f - dy) * b +
           (1.0f - dx) * dy * c +
           dx * dy * d;
}

// Device helper: Fetch UV Bilinearly
// Returns normalized floats [0.0, 1.0]
__device__ inline void interpolate_uv_norm(
    const uint8_t* __restrict__ srcUV,
    int pitch,
    int w_chroma, int h_chroma,
    float x_chroma, float y_chroma,
    float& out_u, float& out_v
    ) {
    int x1 = (int)x_chroma;
    int y1 = (int)y_chroma;
    int x2 = x1 + 1;
    int y2 = y1 + 1;

    if (x2 >= w_chroma) x2 = w_chroma - 1;
    if (y2 >= h_chroma) y2 = h_chroma - 1;

    float dx = x_chroma - x1;
    float dy = y_chroma - y1;

    auto get_val_norm = [&](int xx, int yy, int offset) -> float {
        return (float)srcUV[yy * pitch + (xx * 2) + offset] / 255.0f;
    };

    float u_a = get_val_norm(x1, y1, 0);
    float u_b = get_val_norm(x2, y1, 0);
    float u_c = get_val_norm(x1, y2, 0);
    float u_d = get_val_norm(x2, y2, 0);
    out_u = (1.0f - dx)*(1.0f - dy)*u_a + dx*(1.0f - dy)*u_b + (1.0f - dx)*dy*u_c + dx*dy*u_d;

    float v_a = get_val_norm(x1, y1, 1);
    float v_b = get_val_norm(x2, y1, 1);
    float v_c = get_val_norm(x1, y2, 1);
    float v_d = get_val_norm(x2, y2, 1);
    out_v = (1.0f - dx)*(1.0f - dy)*v_a + dx*(1.0f - dy)*v_b + (1.0f - dx)*dy*v_c + dx*dy*v_d;
}

__global__ void NV12ToRGBPlanarKernel(
    const uint8_t* __restrict__ srcY,
    const uint8_t* __restrict__ srcUV,
    int srcPitch,
    float* __restrict__ dstBase,
    int batchOffsetPixels,
    int srcWidth, int srcHeight,
    int dstWidth, int dstHeight)
{
    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (dst_x >= dstWidth || dst_y >= dstHeight) return;

    float scale_x = (float)srcWidth / (float)dstWidth;
    float scale_y = (float)srcHeight / (float)dstHeight;

    float src_fx = (dst_x + 0.5f) * scale_x - 0.5f;
    float src_fy = (dst_y + 0.5f) * scale_y - 0.5f;

    src_fx = fmaxf(0.0f, fminf(src_fx, (float)srcWidth - 1.001f));
    src_fy = fmaxf(0.0f, fminf(src_fy, (float)srcHeight - 1.001f));

    // Get Normalized Y
    float Y = interpolate_pixel_norm(srcY, srcPitch, srcWidth, srcHeight, src_fx, src_fy);

    float uv_fx = src_fx * 0.5f;
    float uv_fy = src_fy * 0.5f;
    int w_chroma = srcWidth / 2;
    int h_chroma = srcHeight / 2;
    uv_fx = fmaxf(0.0f, fminf(uv_fx, (float)w_chroma - 1.001f));
    uv_fy = fmaxf(0.0f, fminf(uv_fy, (float)h_chroma - 1.001f));

    // Get Normalized UV
    float U, V;
    interpolate_uv_norm(srcUV, srcPitch, w_chroma, h_chroma, uv_fx, uv_fy, U, V);

    // Convert using Normalized math
    float r, g, b;
    YUVToRGB_Normalized(Y, U, V, r, g, b);

    int planePixels = dstWidth * dstHeight;
    float* imgPtr = dstBase + batchOffsetPixels;
    imgPtr[0 * planePixels + dst_y * dstWidth + dst_x] = r;
    imgPtr[1 * planePixels + dst_y * dstWidth + dst_x] = g;
    imgPtr[2 * planePixels + dst_y * dstWidth + dst_x] = b;
}

}

CudaError NV12ToRGBPlanar(
    const uint8_t* srcY, const uint8_t* srcUV, int srcPitch,
    float* dstBatch,
    unsigned int batchIndex,
    unsigned int srcW, unsigned int srcH,
    unsigned int dstW, unsigned int dstH,
    bool use_texture_objects,
    cudaStream_t stream)
{
    if (srcY == nullptr || srcUV == nullptr || srcPitch < 0 ||
        dstBatch == nullptr || // batchIndex < 0 ||
        srcW <= 0 || srcH <= 0 || dstW <= 0 || dstH <= 0) {
            return CudaError(ERROR_SOURCE, "NV12ToRGBPlanar invalid input");
    }
    int batchOffsetElements = batchIndex * (dstW * dstH * 3);
    KernelGrid grid({dstW, dstH}, {16, 16});

    if (use_texture_objects) {
        struct cudaResourceDesc resDescY;
        std::memset(&resDescY, 0, sizeof(resDescY));
        resDescY.resType = cudaResourceTypePitch2D;
        resDescY.res.pitch2D.devPtr = (void*)srcY;
        resDescY.res.pitch2D.desc = cudaCreateChannelDesc<uint8_t>();
        resDescY.res.pitch2D.width = srcW;
        resDescY.res.pitch2D.height = srcH;
        resDescY.res.pitch2D.pitchInBytes = srcPitch;

        struct cudaResourceDesc resDescUV;
        std::memset(&resDescUV, 0, sizeof(resDescUV));
        resDescUV.resType = cudaResourceTypePitch2D;
        resDescUV.res.pitch2D.devPtr = (void*)srcUV;
        resDescUV.res.pitch2D.desc = cudaCreateChannelDesc<uchar2>();
        resDescUV.res.pitch2D.width = srcW / 2;
        resDescUV.res.pitch2D.height = srcH / 2;
        resDescUV.res.pitch2D.pitchInBytes = srcPitch;

        struct cudaTextureDesc texDesc;
        std::memset(&texDesc, 0, sizeof(texDesc));

        // Key setting: Use NormalizedFloat to get 0.0-1.0 and enable hardware linear filtering for uint8
        texDesc.readMode = cudaReadModeNormalizedFloat;
        texDesc.filterMode = cudaFilterModeLinear;
        texDesc.addressMode[0] = cudaAddressModeClamp;
        texDesc.addressMode[1] = cudaAddressModeClamp;
        texDesc.normalizedCoords = 1;

        cudaTextureObject_t texYObj = 0;
        cudaTextureObject_t texUVObj = 0;

        CUDA_TRY(cudaCreateTextureObject(&texYObj, &resDescY, &texDesc, NULL));
        CUDA_TRY(cudaCreateTextureObject(&texUVObj, &resDescUV, &texDesc, NULL));

        NV12ToRGBPlanarTextureKernel<<<grid.gsize(), grid.bsize(), 0, stream>>>(
            texYObj, texUVObj,
            dstBatch, batchOffsetElements,
            dstW, dstH
            );
        CUDA_CHECK_KERNEL(stream);
        CUDA_TRY(cudaStreamSynchronize(stream));
        CUDA_TRY(cudaDestroyTextureObject(texYObj));
        CUDA_TRY(cudaDestroyTextureObject(texUVObj));
    } else {
        NV12ToRGBPlanarKernel<<<grid.gsize(), grid.bsize(), 0, stream>>>(
            srcY, srcUV, srcPitch,
            dstBatch, batchOffsetElements,
            srcW, srcH,
            dstW, dstH
            );
        CUDA_CHECK_KERNEL(stream);
        CUDA_TRY(cudaStreamSynchronize(stream));
    }

    return CudaError(); //CudaError(ERROR_SOURCE, cudaGetLastError());
}

namespace {

// Kernel to resize (Bilinear) and cast uint8_t image to normalized float[0..1] array
__global__ void ResizeAndCastRGBPlanarKernel(
    const uint8_t* __restrict__ srcR,
    const uint8_t* __restrict__ srcG,
    const uint8_t* __restrict__ srcB,
    int srcW, int srcH, int srcPitch,
    float* __restrict__ dstBase,
    int dstW, int dstH)
{
    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (dst_x >= dstW || dst_y >= dstH) return;

    // Bilinear Interpolation Scales
    float scale_x = (float)srcW / (float)dstW;
    float scale_y = (float)srcH / (float)dstH;

    // Centered pixel mapping
    float src_fx = (dst_x + 0.5f) * scale_x - 0.5f;
    float src_fy = (dst_y + 0.5f) * scale_y - 0.5f;

    // Clamp to valid boundaries to prevent out-of-bounds reads
    src_fx = fmaxf(0.0f, fminf(src_fx, (float)srcW - 1.001f));
    src_fy = fmaxf(0.0f, fminf(src_fy, (float)srcH - 1.001f));

    int x1 = (int)src_fx;
    int y1 = (int)src_fy;
    int x2 = min(x1 + 1, srcW - 1);
    int y2 = min(y1 + 1, srcH - 1);

    float dx = src_fx - x1;
    float dy = src_fy - y1;

    // Inline closure for bilinear math across arbitrary planes
    auto bilerp = [&](const uint8_t* plane) {
        float a = plane[y1 * srcPitch + x1];
        float b = plane[y1 * srcPitch + x2];
        float c = plane[y2 * srcPitch + x1];
        float d = plane[y2 * srcPitch + x2];
        return (1.0f - dx) * (1.0f - dy) * a +
               dx * (1.0f - dy) * b +
               (1.0f - dx) * dy * c +
               dx * dy * d;
    };

    // Calculate, cast, and normalize to 0.0 - 1.0 range
    float r = bilerp(srcR) / 255.0f;
    float g = bilerp(srcG) / 255.0f;
    float b = bilerp(srcB) / 255.0f;

    int planeSize = dstW * dstH;
    int dst_idx = dst_y * dstW + dst_x;

    // Write to coalesced memory layout (Planar format [RRR...GGG...BBB])
    dstBase[dst_idx] = r;
    dstBase[dst_idx + planeSize] = g;
    dstBase[dst_idx + 2 * planeSize] = b;
}

} // anonymous namespace

CudaError ResizeAndCastRGBPlanar(const uint8_t* srcR,
                                 const uint8_t* srcG,
                                 const uint8_t* srcB,
                                 int srcW, int srcH, int srcPitch,
                                 float* dstBase, int dstW, int dstH,
                                 cudaStream_t stream)
{
    // Defensive barrier
    if (!srcR || !srcG || !srcB || !dstBase || srcW <= 0 || srcH <= 0 || dstW <= 0 || dstH <= 0 || srcPitch <= 0) {
        return CudaError(ERROR_SOURCE, "Invalid arguments to ResizeAndCastRGBPlanar");
    }

    KernelGrid grid({(unsigned int)dstW, (unsigned int)dstH}, {16, 16});

    ResizeAndCastRGBPlanarKernel<<<grid.gsize(), grid.bsize(), 0, stream>>>(
        srcR, srcG, srcB, srcW, srcH, srcPitch, dstBase, dstW, dstH
        );
    CUDA_CHECK_KERNEL(stream);

    return CudaError();
}

} // namespace cropandweed

