#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include "helpers.h"

namespace cropandweed {

CudaError FloatToUint8(const float* src,
                       uint8_t* dst,
                       int totalElements,
                       cudaStream_t stream);

// NV12 encoding kernel
CudaError RGBPlanarToNV12(const float* src_rgb,
                          uint8_t* dst_y, uint8_t* dst_uv,
                          int pitch, int batchIndex,
                          int width, int height,
                          cudaStream_t stream);

} //namespace cropandweed
