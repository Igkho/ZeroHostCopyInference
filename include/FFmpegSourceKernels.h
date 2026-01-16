#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include "helpers.h"

namespace cropandweed {

CudaError NV12ToRGBPlanar(const uint8_t* srcY,
                          const uint8_t* srcUV,
                          int srcPitch,
                          float* dstBatch,
                          unsigned int batchIndex,
                          unsigned int srcW, unsigned int srcH,  // Source Dimensions
                          unsigned int dstW, unsigned int dstH, // Terget Dimensions
                          bool use_texture_objects = true,
                          cudaStream_t stream = 0
                          );

}
