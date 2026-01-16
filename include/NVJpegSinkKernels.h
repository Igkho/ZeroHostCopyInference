#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include "helpers.h"

namespace cropandweed {

CudaError FloatToUint8(const float* src,
                         uint8_t* dst,
                         int totalElements);

} //namespace cropandweed
