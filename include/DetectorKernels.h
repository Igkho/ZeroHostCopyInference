#pragma once
#include <vector>
#include "Interfaces.h"
#include "Block.h"
#include "DetectionRaw.h"
#include "helpers.h"

namespace cropandweed {

struct Detection {
    float x, y, w, h;
    float score;
    int classId;
    std::string label;
};

    CudaError DecodeAndFilter(const float* d_output,
//                                Block<uint8_t>& candidateBuffer,
                                uint8_t *candidateBuffer,
                                int candidateBufferSize,
//                                Block<int>& countBuffer,
                                int *countBuffer,
                                int batchSize,
                                int numAnchors,
                                int numClasses,
                                float confThreshold,
                                cudaStream_t stream = 0);

CudaError RunNMS_GPU(uint8_t *candidateBuffer,
                     int candidateBufferSize,
                     int *countBuffer,
                     uint8_t *finalOutputBuffer,
                     int finalOutputBufferSize,
                     int *finalOutputCount,
                     Block<uint8_t> &maskBuffer,
                     float nmsThreshold,
                     cudaStream_t stream = 0);

                         // Block<uint8_t>& finalOutputBuffer,
                         //   Block<int>& finalOutputCount,
                         //   Block<uint8_t> &maskBuffer,
                         //   float nmsThreshold,
                         //   cudaStream_t stream = 0);

    std::vector<std::vector<Detection>> RunNMS_CPU(uint8_t *candidateBuffer,
                                                   int candidateBufferSize,
                                                   int *countBuffer,
                                                   int batchSize,
                                                   float nmsThreshold);

/*
    std::vector<std::vector<Detection>> ConvertGpuResultsToVector(
        Block<uint8_t>& gpuData,
        Block<int>& gpuCount,
        int batchSize);
};
*/

}
