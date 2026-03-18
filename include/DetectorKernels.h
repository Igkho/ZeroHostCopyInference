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

// Decodes raw CNN output into candidate structures
CudaError DecodeAndFilter(const Block<float> &d_output,
                          TypedBlock<DetectionRaw> &candidateBuffer,
                          BoundaryBlock<int> &countBuffer,
                          int batchSize,
                          int numAnchors,
                          int numClasses,
                          float confThreshold,
                          cudaStream_t stream = 0);

// Runs NMS and Unpacks results into strided buffer [Batch0][Batch1]...
CudaError RunNMS(TypedBlock<DetectionRaw> &candidateBuffer,
                 BoundaryBlock<int> &candidateCountBuffer,
                 BoundaryTypedBlock<DetectionRaw> &finalOutputBuffer,
                 BoundaryBlock<int> &finalOutputCount,
                 Block<uint8_t> &maskBuffer,
                 float nmsThreshold,
                 int maxOutputPerBatch,       // The stride (MAX_DETECTIONS_PER_FRAME)
                 int batchSize,
                 cudaStream_t stream = 0);

}
