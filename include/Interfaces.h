#pragma once
#include <string>
#include <vector>
#include <memory>
#include "Block.h"
#include "helpers.h"
#include "BatchData.h"
#include "BatchDetections.h"

namespace cropandweed {

struct ModelProperties {
    size_t inputWidth = 0;
    size_t inputHeight = 0;
    int numClasses = 0;
    std::vector<std::string> classNames;
};

class ISource {
public:
    virtual ~ISource() = default;
    virtual CudaError GetNextBatch(BatchData& outBatch, size_t batchSize, bool &process) = 0;
};

class IDetector {
public:
    virtual ~IDetector() = default;
    virtual CudaError Detect(const BatchData& input, BatchDetections &output) = 0;
    virtual ModelProperties GetModelProperties() const = 0;
};

class ISink {
public:
    virtual ~ISink() = default;
    virtual CudaError Save(BatchData& batch, BatchDetections& results) = 0;
    virtual CudaError Close() = 0;
};

} // namespace cropandweed
