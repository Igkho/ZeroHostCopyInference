#pragma once
#include <string>
#include <vector>
#include <memory>
#include "Block.h"
#include "helpers.h"
#include "BatchData.h"
#include "BatchDetections.h"

namespace cropandweed {

class ISource {
public:
    virtual ~ISource() = default;
    virtual CudaError GetNextBatch(BatchData& outBatch, size_t batchSize, bool &process) = 0;
    virtual void SetOutputSize(size_t width, size_t height) = 0;
};

class IDetector {
public:
    virtual ~IDetector() = default;
    virtual CudaError Detect(const BatchData& input, BatchDetections &output) = 0;
    virtual std::pair<size_t, size_t> GetInputSize() const = 0;
};

class ISink {
public:
    virtual ~ISink() = default;
    virtual CudaError Save(const BatchData& batch, const BatchDetections& results) = 0;
};

} // namespace cropandweed
