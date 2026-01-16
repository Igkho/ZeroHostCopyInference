#pragma once
#include "Interfaces.h"
#include "Block.h"
#include "BatchDetections.h"
#include "helpers.h"
#include <vector>
#include <memory>

namespace cropandweed {

class StubDetector : public IDetector {
private:
    // Passkey Idiom for Factory
    struct Token {};

public:
    // Constructor (Lightweight)
    StubDetector(Token);
    ~StubDetector() override = default;

    // --- Strict Factory Method ---
    static CudaError Create(std::unique_ptr<IDetector>& out) {
        auto ptr = std::make_unique<StubDetector>(Token{});
        CUDA_TRY(ptr->Init());
        // Implicit move-conversion from unique_ptr<StubDetector> to unique_ptr<IDetector>
        out = std::move(ptr);
        return CudaError();
    }

    // Interface Implementation
    CudaError Detect(const BatchData& input, BatchDetections& output) override;
    std::pair<size_t, size_t> GetInputSize() const override;

private:
    // Internal Init
    CudaError Init();

    std::unique_ptr<CudaStream> cuda_stream_;

    // Resource Pool to reuse output buffers without allocation
    static const int POOL_SIZE = 4;
    std::vector<BatchDetections> pool_;
    int poolIndex_ = 0;
};

} // namespace cropandweed
