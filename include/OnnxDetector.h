#pragma once
#include "Interfaces.h"
#include <onnxruntime_cxx_api.h>
#include <memory>
#include <vector>
#include "Block.h"
#include "FrameResources.h"

namespace cropandweed {


class OnnxDetector : public IDetector {
private:
    // Passkey Idiom: Restricts constructor access
    struct Token {};

public:
    // Public Constructor (effectively private due to Token)
    OnnxDetector(Token): env_(ORT_LOGGING_LEVEL_WARNING, "OnnxDetector") {}

    // Destructor
    ~OnnxDetector() override {}

    // --- Factory Method ---
    static CudaError Create(std::unique_ptr<IDetector>& out, const std::string &modelPath) {
        auto ptr = std::make_unique<OnnxDetector>(Token{});
        CUDA_TRY(ptr->Init(modelPath));
        out = std::move(ptr);
        return CudaError();
    }

    ModelProperties GetModelProperties() const override {
        ModelProperties props;
        props.inputWidth = inputW_;
        props.inputHeight = inputH_;
        // Logic to derive classes from output tensor shape (calculated in Init)
        // For YOLO: Channels (84) - 4 = 80 classes
        props.numClasses = 80; // Ideally strictly calculated from outputShape in Init()
        return props;
    }

    // Main Interface
    CudaError Detect(const BatchData& input, BatchDetections &output) override;

    // Helpers
    std::pair<size_t, size_t> GetInputSize() const;

private:
    CudaError Init(const std::string& modelPath);

    Ort::Env env_;
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::IoBinding> ioBinding_;
    std::unique_ptr<CudaStream> cuda_stream_;

    size_t inputW_ = 1024;
    size_t inputH_ = 1024;

    static const int POOL_SIZE = 4;
    std::vector<FrameResources> pool_;
    int poolIndex_ = 0;
};

}
