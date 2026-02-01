#pragma once
#include <memory>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include "Interfaces.h"
#include "Block.h"
#include "FrameResources.h"
#include "helpers.h"

namespace cropandweed {

// Custom Deleter for TRT pointers
struct TrtDeleter {
    template <typename T>
    void operator()(T* obj) const {
        if (obj) delete obj; // Newer TRT uses delete, older used destroy()
    }
};

class TrtDetector : public IDetector {
private:
    // Passkey Idiom
    struct Token {};

public:
    // Public Constructor (effectively private)
    TrtDetector(Token);

    // Destructor
    ~TrtDetector() override;

    // --- Factory Method ---
    static CudaError Create(std::unique_ptr<IDetector>& out, const std::string &modelPath) {
        auto ptr = std::make_unique<TrtDetector>(Token{});
        CUDA_TRY(ptr->Init(modelPath));
        out = std::move(ptr);
        return CudaError();
    }

    ModelProperties GetModelProperties() const override {
        ModelProperties props;
        props.inputWidth = (inputDims_.d[3] > 0) ? inputDims_.d[3] : 1024;
        props.inputHeight = (inputDims_.d[2] > 0) ? inputDims_.d[2] : 1024;

        // Assuming YOLO output [Batch, Channels, Anchors]
        // Channels = 4 (xywh) + NumClasses
        if (outputDims_.nbDims >= 2) {
            props.numClasses = outputDims_.d[1] - 4;
        }
        return props;
    }

    CudaError Detect(const BatchData& input, BatchDetections &output) override;

private:

    // Internal Initialization
    CudaError Init(const std::string& modelPath);

    // Helpers
    CudaError LoadEngine(const std::string& enginePath);
    CudaError BuildEngine(const std::string& onnxPath, const std::string& savePath);

    // TensorRT Core objects
    std::shared_ptr<nvinfer1::ILogger> logger_;
    std::shared_ptr<nvinfer1::ICudaEngine> engine_;
    std::shared_ptr<nvinfer1::IExecutionContext> context_;

    std::unique_ptr<CudaStream> cuda_stream_;

    // Metadata
    std::string inputName_;
    std::string outputName_;
    nvinfer1::Dims inputDims_;
    nvinfer1::Dims outputDims_;

    static const int POOL_SIZE = 4;
    std::vector<FrameResources> pool_;
    int poolIndex_ = 0;
};

} // namespace cropandweed
