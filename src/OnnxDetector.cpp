#include "OnnxDetector.h"
#include "DetectionRaw.h"
#include "DetectorKernels.h"
#include <iostream>

using namespace cropandweed;

CudaError OnnxDetector::Init(const std::string& modelPath) {
    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // Enable CUDA
    OrtCUDAProviderOptions cudaOptions;
    cudaOptions.device_id = 0;
//    size_t limit = 6ULL * 1024 * 1024 * 1024;
//    cudaOptions.gpu_mem_limit = limit;
//    cudaOptions.arena_extend_strategy = 1;

    // Initialize Session
    try {
        // This throws if CUDA libraries are missing or incompatible
        options.AppendExecutionProvider_CUDA(cudaOptions);

        // This throws if the model path is invalid
        session_ = std::make_unique<Ort::Session>(env_, modelPath.c_str(), options);
    } catch (const Ort::Exception& e) {
        return CudaError(ERROR_SOURCE, std::string("ONNX Runtime Init Error: ") + e.what());
    }

    // Create IoBinding
    try {
        ioBinding_ = std::make_unique<Ort::IoBinding>(*session_);
    } catch (const Ort::Exception& e) {
        return CudaError(ERROR_SOURCE, std::string("ONNX IoBinding Error: ") + e.what());
    }

    CUDA_TRY(CudaStream::Create(cuda_stream_, cudaStreamNonBlocking));

    // 1. READ INPUT DIMENSIONS
    try {
        auto typeInfo = session_->GetInputTypeInfo(0);
        auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
        auto shape = tensorInfo.GetShape();

        if (shape.size() >= 4) {
            int64_t h = shape[2];
            int64_t w = shape[3];
            inputH_ = (h > 0) ? (size_t)h : 1024;
            inputW_ = (w > 0) ? (size_t)w : 1024;
            std::cout << "[OnnxDetector] Model Input: " << inputW_ << "x" << inputH_ << std::endl;
        }
    } catch (const Ort::Exception& e) {
        return CudaError(ERROR_SOURCE, std::string("ONNX Input Shape Error: ") + e.what());
    }

    // 2. READ OUTPUT DIMENSIONS (For Pre-allocation)
    int channels = 84;
    int anchors = 8400; // Default for 640x640
    try {
        auto outTypeInfo = session_->GetOutputTypeInfo(0);
        auto outTensorInfo = outTypeInfo.GetTensorTypeAndShapeInfo();
        auto outShape = outTensorInfo.GetShape(); // e.g. [Batch, 84, 8400]

        if (outShape.size() > 1 && outShape[1] > 0) channels = (int)outShape[1];
//        if (outShape.size() > 2 && outShape[2] > 0) anchors = (int)outShape[2];
        // YOLO Standard Strides: 8, 16, 32
        if (inputW_ > 0 && inputH_ > 0) {
            float height = inputH_;
            float width = inputW_;
            int calcAnchors = (floor(floor(floor(height/2 - 1./2)/2)/2) + 1) *
                                  (floor(floor(floor(width/2 - 1./2)/2)/2) + 1) +
                                  (floor(floor(floor(floor(height/2 - 1./2)/2)/2)/2) + 1) *
                                          (floor(floor(floor(floor(width/2 - 1./2)/2)/2)/2) + 1) +
                                  (floor(floor(floor(floor(floor(height/2 - 1./2)/2)/2)/2)/2) + 1) *
                                          (floor(floor(floor(floor(floor(width/2 - 1./2)/2)/2)/2)/2) + 1);
            if (outShape.size() > 2 && outShape[2] > 0 && outShape[2] != calcAnchors) {
                std::cout << "[OnnxDetector] Metadata reported " << outShape[2]
                          << " anchors, but input resolution " << inputW_ << "x" << inputH_
                          << " requires " << calcAnchors << ". Using calculated value." << std::endl;
            }
            anchors = calcAnchors;
        }
        std::cout << "[OnnxDetector] Output detected: " << channels << " channels, " << anchors << " anchors" << std::endl;
    } catch (const Ort::Exception& e) {
        // Fallback to defaults if output 0 isn't standard
        std::cout << "[OnnxDetector] Warning: Could not determine output shape, using defaults (84x8400)" << std::endl;
    }

    // 3. INITIALIZE POOL
    pool_.resize(POOL_SIZE);

    size_t rawVol = BatchData::MAX_BATCH_SIZE * channels * anchors;
    size_t finalVol = BatchData::MAX_BATCH_SIZE * BatchDetections::MAX_DETECTIONS_PER_FRAME * sizeof(DetectionRaw);
    int maxCandidates = BatchDetections::MAX_DETECTIONS_PER_FRAME * 10;

    for (auto& res : pool_) {
//        std::cout << "Reserving floats: " << rawVol << std::endl;
        CUDA_TRY(res.rawOutput.reserve(rawVol));
        CUDA_TRY(res.candidates.reserve(maxCandidates * sizeof(DetectionRaw)));
        CUDA_TRY(res.candidateCount.reserve(1));

        // Ensure result structure is ready
        CUDA_TRY(res.result.data.reserve(finalVol));
        CUDA_TRY(res.result.counts.reserve(BatchData::MAX_BATCH_SIZE));

        // Zero out counts safely
        CUDA_TRY(cudaMemsetAsync(res.candidateCount.data(), 0, sizeof(int), *cuda_stream_));
    }

    std::cout << "[OnnxDetector] Memory pool initialized (Depth: " << POOL_SIZE << ")" << std::endl;
    return CudaError();
}

std::pair<size_t, size_t> OnnxDetector::GetInputSize() const {
    return {inputW_, inputH_};
}

CudaError OnnxDetector::Detect(const BatchData& input, BatchDetections &output) {
    FrameResources& res = pool_[poolIndex_];
    poolIndex_ = (poolIndex_ + 1) % POOL_SIZE;

    if (!session_) {
        return CudaError(ERROR_SOURCE, "Detector not initialized");
    }

    if (input.readyEvent) {
        CUDA_TRY(cudaStreamWaitEvent(*cuda_stream_, *input.readyEvent, 0));
//        CUDA_TRY(cudaStreamSynchronize(*cuda_stream_));
    }

    // Determine Shapes
    int channels = 84;
    float height = input.height;
    float width = input.width;
    int anchors = (floor(floor(floor(height/2 - 1./2)/2)/2) + 1) *
                          (floor(floor(floor(width/2 - 1./2)/2)/2) + 1) +
                      (floor(floor(floor(floor(height/2 - 1./2)/2)/2)/2) + 1) *
                          (floor(floor(floor(floor(width/2 - 1./2)/2)/2)/2) + 1) +
                      (floor(floor(floor(floor(floor(height/2 - 1./2)/2)/2)/2)/2) + 1) *
                          (floor(floor(floor(floor(floor(width/2 - 1./2)/2)/2)/2)/2) + 1);
    try {
        auto outTypeInfo = session_->GetOutputTypeInfo(0);
        auto outTensorInfo = outTypeInfo.GetTensorTypeAndShapeInfo();
        auto outShape = outTensorInfo.GetShape();
        if (outShape.size() > 1 && outShape[1] > 0) channels = (int)outShape[1];
//        if (outShape.size() > 2 && outShape[2] > 0) anchors = (int)outShape[2];
    } catch(const Ort::Exception& e) {
        return CudaError(ERROR_SOURCE, std::string("ONNX Output Info Error: ") + e.what());
    }

    // Resize Buffers (Efficient Reuse)
    CUDA_TRY(res.rawOutput.resize(input.batchSize * channels * anchors));
    CUDA_TRY(res.candidates.resize(BatchDetections::MAX_DETECTIONS_PER_FRAME * 10 * sizeof(DetectionRaw)));
    CUDA_TRY(res.candidateCount.resize(1));
    CUDA_TRY(res.result.data.resize(input.batchSize * BatchDetections::MAX_DETECTIONS_PER_FRAME * sizeof(DetectionRaw)));
    CUDA_TRY(res.result.counts.resize(input.batchSize));

    // Ensure Events exist
    if (!res.result.readyEvent) {
        CUDA_TRY(CudaEvent::Create(res.result.readyEvent));
    }

    // Setup Input Shape [Batch, 3, H, W]
    std::vector<int64_t> inputShape = {
        (int64_t)input.batchSize, 3, (int64_t)input.height, (int64_t)input.width
    };

    // Create Memory Info (Tells ORT the data is on GPU ID 0)
    Ort::MemoryInfo memInfo("Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault);

    // Create Input Tensor (No Copy, View over Block data)
    // It does NOT allocate new GPU memory and does NOT copy data.
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memInfo,
        const_cast<float*>(input.deviceData.data()),
        input.deviceData.size(),
        inputShape.data(),
        inputShape.size()
        );

    // Bind the Tensor to the Input Name
    ioBinding_->BindInput("images", inputTensor);

    // Setup Output Binding
    std::vector<int64_t> outputShape = {
        (int64_t)input.batchSize, (int64_t)channels, (int64_t)anchors
    };

    Ort::Value outputTensor = Ort::Value::CreateTensor<float>(
        memInfo,
        res.rawOutput.data(),
        res.rawOutput.size(),
        outputShape.data(),
        outputShape.size()
        );

    // Bind this tensor to the output name
    ioBinding_->BindOutput("output0", outputTensor);

    // Run Inference
    try {
        session_->Run(Ort::RunOptions{nullptr}, *ioBinding_);
    } catch (const Ort::Exception& e) {
        return CudaError(ERROR_SOURCE, std::string("ONNX Inference Error: ") + e.what());
    }

    // Post-Processing (Kernel)
    CUDA_TRY(DecodeAndFilter(
        res.rawOutput.data(),
        res.candidates.data(),
        res.candidates.size(),
        res.candidateCount.data(),
        input.batchSize,
        anchors,
        channels - 4,
        0.25f,
        *cuda_stream_
    ));

    // NMS (Kernel)
    CUDA_TRY(RunNMS_GPU(
        res.candidates.data(),
        res.candidates.size(),
        res.candidateCount.data(),
        res.result.data.data(),
        res.result.data.size(),
        res.result.counts.data(),
        res.nmsMask,
        0.45f,
        *cuda_stream_
    ));

    // Record Event
    CUDA_TRY(cudaEventRecord(*res.result.readyEvent, *cuda_stream_));

    // Cleanup Bindings
    ioBinding_->ClearBoundInputs();
    ioBinding_->ClearBoundOutputs();

    // Swap Results (Efficient Memory Exchange)
    // 'output' gets the new data.
    // 'res.result' gets 'output's old buffer (reclaiming it for the pool).
    std::swap(output, res.result);

    return CudaError();
}
