#include "TrtDetector.h"
#include "DetectionRaw.h"
#include "DetectorKernels.h"
#include "helpers.h"
#include <filesystem>
#include <fstream>
#include <iostream>

namespace fs = std::filesystem;

namespace cropandweed {

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << "[TRT] " << msg << std::endl;
    }
};

TrtDetector::TrtDetector(Token) {
    logger_ = std::make_unique<Logger>();
}

TrtDetector::~TrtDetector() {
    // Smart ptrs handle cleanup, but context needs explicit destruction before engine usually in older TRT
    context_.reset();
    engine_.reset();
    runtime_.reset();
}

CudaError TrtDetector::Init(const std::string& modelPath) {
    // Initialize Stream
    CUDA_TRY(CudaStream::Create(cuda_stream_, cudaStreamNonBlocking));

    fs::path inputPath(modelPath);
    fs::path enginePath = inputPath;

    // Determine the expected engine path
    if (inputPath.extension() != ".engine" && inputPath.extension() != ".plan") {
        enginePath = inputPath;
        enginePath.replace_extension(".engine");
    }

    // Load or Build
    bool loaded = false;
    if (fs::exists(enginePath)) {
        CudaError err = LoadEngine(enginePath.string());
        if (!CudaError::IsFailure(err)) {
            loaded = true;
        } else {
            std::cout << "[TRT] Cached engine failed to load (" << err.Text() << "). Rebuilding..." << std::endl;
        }
    }
    if (!loaded) {
        CUDA_TRY(BuildEngine(inputPath.string(), enginePath.string()));
    }

    // Pre-allocate Memory Pool
    pool_.resize(POOL_SIZE);

    int channels = outputDims_.d[1];
    int anchors = outputDims_.d[2];

    size_t rawVol = BatchData::MAX_BATCH_SIZE * channels * anchors;
    size_t maxCandidates = BatchData::MAX_BATCH_SIZE * BatchDetections::MAX_DETECTIONS_PER_FRAME * 10;

    for (auto& res : pool_) {
        CUDA_TRY(res.rawOutput.reserve(rawVol, *cuda_stream_));
        CUDA_TRY(res.candidates.reserve(maxCandidates, *cuda_stream_));
        CUDA_TRY(res.candidateCount.reserve(1, *cuda_stream_));
        CUDA_TRY(res.candidateCount.fill(0, *cuda_stream_));
    }
    CUDA_TRY(cudaStreamSynchronize(*cuda_stream_));
    return CudaError();
}

CudaError TrtDetector::DeserializeEngine(const void* blob, std::size_t size) {
    runtime_ = std::unique_ptr<nvinfer1::IRuntime, TrtDeleter>(nvinfer1::createInferRuntime(*logger_));
    if (!runtime_) {
        return CudaError(ERROR_SOURCE, "Failed to create TRT Runtime");
    }

    engine_ = std::unique_ptr<nvinfer1::ICudaEngine, TrtDeleter>(
        runtime_->deserializeCudaEngine(blob, size)
        );
    if (!engine_) {
        return CudaError(ERROR_SOURCE, "Failed to deserialize CUDA Engine");
    }

    context_ = std::unique_ptr<nvinfer1::IExecutionContext, TrtDeleter>(
        engine_->createExecutionContext()
        );
    if (!context_) {
        return CudaError(ERROR_SOURCE, "Failed to create Execution Context");
    }

    // Setup Metadata
    inputName_ = "images";
    outputName_ = "output0";
    inputDims_ = engine_->getTensorShape(inputName_.c_str());
    outputDims_ = engine_->getTensorShape(outputName_.c_str());

    std::cout << "[TRT] Engine ready for inference." << std::endl;
    return CudaError();
}

CudaError TrtDetector::LoadEngine(const std::string& enginePath) {

    std::ifstream file(enginePath, std::ios::binary | std::ios::ate);
    if (!file.good()) {
        CudaError(ERROR_SOURCE, "Could not open engine file: " + enginePath);
    }

    std::cout << "[TRT] Loading cached engine: " << enginePath << std::endl;
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        return CudaError(ERROR_SOURCE, "Failed to read engine file");
    }
    return DeserializeEngine(buffer.data(), size);
}

CudaError TrtDetector::BuildEngine(const std::string& onnxPath, const std::string& savePath) {
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(*logger_));
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, *logger_));

    if (!parser->parseFromFile(onnxPath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        return CudaError(ERROR_SOURCE, "Failed to parse ONNX file: " + onnxPath);
    }

    // Robust FP16 Enablement
    // We assume CC 7.0+ (Hardware FP16 support exists).
    // We surround the flag with pragmas to suppress TRT 10+ deprecation warnings.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    // kFP16 is deprecated in TRT 10 but required to enable FP16 tactics
    // without implementing full Strong Typing.
    config->setFlag(nvinfer1::BuilderFlag::kINT8);
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
#pragma GCC diagnostic pop
    std::cout << "[TRT] INT8 & FP16 Precision Enabled." << std::endl;

    auto profile = builder->createOptimizationProfile();
    bool hasDynamic = false;

    for (int i = 0; i < network->getNbInputs(); ++i) {
        auto input = network->getInput(i);
        auto dims = input->getDimensions();
        const char* name = input->getName();

        if (dims.nbDims == 4) {
            nvinfer1::Dims4 minDims = {dims.d[0], dims.d[1], dims.d[2], dims.d[3]};
            nvinfer1::Dims4 optDims = {dims.d[0], dims.d[1], dims.d[2], dims.d[3]};
            nvinfer1::Dims4 maxDims = {dims.d[0], dims.d[1], dims.d[2], dims.d[3]};

            // Handle Dynamic Batch
            if (minDims.d[0] == -1) {
                minDims.d[0] = (int)BatchData::MIN_BATCH_SIZE;
                optDims.d[0] = (int)BatchData::OPTIMUM_BATCH_SIZE;
                maxDims.d[0] = (int)BatchData::MAX_BATCH_SIZE;
                hasDynamic = true;
            }

            // Handle Dynamic H/W (Fallback to 1024 if unknown)
            if (minDims.d[2] == -1) {
                minDims.d[2] = 1024;
                optDims.d[2] = 1024;
                maxDims.d[2] = 1024;
                hasDynamic = true;
            }
            if (minDims.d[3] == -1) {
                minDims.d[3] = 1024;
                optDims.d[3] = 1024;
                maxDims.d[3] = 1024;
                hasDynamic = true;
            }

            if (hasDynamic) {
                std::cout << "[TRT] Adding optimization profile for input: " << name << "\n"
                          << "      Min: " << minDims.d[0] << "x" << minDims.d[1] << "x" << minDims.d[2] << "x" << minDims.d[3] << "\n"
                          << "      Opt: " << optDims.d[0] << "x" << optDims.d[1] << "x" << optDims.d[2] << "x" << optDims.d[3] << "\n"
                          << "      Max: " << maxDims.d[0] << "x" << maxDims.d[1] << "x" << maxDims.d[2] << "x" << maxDims.d[3] << std::endl;

                profile->setDimensions(name, nvinfer1::OptProfileSelector::kMIN, minDims);
                profile->setDimensions(name, nvinfer1::OptProfileSelector::kOPT, optDims);
                profile->setDimensions(name, nvinfer1::OptProfileSelector::kMAX, maxDims);
            }
        }
    }

    if (hasDynamic) {
        config->addOptimizationProfile(profile);
    }

    std::cout << "[TRT] Optimizing model..." << std::endl;
    auto plan = std::unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
    if (!plan) {
        std::cout << "[TRT] INT8 Calibration missing or failed. Falling back to strict FP16." << std::endl;
        // Fallback Attempt: Clear the config, drop INT8, and retry
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
        config->clearFlag(nvinfer1::BuilderFlag::kINT8);
#pragma GCC diagnostic pop
        plan = std::unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
        if (!plan) {
            return CudaError(ERROR_SOURCE, "Failed to build TRT Engine even after FP16 fallback");
        }
    }

    // Save engine to disk
    std::ofstream engineFile(savePath, std::ios::binary);
    if (engineFile) {
        engineFile.write(reinterpret_cast<const char*>(plan->data()), plan->size());
        std::cout << "[TRT] Engine saved to " << savePath << std::endl;
    } else {
        std::cerr << "[TRT] Warning: Could not save engine to " << savePath << std::endl;
    }

    // Deserialize for immediate use
    return DeserializeEngine(plan->data(), plan->size());
}

CudaError TrtDetector::Detect(const BatchData& input, BatchDetections &output) {
    FrameResources& res = pool_[poolIndex_];
    poolIndex_ = (poolIndex_ + 1) % POOL_SIZE;

    // Wait for Input
    if (input.readyEvent) {
        CUDA_TRY(cudaStreamWaitEvent(*cuda_stream_, *input.readyEvent, 0));
    }

    int validBatchSize = input.batchSize;
    // Derive strict engine batch size
    int engineBatchSize = input.deviceData.size() / (input.width * input.height * 3);
    if (engineBatchSize <= 0) {
        engineBatchSize = validBatchSize;
    }

    int channels = outputDims_.d[1]; // 84
    int anchors = outputDims_.d[2];  // 8400

    // Resize Raw Output to Engine Batch Size
    CUDA_TRY(res.rawOutput.resize(engineBatchSize * channels * anchors, *cuda_stream_));

    // Limit Candidates to Valid Batch Size
    size_t candSize = validBatchSize * BatchDetections::MAX_DETECTIONS_PER_FRAME * 10;
    CUDA_TRY(res.candidates.resize(candSize, *cuda_stream_));

    CUDA_TRY(res.candidateCount.resize(1, *cuda_stream_));
    CUDA_TRY(res.candidateCount.fill(0, *cuda_stream_));

    // Resize Output Structure (validBatchSize)
    size_t resultSize = validBatchSize * BatchDetections::MAX_DETECTIONS_PER_FRAME;
    CUDA_TRY(output.data.resize(resultSize, *cuda_stream_));

    CUDA_TRY(output.counts.resize(validBatchSize, *cuda_stream_));
    CUDA_TRY(output.counts.fill(0, *cuda_stream_));

    if (!output.readyEvent) {
        CUDA_TRY(CudaEvent::Create(output.readyEvent));
    }

    // Inference
    // Set the memory address for each tensor by name
    context_->setTensorAddress(inputName_.c_str(), const_cast<float*>(input.deviceData.data()));
    context_->setTensorAddress(outputName_.c_str(), res.rawOutput.data());

    // Bind dynamic dimensions using the strict derived batch size
    if (inputDims_.d[0] == -1) {
        context_->setInputShape(inputName_.c_str(),
                                nvinfer1::Dims4{engineBatchSize, 3, (int)input.height, (int)input.width});
    }

    // Execute (enqueueV3 is the new standard)
    if (!context_->enqueueV3(*cuda_stream_)) {
        return CudaError(ERROR_SOURCE, "TRT Inference Failed (enqueueV3 returned false)");
    }

    // Pass validBatchSize to the kernels to ignore padded data
    CUDA_TRY(DecodeAndFilter(
        res.rawOutput,
        res.candidates,
        res.candidateCount,
        validBatchSize,
        anchors,
        channels - 4,
        0.2f, // Confidence Threshold
        *cuda_stream_));

    // NMS (Kernel)
    CUDA_TRY(RunNMS(
        res.candidates,
        res.candidateCount,
        output.data,
        output.counts,
        res.nmsMask,
        0.45f, // IOU threshold
        BatchDetections::MAX_DETECTIONS_PER_FRAME, // Stride
        validBatchSize,
        *cuda_stream_));

    // Finalize
    CUDA_TRY(cudaEventRecord(*output.readyEvent, *cuda_stream_));

    return CudaError();
}

} // namespace cropandweed
