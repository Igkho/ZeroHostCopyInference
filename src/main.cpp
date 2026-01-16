#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "CLI/CLI.hpp"

#include "helpers.h"
#include "Block.h"
#include "InferencePipeline.h"
#include "FFmpegSource.h"
#include "Interfaces.h"
#include "NVJpegSink.h"
#include "PerformanceTimer.h"
#include "DetectionRaw.h"
#include "StubDetector.h"
#include "OnnxDetector.h"
#include "TrtDetector.h"


using namespace cropandweed;

constexpr const char* TOOL_VERSION = "1.0.0";

int main(int argc, char** argv) {
    CLI::App app{"CropAndWeed Inference Tool"};

    app.set_version_flag("--version", std::string(TOOL_VERSION));

    std::string videoPath;
    app.add_option("-i,--input", videoPath, "Path to input video file")
        ->required()
        ->check(CLI::ExistingFile);

    std::string modelPath;
    app.add_option("-m,--model", modelPath, "Path to ONNX model file")
//        ->required()
        ->check(CLI::ExistingFile);

    std::string outputPath;
    app.add_option("-o,--output", outputPath, "Path to output folder") // Changed desc to 'folder'
        ->required();

    std::string backend = "stub"; // Default changed to stub for testing
    app.add_option("--backend", backend, "Inference backend engine")
        ->check(CLI::IsMember({"onnx", "trt", "stub"}, CLI::ignore_case))->default_val("stub");

    int batchSize = 4;
    app.add_option("-b,--batch-size", batchSize, "Inference batch size")
        ->default_val(4)->check(CLI::Range(1, (int)BatchData::MAX_BATCH_SIZE));

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
        return app.exit(e);
    }

    std::cout << "Starting CropAndWeed Inference Tool v" << TOOL_VERSION << std::endl;

    try {
        // Measure initialization time
        PerformanceTimer initTimer("Initialization");
        
        // Create Source
        std::unique_ptr<ISource> source;
        CUDA_CALL(FFmpegSource::Create(source, videoPath))

        // Create Detector based on Backend Flag
        std::unique_ptr<IDetector> detector;

        if (backend == "stub") {
            std::cout << "[Main] Selected Backend: Stub (Pass-through)" << std::endl;
            CUDA_CALL(StubDetector::Create(detector));
        }
        else if (backend == "trt") {
            if (modelPath.empty()) throw std::runtime_error("TensorRT backend requires --model argument");
            std::cout << "[Main] Selected Backend: TensorRT" << std::endl;
            CUDA_CALL(TrtDetector::Create(detector, modelPath));
        } else {
            if (modelPath.empty()) throw std::runtime_error("ONNX backend requires --model argument");
            std::cout << "[Main] Selected Backend: ONNX Runtime" << std::endl;
            CUDA_CALL(OnnxDetector::Create(detector, modelPath));
        }

        const auto [reqW, reqH] = detector->GetInputSize();
        std::cout << "[Main] Model requires input: " << reqW << "x" << reqH << std::endl;
        source->SetOutputSize(reqW, reqH);

        // Create Sink
        std::unique_ptr<ISink> sink;
        CUDA_CALL(NVJpegSink::Create(sink, outputPath));

        // Create and Run Pipeline
        InferencePipeline pipeline(
            std::move(source),
            std::move(detector),
            std::move(sink),
            batchSize
            );

        // Stop init timer manually to get the value for the report
        long long initMs = initTimer.Stop();
        std::cout << "[PERFORMANCE] Initialization completed in " << initMs << " ms" << std::endl;

        //Run
        CUDA_CALL(pipeline.Run());

        // Print Report
        pipeline.PrintStats(initMs);

    } catch (const std::exception& e) {
        std::cerr << "CRITICAL ERROR: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
