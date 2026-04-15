#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <filesystem>

#include "CLI/CLI.hpp"

#include "helpers.h"
#include "Block.h"
#include "InferencePipeline.h"
#include "FFmpegSource.h"

#if defined(PLATFORM_JETSON) && !defined(USE_JETSON_CUDA_JPEG)
// Jetson MMAPI Hardware Default
#include "MMAPIJpegSource.h"
#include "MMAPIJpegSink.h"
#else
// PC Platform OR Jetson CUDA Fallback
#include "NVJpegSource.h"
#include "NVJpegSink.h"
#endif

#include "Interfaces.h"
#include "PerformanceTimer.h"
#include "BatchData.h"
#include "StubDetector.h"
#include "OnnxDetector.h"
#include "TrtDetector.h"
#include "ClassNamesReader.h"

namespace fs = std::filesystem;
using namespace cropandweed;

enum class BackendType {
    TRT,
    ONNX,
    STUB,
};

constexpr const char* TOOL_VERSION = "1.0.1";

int main(int argc, char** argv) {
    CLI::App app{"ZeroHostCopy Inference Tool"};

    app.set_version_flag("--version", std::string(TOOL_VERSION));

    std::string inputPath;
    app.add_option("-i,--input", inputPath, "Path to input video file or image directory")
        ->required()
        ->check(CLI::ExistingPath);

    std::string modelPath;
    app.add_option("-m,--model", modelPath, "Path to ONNX model file")
//        ->required()
        ->check(CLI::ExistingFile);

    std::string outputPath;
    app.add_option("-o,--output", outputPath, "Path to output folder")
        ->required();

    std::string classesPath;
    app.add_option("-c,--classes", classesPath, "Class names file")->check(CLI::ExistingFile);

    BackendType backend = BackendType::TRT;
    std::map<std::string, BackendType> backendMap{
        {"trt",  BackendType::TRT},
        {"onnx", BackendType::ONNX},
        {"stub", BackendType::STUB},
    };
    app.add_option("--backend", backend, "Inference backend engine: trt, onnx, stub")
        ->transform(CLI::CheckedTransformer(backendMap, CLI::ignore_case));

    int batchSize = 16;
    app.add_option("-b,--batch", batchSize, "Inference batch size")
        ->default_val(16)->check(CLI::Range((int)BatchData::MIN_BATCH_SIZE,
                                            (int)BatchData::MAX_BATCH_SIZE));

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
        return app.exit(e);
    }

    std::cout << "Starting ZeroHostCopy Inference Tool v" << TOOL_VERSION << std::endl;

    try {
        // Measure initialization time
        PerformanceTimer initTimer("Initialization");
        
        // Create Detector based on Backend Flag
        std::unique_ptr<IDetector> detector;

        switch (backend) {
            case BackendType::STUB:
                std::cout << "[Main] Selected Backend: Stub (Pass-through)" << std::endl;
                CUDA_CALL(StubDetector::Create(detector));
                break;

            case BackendType::TRT:
                if (modelPath.empty()) {
                    throw std::runtime_error("TensorRT backend requires --model argument");
                }
                std::cout << "[Main] Selected Backend: TensorRT" << std::endl;
                CUDA_CALL(TrtDetector::Create(detector, modelPath));
                break;

            case BackendType::ONNX:
#ifdef PLATFORM_JETSON
                // Explicitly block ONNX on Jetson with a clean, helpful error
                throw std::runtime_error("[Fatal Error] The ONNX Runtime GPU backend is not supported on Jetson architecture. "
                "Please use the native TensorRT backend: '--backend trt'"
                );
#else
                // PC Execution path
                if (modelPath.empty()) {
                    throw std::runtime_error("ONNX backend requires --model argument");
                }
                std::cout << "[Main] Selected Backend: ONNX Runtime" << std::endl;
                CUDA_CALL(OnnxDetector::Create(detector, modelPath));
#endif
                break;

            default:
                throw std::runtime_error("Invalid Backend Type");
        }

        // Extract Properties
        ModelProperties props = detector->GetModelProperties();

        // Load Class Names from file if provided, otherwise generate generic
        if (!classesPath.empty()) {
            std::cout << "[Main] Loading classes from " << classesPath << "..." << std::endl;
            auto nameMap = ClassNamesReader::Read(classesPath);

            // Convert to vector for the Sink
            props.classNames = ClassNamesReader::ToVector(nameMap);

            // Validate count match
            if (props.numClasses != props.classNames.size()) {
                std::cerr << "[Warning] Model predicts " << props.numClasses
                          << " classes, but file provided " << props.classNames.size() << std::endl;
            }
        } else {
            // Generate generic names "Class 0", "Class 1"...
            for(int i=0; i<props.numClasses; ++i) {
                props.classNames.push_back("Class " + std::to_string(i));
            }
        }

        // Create Source
        std::unique_ptr<ISource> source;
        if (fs::is_directory(inputPath)) {
#if defined(PLATFORM_JETSON) && !defined(USE_JETSON_CUDA_JPEG)
            std::cout << "[Main] Input is a directory. Initializing MMAPIJpegSource..." << std::endl;
            CUDA_CALL(MMAPIJpegSource::Create(source, inputPath, props.inputWidth,
                                              props.inputHeight, batchSize));
#else
            std::cout << "[Main] Input is a directory. Initializing NVJpegSource..." << std::endl;
            CUDA_CALL(NVJpegSource::Create(source, inputPath, props.inputWidth, props.inputHeight));
#endif
        } else if (fs::is_regular_file(inputPath)) {
            // Hardware-aware routing: Prevent Jetson from attempting unsupported NVDEC video decoding
#ifdef PLATFORM_JETSON
            throw std::runtime_error(
                "Video decoding via FFmpeg is not supported on Jetson UMA architecture yet. "
                "Please extract your video to a folder of separate JPEG frames and pass the folder path as --input."
                );
#else
            std::cout << "[Main] Input is a file. Initializing FFmpegSource..." << std::endl;
            CUDA_CALL(FFmpegSource::Create(source, inputPath, props.inputWidth, props.inputHeight));
#endif
        } else {
            throw std::runtime_error("Input path is neither a regular file nor a directory.");
        }

        // Create Sink
        std::unique_ptr<ISink> sink;
#if defined(PLATFORM_JETSON) && !defined(USE_JETSON_CUDA_JPEG)
        CUDA_CALL(MMAPIJpegSink::Create(sink, outputPath, props, batchSize));
#else
        CUDA_CALL(NVJpegSink::Create(sink, outputPath, props));
#endif
        // Create and Run Pipeline
        InferencePipeline pipeline(
            std::move(source),
            std::move(detector),
            std::move(sink),
            batchSize
            );

        // Stop init timer manually to get the value for the report
        long long initMs = initTimer.Stop();

        //Run
        CUDA_CALL(pipeline.Run());

        // Print Report
        pipeline.PrintStats(initMs);

    } catch (const std::exception& e) {
        std::cerr << "CRITICAL ERROR: " << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "UNKNOWN PROPRIETARY EXCEPTION THROWN!" << std::endl;
    }

    return 0;
}
