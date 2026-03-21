#include <gtest/gtest.h>
#include <string>
#include <iostream>
#include <filesystem>
#include <vector>

namespace fs = std::filesystem;

namespace cropandweed {
    // Global configuration variable accessible by all test fixtures
    std::string g_test_model_path = "";
    bool g_model_available = false;
}

// --- Test Suites ---
#include "HelpersTests.h"
#include "BlockTests.h"
#include "SafeQueueTests.h"
#include "FFMpegSourceTests.h"
#include "NVJpegSourceTests.h"
#include "StubDetectorTests.h"
#include "NVJpegSinkTests.h"
#include "PerformanceTimerTests.h"
#include "InferencePipelineTests.h"
#include "DetectorKernelsTests.h"
#include "ObjectTrackerTests.h"
#include "OnnxDetectorTests.h"
#include "TrtDetectorTests.h"
#include "DataStructuresTests.h"

int main(int argc, char **argv)
{
    // 1. Let GTest consume its own arguments (e.g., --gtest_filter)
    ::testing::InitGoogleTest(&argc, argv);

    // 2. Parse remaining arguments for our custom pipeline inputs
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--model" || arg == "-m") && i + 1 < argc) {
            cropandweed::g_test_model_path = argv[i + 1];
            std::cout << "[Test Config] Using injected model path: "
                      << cropandweed::g_test_model_path << std::endl;
            break;
        }
    }

    // 3. Resolve Model Path (Injected or Fallback)
    if (!cropandweed::g_test_model_path.empty() && fs::exists(cropandweed::g_test_model_path)) {
        cropandweed::g_test_model_path = fs::absolute(cropandweed::g_test_model_path).string();
        cropandweed::g_model_available = true;
        std::cout << "[Test Config] Using injected model: " << cropandweed::g_test_model_path << std::endl;
    } else {
        std::cout << "[Test Config] No valid --model provided. Executing fallback search..." << std::endl;
        std::vector<fs::path> candidates = {
            "best.onnx", "models/best.onnx",
            "../best.onnx", "../models/best.onnx",
            "../../best.onnx", "../../models/best.onnx"
        };
        for (const auto& p : candidates) {
            if (fs::exists(p)) {
                cropandweed::g_test_model_path = fs::absolute(p).string();
                cropandweed::g_model_available = true;
                std::cout << "[Test Config] Found fallback model: " << cropandweed::g_test_model_path << std::endl;
                break;
            }
        }
    }
    if (!cropandweed::g_model_available) {
        std::cerr << "[Test Config] WARNING: No ONNX model found. Model-dependent tests will be skipped." << std::endl;
    }

    // 4. Execute all registered tests
    return RUN_ALL_TESTS();
}
