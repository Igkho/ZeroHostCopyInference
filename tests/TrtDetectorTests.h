#pragma once
#include <gtest/gtest.h>
#include <filesystem>
#include <vector>
#include <fstream>
#include "TrtDetector.h"
#include "BatchData.h"
#include "BatchDetections.h"
#include "helpers.h"

namespace cropandweed {

namespace fs = std::filesystem;

#ifndef ASSERT_CUDA_SUCCESS
#define ASSERT_CUDA_SUCCESS(err) ASSERT_FALSE(CudaError::IsFailure(err)) << (err).Text()
#endif

class TrtDetectorTest : public ::testing::Test {
protected:
    fs::path onnx_path_;
    fs::path engine_path_;
    bool onnx_available_ = false;
    fs::path temp_dir_;

    void SetUp() override {
        cudaGetLastError();
        temp_dir_ = fs::current_path() / "test_trt_cache";
        if (!fs::exists(temp_dir_)) fs::create_directory(temp_dir_);

        std::vector<fs::path> candidates = {
            "test_model.onnx",
            "models/test_model.onnx",
            "../models/test_model.onnx",
            "../../models/test_model.onnx"
        };

        for (const auto& p : candidates) {
            if (fs::exists(p)) {
                onnx_path_ = fs::absolute(p);
                onnx_available_ = true;
                break;
            }
        }
    }

    void TearDown() override {
        if (fs::exists(temp_dir_)) {
            fs::remove_all(temp_dir_);
        }
        cudaGetLastError();
    }
};

TEST_F(TrtDetectorTest, InitFailsOnMissingFile) {
    std::unique_ptr<IDetector> detector;
    CudaError err = TrtDetector::Create(detector, "non_existent.onnx");
    EXPECT_TRUE(CudaError::IsFailure(err));
}

TEST_F(TrtDetectorTest, BuildEngineFromOnnx) {
    if (!onnx_available_) GTEST_SKIP() << "Skipping: 'test_model.onnx' not found.";

    std::unique_ptr<IDetector> detector;
    fs::path temp_onnx = temp_dir_ / "build_test.onnx";
    fs::path temp_engine = temp_dir_ / "build_test.engine";

    fs::copy_file(onnx_path_, temp_onnx, fs::copy_options::overwrite_existing);

    ASSERT_CUDA_SUCCESS(TrtDetector::Create(detector, temp_onnx.string()));

    EXPECT_TRUE(fs::exists(temp_engine));

    ModelProperties props = detector->GetModelProperties();
    EXPECT_GT(props.inputWidth, 0);
    EXPECT_GT(props.inputHeight, 0);
}

TEST_F(TrtDetectorTest, LoadCachedEngine) {
    if (!onnx_available_) GTEST_SKIP() << "Skipping: 'test_model.onnx' not found.";

    fs::path temp_onnx = temp_dir_ / "cache_test.onnx";
    fs::copy_file(onnx_path_, temp_onnx, fs::copy_options::overwrite_existing);

    // Build
    {
        std::unique_ptr<IDetector> det1;
        ASSERT_CUDA_SUCCESS(TrtDetector::Create(det1, temp_onnx.string()));
    }

    // Load
    {
        std::unique_ptr<IDetector> det2;
        ASSERT_CUDA_SUCCESS(TrtDetector::Create(det2, temp_onnx.string()));
        ModelProperties props = det2->GetModelProperties();
        EXPECT_GT(props.inputWidth, 0);
    }
}

TEST_F(TrtDetectorTest, DetectRunsEndToEnd) {
    if (!onnx_available_) GTEST_SKIP() << "Skipping: 'test_model.onnx' not found.";

    fs::path temp_onnx = temp_dir_ / "infer_test.onnx";
    fs::copy_file(onnx_path_, temp_onnx, fs::copy_options::overwrite_existing);

    std::unique_ptr<IDetector> detector;
    ASSERT_CUDA_SUCCESS(TrtDetector::Create(detector, temp_onnx.string()));
    ModelProperties props = detector->GetModelProperties();

    int batchSize = 2;
    std::shared_ptr<BatchData> input;
    ASSERT_CUDA_SUCCESS(BatchData::Create(input, 0, batchSize, props.inputWidth, props.inputHeight));

    // [FIX] Use ASSERT_EQ
    ASSERT_EQ(cudaMemsetAsync(input->deviceData.data(), 0, input->deviceData.byte_size(), 0), cudaSuccess);

    if (input->readyEvent) cudaEventRecord(*input->readyEvent, 0);

    BatchDetections output;
    ASSERT_CUDA_SUCCESS(detector->Detect(*input, output));

    EXPECT_EQ(output.counts.size(), batchSize);
    EXPECT_EQ(output.data.size(), batchSize * BatchDetections::MAX_DETECTIONS_PER_FRAME);

    ASSERT_NE(output.readyEvent, nullptr);
    cudaEventSynchronize(*output.readyEvent);
}

} // namespace cropandweed
