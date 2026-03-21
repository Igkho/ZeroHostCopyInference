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

// Declare the global variables from main.cpp
extern std::string g_test_model_path;
extern bool g_model_available;

#ifndef ASSERT_CUDA_SUCCESS
#define ASSERT_CUDA_SUCCESS(err) ASSERT_FALSE(CudaError::IsFailure(err)) << (err).Text()
#endif

class TrtDetectorTest : public ::testing::Test {
protected:
    fs::path onnx_path_;
    bool onnx_available_ = false;

    void SetUp() override {
        cudaGetLastError(); // Clear sticky errors
        // Read the pre-resolved configuration from main
        onnx_path_ = g_test_model_path;
        onnx_available_ = g_model_available;
    }

    void TearDown() override {
        cudaGetLastError();
    }
};

TEST_F(TrtDetectorTest, InitFailsOnMissingFile) {
    std::unique_ptr<IDetector> detector;
    CudaError err = TrtDetector::Create(detector, "non_existent.onnx");
    EXPECT_TRUE(CudaError::IsFailure(err));
}

TEST_F(TrtDetectorTest, BuildOrLoadEngine) {
    if (!onnx_available_) GTEST_SKIP() << "Skipping: Model not found.";

    std::unique_ptr<IDetector> detector;
    // If it's the first time running the suite, this builds. Otherwise, it loads instantly.
    ASSERT_CUDA_SUCCESS(TrtDetector::Create(detector, onnx_path_.string()));

    fs::path expected_engine = onnx_path_;
    expected_engine.replace_extension(".engine");
    EXPECT_TRUE(fs::exists(expected_engine));

    ModelProperties props = detector->GetModelProperties();
    EXPECT_GT(props.inputWidth, 0);
    EXPECT_GT(props.inputHeight, 0);
}

TEST_F(TrtDetectorTest, DetectRunsEndToEnd) {
    if (!onnx_available_) GTEST_SKIP() << "Skipping: Model not found.";

    std::unique_ptr<IDetector> detector;
    ASSERT_CUDA_SUCCESS(TrtDetector::Create(detector, onnx_path_.string()));

    ModelProperties props = detector->GetModelProperties();
    int batchSize = BatchData::OPTIMUM_BATCH_SIZE;
    std::shared_ptr<BatchData> input;
    ASSERT_CUDA_SUCCESS(BatchData::Create(input, 0, batchSize, props.inputWidth, props.inputHeight));

    ASSERT_CUDA_SUCCESS(input->deviceData.fill(0, 0));
    if (input->readyEvent) cudaEventRecord(*input->readyEvent, 0);

    BatchDetections output;
    ASSERT_CUDA_SUCCESS(detector->Detect(*input, output));

    EXPECT_EQ(output.counts.size(), batchSize);
    EXPECT_EQ(output.data.size(), batchSize * BatchDetections::MAX_DETECTIONS_PER_FRAME);

    if (output.readyEvent) cudaEventSynchronize(*output.readyEvent);
}

// Included the partial batch test from previous reviews
TEST_F(TrtDetectorTest, HandlesPartialBatchWithStrictEngineSizing) {
    if (!onnx_available_) GTEST_SKIP() << "Skipping: Model not found.";

    std::unique_ptr<IDetector> detector;
    ASSERT_CUDA_SUCCESS(TrtDetector::Create(detector, onnx_path_.string()));
    ModelProperties props = detector->GetModelProperties();

    int engineBatch = BatchData::OPTIMUM_BATCH_SIZE;
    int validBatch = 2; // Simulate partial batch

    std::shared_ptr<BatchData> input;
    // Allocate for FULL engineBatch
    ASSERT_CUDA_SUCCESS(BatchData::Create(input, 0, engineBatch, props.inputWidth, props.inputHeight));
    ASSERT_CUDA_SUCCESS(input->deviceData.fill(0, 0));

    // Override the valid size
    input->batchSize = validBatch;
    if (input->readyEvent) cudaEventRecord(*input->readyEvent, 0);

    BatchDetections output;
    ASSERT_CUDA_SUCCESS(detector->Detect(*input, output));

    EXPECT_EQ(output.counts.size(), validBatch)
        << "TRT Detector must limit output counts to the valid batch size";

    size_t expectedDataElements = validBatch * BatchDetections::MAX_DETECTIONS_PER_FRAME;
    EXPECT_EQ(output.data.size(), expectedDataElements)
        << "TRT Detector must limit bounding box vectors to the valid batch size";

    if (output.readyEvent) cudaEventSynchronize(*output.readyEvent);
}

} // namespace cropandweed
