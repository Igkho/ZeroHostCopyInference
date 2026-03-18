#pragma once
#include <gtest/gtest.h>
#include <filesystem>
#include <vector>
#include "OnnxDetector.h"
#include "BatchData.h"
#include "BatchDetections.h"
#include "helpers.h"

namespace cropandweed {

namespace fs = std::filesystem;

#ifndef ASSERT_CUDA_SUCCESS
#define ASSERT_CUDA_SUCCESS(err) ASSERT_FALSE(CudaError::IsFailure(err)) << (err).Text()
#endif

class OnnxDetectorTest : public ::testing::Test {
protected:
    fs::path model_path_;
    bool model_available_ = false;

    void SetUp() override {
        cudaGetLastError();
        std::vector<fs::path> candidates = {
            "test_model.onnx",
            "models/test_model.onnx",
            "../models/test_model.onnx",
            "../../models/test_model.onnx"
        };

        for (const auto& p : candidates) {
            if (fs::exists(p)) {
                model_path_ = fs::absolute(p);
                model_available_ = true;
                break;
            }
        }
    }

    void TearDown() override {
        cudaGetLastError();
    }
};

TEST_F(OnnxDetectorTest, InitFailsOnMissingFile) {
    std::unique_ptr<IDetector> detector;
    CudaError err = OnnxDetector::Create(detector, "non_existent_model.onnx");
    EXPECT_TRUE(CudaError::IsFailure(err));
    EXPECT_NE(err.Text().find("ONNX"), std::string::npos);
}

TEST_F(OnnxDetectorTest, CreateSuccess) {
    if (!model_available_) GTEST_SKIP() << "Skipping: 'test_model.onnx' not found.";

    std::unique_ptr<IDetector> detector;
    ASSERT_CUDA_SUCCESS(OnnxDetector::Create(detector, model_path_.string()));
    ASSERT_NE(detector, nullptr);

    ModelProperties props = detector->GetModelProperties();
    EXPECT_GT(props.inputWidth, 0);
    EXPECT_GT(props.inputHeight, 0);
    EXPECT_GT(props.numClasses, 0);
}

TEST_F(OnnxDetectorTest, DetectRunsEndToEnd) {
    if (!model_available_) GTEST_SKIP() << "Skipping: 'test_model.onnx' not found.";

    std::unique_ptr<IDetector> detector;
    ASSERT_CUDA_SUCCESS(OnnxDetector::Create(detector, model_path_.string()));
    ModelProperties props = detector->GetModelProperties();

    int batchSize = 1;
    std::shared_ptr<BatchData> input;
    ASSERT_CUDA_SUCCESS(BatchData::Create(input, 0, batchSize, props.inputWidth, props.inputHeight));

    // [FIX] Use ASSERT_EQ instead of CUDA_TRY for raw CUDA calls in void function
    ASSERT_CUDA_SUCCESS(input->deviceData.fill(0, 0));

    if (input->readyEvent) cudaEventRecord(*input->readyEvent, 0);

    BatchDetections output;
    ASSERT_CUDA_SUCCESS(detector->Detect(*input, output));

    EXPECT_EQ(output.counts.size(), batchSize);
    size_t expectedSize = batchSize * BatchDetections::MAX_DETECTIONS_PER_FRAME;
    EXPECT_EQ(output.data.size(), expectedSize);

    ASSERT_NE(output.readyEvent, nullptr);
    cudaEventSynchronize(*output.readyEvent);
}

TEST_F(OnnxDetectorTest, HandlesBatchSizeChanges) {
    if (!model_available_) GTEST_SKIP() << "Skipping: 'test_model.onnx' not found.";

    std::unique_ptr<IDetector> detector;
    ASSERT_CUDA_SUCCESS(OnnxDetector::Create(detector, model_path_.string()));
    ModelProperties props = detector->GetModelProperties();

    // Batch 1
    {
        std::shared_ptr<BatchData> input;
        ASSERT_CUDA_SUCCESS(BatchData::Create(input, 0, 1, props.inputWidth, props.inputHeight));
        ASSERT_CUDA_SUCCESS(input->deviceData.fill(0, 0));

        if (input->readyEvent) cudaEventRecord(*input->readyEvent, 0);

        BatchDetections output;
        ASSERT_CUDA_SUCCESS(detector->Detect(*input, output));
        EXPECT_EQ(output.counts.size(), 1);
    }

    // Batch 4
    {
        std::shared_ptr<BatchData> input;
        ASSERT_CUDA_SUCCESS(BatchData::Create(input, 1, 4, props.inputWidth, props.inputHeight));
        ASSERT_CUDA_SUCCESS(input->deviceData.fill(0, 0));

        if (input->readyEvent) cudaEventRecord(*input->readyEvent, 0);

        BatchDetections output;
        ASSERT_CUDA_SUCCESS(detector->Detect(*input, output));
        EXPECT_EQ(output.counts.size(), 4);
    }
}

} // namespace cropandweed
