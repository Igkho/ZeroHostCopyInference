#pragma once
#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include "StubDetector.h"
#include "BatchData.h"
#include "BatchDetections.h"
#include "helpers.h"

namespace cropandweed {

// Define helper macro for this test suite
#ifndef ASSERT_CUDA_SUCCESS
#define ASSERT_CUDA_SUCCESS(err) ASSERT_FALSE(CudaError::IsFailure(err)) << (err).Text()
#endif

class StubDetectorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Optional: Reset device or verify state
    }

    void TearDown() override {
        // Optional: Cleanup
    }
};

// --- 1. Creation & Lifecycle ---

TEST_F(StubDetectorTest, CreateUniquePtr) {
    std::unique_ptr<IDetector> detector;
    ASSERT_CUDA_SUCCESS(StubDetector::Create(detector));
    ASSERT_NE(detector, nullptr);
    
    // Check metadata
    auto size = detector->GetInputSize();
    EXPECT_EQ(size.first, 1024);
    EXPECT_EQ(size.second, 1024);
}

// --- 2. Detection Flow ---

TEST_F(StubDetectorTest, BasicDetection) {
    // 1. Setup Detector
    std::unique_ptr<IDetector> detector;
    ASSERT_CUDA_SUCCESS(StubDetector::Create(detector));

    // 2. Setup Input (Batch of 2 images)
    std::shared_ptr<BatchData> input;
    // Create BatchData: ID=1, Batch=2, W=1024, H=1024
    ASSERT_CUDA_SUCCESS(BatchData::Create(input, 1, 2, 1024, 1024));

    // 3. Prepare Output
    BatchDetections output;
    
    // 4. Run Detect
    ASSERT_CUDA_SUCCESS(detector->Detect(*input, output));

    // 5. Verification
    // Counts buffer should handle 'batchSize' integers
    EXPECT_EQ(output.counts.size(), 2);
    
    // Data buffer should handle batchSize * MAX_DETECTIONS * sizeof(DetectionRaw)
    size_t expectedDataBytes = 2 * BatchDetections::MAX_DETECTIONS_PER_FRAME * sizeof(DetectionRaw);
    EXPECT_EQ(output.data.size(), expectedDataBytes);

    // Verify Event Synchronization
    // The output should have a valid event recorded.
    // We access the CudaEvent wrapper directly.
    ASSERT_NE(output.readyEvent, nullptr);
    
    // Use the implicit cast operator to get the raw handle for the query
    cudaError_t eventStatus = cudaEventQuery(*output.readyEvent);
    
    // It should be Success (done) or NotReady (pending), but valid.
    EXPECT_TRUE(eventStatus == cudaSuccess || eventStatus == cudaErrorNotReady);
}

// --- 3. Pool Mechanics & Memory Reuse ---

TEST_F(StubDetectorTest, PoolCycling) {
    std::unique_ptr<IDetector> detector;
    ASSERT_CUDA_SUCCESS(StubDetector::Create(detector));

    std::shared_ptr<BatchData> input;
    ASSERT_CUDA_SUCCESS(BatchData::Create(input, 1, 1, 1024, 1024));

    // The StubDetector has a pool size of 4.
    // Run 5 times to force a wrap-around and reuse of the first resource.
    for (int i = 0; i < 5; ++i) {
        BatchDetections output;
        ASSERT_CUDA_SUCCESS(detector->Detect(*input, output));
        
        EXPECT_EQ(output.counts.size(), 1);
        ASSERT_NE(output.readyEvent, nullptr);
    }
}

// --- 4. Edge Cases ---

TEST_F(StubDetectorTest, HandleZeroBatchSize) {
    std::unique_ptr<IDetector> detector;
    ASSERT_CUDA_SUCCESS(StubDetector::Create(detector));

    std::shared_ptr<BatchData> input;
    // Create input with BatchSize = 0
    ASSERT_CUDA_SUCCESS(BatchData::Create(input, 1, 0, 1024, 1024));

    BatchDetections output;
    ASSERT_CUDA_SUCCESS(detector->Detect(*input, output));

    // Should have resized to 0 without error
    EXPECT_EQ(output.counts.size(), 0);
    EXPECT_EQ(output.data.size(), 0);
}

TEST_F(StubDetectorTest, HandleLargeBatchReallocation) {
    std::unique_ptr<IDetector> detector;
    ASSERT_CUDA_SUCCESS(StubDetector::Create(detector));

    // StubDetector pre-allocates for Batch Size 8 in Init().
    // We request Batch Size 10 to trigger internal reallocation logic.
    std::shared_ptr<BatchData> input;
    ASSERT_CUDA_SUCCESS(BatchData::Create(input, 1, 10, 1024, 1024));

    BatchDetections output;
    ASSERT_CUDA_SUCCESS(detector->Detect(*input, output));

    EXPECT_EQ(output.counts.size(), 10);
    size_t expectedDataBytes = 10 * BatchDetections::MAX_DETECTIONS_PER_FRAME * sizeof(DetectionRaw);
    EXPECT_EQ(output.data.size(), expectedDataBytes);
}

TEST_F(StubDetectorTest, InputEventWait) {
    // Verify that the detector correctly accepts an input with a recorded event
    std::unique_ptr<IDetector> detector;
    ASSERT_CUDA_SUCCESS(StubDetector::Create(detector));

    std::shared_ptr<BatchData> input;
    ASSERT_CUDA_SUCCESS(BatchData::Create(input, 1, 1, 1024, 1024));

    // --- REPLACEMENT: Use CudaStream wrapper instead of raw API ---
    std::unique_ptr<CudaStream> stream;
    ASSERT_CUDA_SUCCESS(CudaStream::Create(stream));
    
    // Manually record the input event (simulate previous stage completion).
    // *input->readyEvent dereferences unique_ptr<CudaEvent>, giving CudaEvent&.
    // *stream dereferences unique_ptr<CudaStream>, giving CudaStream&.
    // Implicit cast operators in helpers.h handle the conversion to cudaEvent_t/cudaStream_t.
    cudaEventRecord(*input->readyEvent, *stream);

    BatchDetections output;
    ASSERT_CUDA_SUCCESS(detector->Detect(*input, output));
    
    // Stream is destroyed automatically by unique_ptr here.
    // No manual cudaStreamDestroy needed.
    SUCCEED();
}

} // namespace cropandweed
