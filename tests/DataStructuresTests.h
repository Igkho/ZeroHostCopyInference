#pragma once
#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include "BatchData.h"
#include "BatchDetections.h"
#include "FrameResources.h"
#include "helpers.h"

namespace cropandweed {

#ifndef ASSERT_CUDA_SUCCESS
#define ASSERT_CUDA_SUCCESS(err) ASSERT_FALSE(CudaError::IsFailure(err)) << (err).Text()
#endif

class DataStructuresTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaGetLastError();
    }
    void TearDown() override {
        cudaGetLastError();
    }
};

// ==========================================
// 1. BatchData Tests
// ==========================================

TEST_F(DataStructuresTest, BatchData_CreateAndInit) {
    std::shared_ptr<BatchData> batch;
    int batchId = 5;
    size_t bSize = 4;
    size_t w = 100;
    size_t h = 100;

    ASSERT_CUDA_SUCCESS(BatchData::Create(batch, batchId, bSize, w, h));
    ASSERT_NE(batch, nullptr);

    EXPECT_EQ(batch->batchId, batchId);
    EXPECT_EQ(batch->batchSize, bSize);
    EXPECT_EQ(batch->width, w);
    EXPECT_EQ(batch->height, h);

    size_t expectedElements = bSize * w * h * 3;
    EXPECT_EQ(batch->deviceData.size(), expectedElements);
    EXPECT_GT(batch->deviceData.capacity(), 0);

    ASSERT_NE(batch->readyEvent, nullptr);
    EXPECT_EQ(cudaEventQuery(*batch->readyEvent), cudaSuccess);
    EXPECT_EQ(batch->sourceIdentifiers.size(), bSize);
}

TEST_F(DataStructuresTest, BatchData_MoveSemantics) {
    std::shared_ptr<BatchData> batch;
    ASSERT_CUDA_SUCCESS(BatchData::Create(batch, 1, 2, 64, 64));

    float* ptr = batch->deviceData.data();
    ASSERT_NE(ptr, nullptr);

    std::shared_ptr<BatchData> batch2 = std::move(batch);

    EXPECT_EQ(batch, nullptr);
    ASSERT_NE(batch2, nullptr);
    EXPECT_EQ(batch2->deviceData.data(), ptr); // Pointer address must be preserved
    EXPECT_EQ(batch2->width, 64);
}

// ==========================================
// 2. BatchDetections Tests
// ==========================================

TEST_F(DataStructuresTest, BatchDetections_CreateAndInit) {
    std::shared_ptr<BatchDetections> dets;
    size_t bSize = 8;

    ASSERT_CUDA_SUCCESS(BatchDetections::Create(dets, bSize));
    ASSERT_NE(dets, nullptr);

    size_t expectedElements = bSize * BatchDetections::MAX_DETECTIONS_PER_FRAME;
    EXPECT_EQ(dets->data.size(), expectedElements);
    EXPECT_EQ(dets->counts.size(), bSize);

    ASSERT_NE(dets->readyEvent, nullptr);
    EXPECT_EQ(cudaEventQuery(*dets->readyEvent), cudaSuccess);
}

TEST_F(DataStructuresTest, BatchDetections_TypedAccess) {
    std::shared_ptr<BatchDetections> dets;
    ASSERT_CUDA_SUCCESS(BatchDetections::Create(dets, 1));

    DetectionRaw* ptr = dets->data.data();
    ASSERT_NE(ptr, nullptr);

    size_t expectedBytes = 1 * BatchDetections::MAX_DETECTIONS_PER_FRAME * sizeof(DetectionRaw);
    EXPECT_EQ(dets->data.byte_size(), expectedBytes);
}

// [NEW] Added Move Semantics Test for Equality
TEST_F(DataStructuresTest, BatchDetections_MoveSemantics) {
    std::shared_ptr<BatchDetections> dets;
    ASSERT_CUDA_SUCCESS(BatchDetections::Create(dets, 1));

    DetectionRaw* ptr = dets->data.data();
    int* countsPtr = dets->counts.data();

    // Move
    std::shared_ptr<BatchDetections> dets2 = std::move(dets);

    EXPECT_EQ(dets, nullptr);
    ASSERT_NE(dets2, nullptr);
    // Verify pointers explicitly to ensure GPU memory wasn't freed/realloc'd
    EXPECT_EQ(dets2->data.data(), ptr);
    EXPECT_EQ(dets2->counts.data(), countsPtr);
    ASSERT_NE(dets2->readyEvent, nullptr);
}

// ==========================================
// 3. FrameResources Tests
// ==========================================

TEST_F(DataStructuresTest, FrameResources_CreateAndInit) {
    std::shared_ptr<FrameResources> res;
    int w = 512;
    int h = 512;

    ASSERT_CUDA_SUCCESS(FrameResources::Create(res, w, h));
    ASSERT_NE(res, nullptr);

    EXPECT_EQ(res->rawOutput.size(), w * h);
    EXPECT_EQ(res->candidates.size(), 1000);
    EXPECT_EQ(res->candidateCount.size(), 1);
    EXPECT_EQ(res->nmsMask.size(), 1);

    ASSERT_NE(res->readyEvent, nullptr);
    EXPECT_EQ(cudaEventQuery(*res->readyEvent), cudaSuccess);
}

// [NEW] Added Move Semantics Test for Equality
TEST_F(DataStructuresTest, FrameResources_MoveSemantics) {
    std::shared_ptr<FrameResources> res;
    ASSERT_CUDA_SUCCESS(FrameResources::Create(res, 256, 256));

    float* rawPtr = res->rawOutput.data();
    DetectionRaw* candPtr = res->candidates.data();

    // Move
    std::shared_ptr<FrameResources> res2 = std::move(res);

    EXPECT_EQ(res, nullptr);
    ASSERT_NE(res2, nullptr);
    // Verify resource persistence
    EXPECT_EQ(res2->rawOutput.data(), rawPtr);
    EXPECT_EQ(res2->candidates.data(), candPtr);
    ASSERT_NE(res2->readyEvent, nullptr);
}

} // namespace cropandweed
