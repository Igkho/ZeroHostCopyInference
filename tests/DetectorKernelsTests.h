#pragma once
#include <gtest/gtest.h>
#include <vector>
#include <algorithm>
#include "DetectorKernels.h"
#include "DetectionRaw.h"
#include "Block.h"
#include "helpers.h"

namespace cropandweed {

#ifndef ASSERT_CUDA_SUCCESS
#define ASSERT_CUDA_SUCCESS(err) ASSERT_FALSE(CudaError::IsFailure(err)) << (err).Text()
#endif

class DetectorKernelsTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// ==========================================
// 1. Tensor Decoding Tests
// ==========================================

TEST_F(DetectorKernelsTest, DecodeAndFilter_ThresholdsAndLayout) {
    int batchSize = 1;
    int numAnchors = 2;
    int numClasses = 1;
    int channels = 4 + numClasses; // 5

    Block<float> d_output;
    size_t totalElements = batchSize * channels * numAnchors;
    ASSERT_CUDA_SUCCESS(d_output.resize(totalElements));

    std::vector<float> h_output(totalElements, 0.0f);

    // Anchor 0: Valid Detection
    h_output[0 * numAnchors + 0] = 0.5f; // x
    h_output[1 * numAnchors + 0] = 0.5f; // y
    h_output[2 * numAnchors + 0] = 0.2f; // w
    h_output[3 * numAnchors + 0] = 0.2f; // h
    h_output[4 * numAnchors + 0] = 0.9f; // score (Class 0)

    // Anchor 1: Noise (Low Score)
    h_output[4 * numAnchors + 1] = 0.1f; // score (Class 0)

    ASSERT_CUDA_SUCCESS(d_output.assign(h_output));

    TypedBlock<DetectionRaw> d_candidates;
    Block<int> d_count;

    ASSERT_CUDA_SUCCESS(d_candidates.resize(10));
    ASSERT_CUDA_SUCCESS(d_count.resize(1));

    ASSERT_CUDA_SUCCESS(DecodeAndFilter(
        d_output.data(),
        d_candidates.data(),
        10,
        d_count.data(),
        batchSize,
        numAnchors,
        numClasses,
        0.5f,
        0
        ));

    std::vector<int> h_count;
    ASSERT_CUDA_SUCCESS(d_count.to_vector(h_count));
    EXPECT_EQ(h_count[0], 1);

    std::vector<DetectionRaw> h_candidates;
    ASSERT_CUDA_SUCCESS(d_candidates.to_vector(h_candidates));

    EXPECT_NEAR(h_candidates[0].x, 0.5f, 1e-4);
    EXPECT_NEAR(h_candidates[0].score, 0.9f, 1e-4);
}

// ==========================================
// 2. NMS (Non-Maximum Suppression) Tests
// ==========================================

TEST_F(DetectorKernelsTest, RunNMS_SuppressesOverlap) {
    std::vector<DetectionRaw> candidates = {
        {100.f, 100.f, 50.f, 50.f, 0.9f, 0.f, 0.f, 0.f}, // A
        {105.f, 105.f, 50.f, 50.f, 0.8f, 0.f, 0.f, 0.f}, // B (overlap A)
        {300.f, 300.f, 50.f, 50.f, 0.7f, 0.f, 0.f, 0.f}  // C
    };

    int numCandidates = (int)candidates.size();

    TypedBlock<DetectionRaw> d_candidates;
    Block<int> d_candCount;

    ASSERT_CUDA_SUCCESS(d_candidates.assign(candidates));
    ASSERT_CUDA_SUCCESS(d_candCount.assign({numCandidates}));

    TypedBlock<DetectionRaw> d_final;
    Block<int> d_finalCount;
    Block<uint8_t> d_mask;

    ASSERT_CUDA_SUCCESS(d_final.resize(10));
    ASSERT_CUDA_SUCCESS(d_finalCount.resize(1));

    ASSERT_CUDA_SUCCESS(RunNMS(
        d_candidates.data(),
        numCandidates,
        d_candCount.data(),
        d_final.data(),
        10,
        d_finalCount.data(),
        d_mask,
        0.45f,
        10,
        1,
        0
        ));

    std::vector<int> h_finalCount;
    ASSERT_CUDA_SUCCESS(d_finalCount.to_vector(h_finalCount));
    ASSERT_EQ(h_finalCount[0], 2);

    std::vector<DetectionRaw> h_final;
    ASSERT_CUDA_SUCCESS(d_final.to_vector(h_final));

    // NMS sorts by score
    EXPECT_NEAR(h_final[0].score, 0.9f, 1e-4);
    EXPECT_NEAR(h_final[1].score, 0.7f, 1e-4);

    bool has_score_08 = false;
    for(int i=0; i<2; ++i) if(fabs(h_final[i].score - 0.8f) < 1e-4) has_score_08 = true;
    EXPECT_FALSE(has_score_08);
}

} // namespace cropandweed
