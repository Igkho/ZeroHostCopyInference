#pragma once
#include <gtest/gtest.h>
#include <vector>
#include "ObjectTracker.h"
#include "ObjectTrackerKernels.h"
#include "Block.h"
#include "helpers.h"

namespace cropandweed {

#ifndef ASSERT_CUDA_SUCCESS
#define ASSERT_CUDA_SUCCESS(err) ASSERT_FALSE(CudaError::IsFailure(err)) << (err).Text()
#endif

// ==========================================
// 1. Test Fixture & Helpers
// ==========================================

class ObjectTrackerTest : public ::testing::Test {
protected:

    // Helper context for White-Box Kernel Testing
    // This allows us to manually manipulate GPU buffers that are usually private to the class.
    struct TrackerContext {
        TypedBlock<TrackState> tracks;
        Block<int> trackCount;
        Block<int> nextTrackId;
        Block<int> matches;

        int maxTracks = 100;

        void Init() {
            ASSERT_CUDA_SUCCESS(tracks.resize(maxTracks));
            // Initialize with 0 tracks
            ASSERT_CUDA_SUCCESS(trackCount.assign({0}));
            ASSERT_CUDA_SUCCESS(nextTrackId.assign({1}));
            ASSERT_CUDA_SUCCESS(matches.resize(100));
        }
    };
};

// ==========================================
// 2. Kernel Logic Tests (White Box)
// ==========================================
// These tests verify the math/logic in ObjectTrackerKernels.cu independently of the class.

TEST_F(ObjectTrackerTest, CreatesNewTrack) {
    TrackerContext ctx;
    ctx.Init();

    // 1. Prepare Input: 1 Detection at (100, 100)
    std::vector<DetectionRaw> dets = {
        {100.f, 100.f, 50.f, 50.f, 0.9f, 0.f, 0.f, 0.f}
    };

    BoundaryTypedBlock<DetectionRaw> d_dets;
    BoundaryBlock<int> d_detCount;

    ASSERT_CUDA_SUCCESS(d_dets.assign(dets));
    ASSERT_CUDA_SUCCESS(d_detCount.assign({(int)dets.size()}));

    // 2. Run Kernel Wrapper Directly
    ASSERT_CUDA_SUCCESS(TrackBatch(
        0, // Batch Index
        d_dets,
        d_detCount,
        ctx.tracks,
        ctx.trackCount,
        ctx.nextTrackId,
        ctx.matches,
        100, // Stride
        ctx.maxTracks,
        1,   // Active Classes
        0.1f,// Alpha
        1024, 1024, // Image Dims
        0    // Stream
        ));

    // 3. Verify Track Count
    std::vector<int> h_trackCount;
    ASSERT_CUDA_SUCCESS(ctx.trackCount.to_vector(h_trackCount));
    EXPECT_EQ(h_trackCount[0], 1) << "Should create 1 new track";

    // 4. Verify Track State
    std::vector<TrackState> h_tracks;
    ASSERT_CUDA_SUCCESS(ctx.tracks.to_vector(h_tracks));

    EXPECT_EQ(h_tracks[0].id, 1);
    EXPECT_EQ(h_tracks[0].age, 1);
    EXPECT_NEAR(h_tracks[0].x, 100.f, 0.1f);
}

TEST_F(ObjectTrackerTest, UpdatesExistingTrack) {
    TrackerContext ctx;
    ctx.Init();

    // 1. Manually Seed an existing track
    // [Fix] Zero-initialize to ensure timeSinceUpdate is 0 (prevents ghosting logic issues)
    TrackState seed = {};
    seed.id = 5;
    seed.age = 10;
    seed.x = 100.f; seed.y = 100.f;
    seed.w = 50.f; seed.h = 50.f;
    seed.vx = 0; seed.vy = 0;

    std::vector<TrackState> seedVec(ctx.maxTracks);
    seedVec[0] = seed;

    ASSERT_CUDA_SUCCESS(ctx.tracks.assign(seedVec));
    // [Fix] Use assign to explicitly set count to 1 (resize(1,1) is no-op if size is already 1)
    ASSERT_CUDA_SUCCESS(ctx.trackCount.assign({1}));

    // 2. Input: Detection slightly moved (110, 100)
    std::vector<DetectionRaw> dets = {
        {110.f, 100.f, 50.f, 50.f, 0.9f, 0.f, 0.f, 0.f}
    };
    BoundaryTypedBlock<DetectionRaw> d_dets;
    BoundaryBlock<int> d_detCount;

    ASSERT_CUDA_SUCCESS(d_dets.assign(dets));
    ASSERT_CUDA_SUCCESS(d_detCount.assign({1}));

    // 3. Run Tracking
    ASSERT_CUDA_SUCCESS(TrackBatch(
        0, d_dets, d_detCount,
        ctx.tracks, ctx.trackCount, ctx.nextTrackId, ctx.matches,
        100, ctx.maxTracks, 1, 0.1f, 1024, 1024, 0
        ));

    // 4. Verify Updates
    std::vector<TrackState> h_tracks;
    ASSERT_CUDA_SUCCESS(ctx.tracks.to_vector(h_tracks));

    EXPECT_EQ(h_tracks[0].id, 5) << "ID should remain the same";
    EXPECT_NEAR(h_tracks[0].x, 110.f, 0.1f) << "X Position should update to detection";
    EXPECT_EQ(h_tracks[0].age, 11) << "Age should increment";
}

TEST_F(ObjectTrackerTest, GhostsMissingTrack) {
    TrackerContext ctx;
    ctx.Init();

    // 1. Seed existing track (Moving)
    // [Fix] Zero-initialize to ensure timeSinceUpdate is 0.
    // If this is garbage, GhostAndCleanupKernel might think the track is stale.
    TrackState seed = {};
    seed.id = 1;
    seed.age = 20;
    seed.x = 500.f; seed.y = 500.f;
    seed.w = 50.f; seed.h = 50.f;
    seed.vx = 5.0f; // Moving right (+5 per frame)
    seed.vy = 0.0f;

    std::vector<TrackState> seedVec(ctx.maxTracks);
    seedVec[0] = seed;

    ASSERT_CUDA_SUCCESS(ctx.tracks.assign(seedVec));
    ASSERT_CUDA_SUCCESS(ctx.trackCount.assign({1}));

    // 2. Input: NO detections (Empty batch)
    BoundaryTypedBlock<DetectionRaw> d_dets;
    ASSERT_CUDA_SUCCESS(d_dets.resize(100));
    BoundaryBlock<int> d_detCount;
    ASSERT_CUDA_SUCCESS(d_detCount.assign({0})); // 0 detections

    // 3. Run Tracking
    ASSERT_CUDA_SUCCESS(TrackBatch(
        0, d_dets, d_detCount,
        ctx.tracks, ctx.trackCount, ctx.nextTrackId, ctx.matches,
        100, ctx.maxTracks, 1, 0.1f, 1024, 1024, 0
        ));

    // 4. Verify Track State (Prediction)
    std::vector<TrackState> h_tracks;
    ASSERT_CUDA_SUCCESS(ctx.tracks.to_vector(h_tracks));

    // Should assume coasting velocity: 500 + 5 = 505
    EXPECT_NEAR(h_tracks[0].x, 505.f, 0.1f);
    EXPECT_EQ(h_tracks[0].missedFrames, 1);

    // 5. Verify Ghost Output (Result Buffer)
    // The kernel should add the predicted ghost back into the detection list for rendering.
    std::vector<int> h_detCountOut;
    ASSERT_CUDA_SUCCESS(d_detCount.to_vector(h_detCountOut));

    EXPECT_EQ(h_detCountOut[0], 1) << "Should emit 1 ghost detection";
}

// ==========================================
// 3. Class Integration Tests (Black Box)
// ==========================================
// These tests verify ObjectTracker.cpp correctly manages memory and orchestrates kernels.

TEST_F(ObjectTrackerTest, Class_EndToEndIntegration) {
    // 1. Create the high-level ObjectTracker class
    // Uses the Factory pattern defined in ObjectTracker.h
    std::unique_ptr<ObjectTracker> tracker;
    ASSERT_CUDA_SUCCESS(ObjectTracker::Create(tracker, 10, 0)); // 10 classes, default stream

    // 2. Create Input
    // Single detection at (100,100) with no Track ID assigned yet (default 0 or -1)
    std::vector<DetectionRaw> dets = {
        {100.f, 100.f, 50.f, 50.f, 0.9f, 0.f, 0.f, 0.f}
    };
    BoundaryTypedBlock<DetectionRaw> d_dets;
    ASSERT_CUDA_SUCCESS(d_dets.assign(dets));

    BoundaryBlock<int> d_counts;
    ASSERT_CUDA_SUCCESS(d_counts.assign({1}));

    // 3. Run ProcessBatch via the Class API
    // This tests if the internal buffers (tracks_, nextTrackId_) are correctly allocated and used.
    ASSERT_CUDA_SUCCESS(tracker->ProcessBatch(
        0,
        d_dets,
        d_counts,
        100, // Stride
        1024, 1024,
        0 // Stream
        ));

    // 4. Verify Output Side-Effects
    // The tracker modifies the input 'detections' buffer to attach Track IDs.
    std::vector<DetectionRaw> h_result;
    ASSERT_CUDA_SUCCESS(d_dets.to_vector(h_result));

    EXPECT_GT(h_result[0].track_id, 0.0f) << "Class should assign a valid Track ID (>=1)";
    EXPECT_EQ(h_result[0].x, 100.f);
}

} // namespace cropandweed
