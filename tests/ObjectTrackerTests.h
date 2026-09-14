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
        std::vector<int> trackCountHost;
        Block<int> nextTrackId;
        Block<int> matches;
        std::vector<int> countBufferHost;

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
        ctx.countBufferHost,
        ctx.tracks,
        ctx.trackCount,
        ctx.trackCountHost,
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
        0, d_dets, d_detCount, ctx.countBufferHost,
        ctx.tracks, ctx.trackCount, ctx.trackCountHost, ctx.nextTrackId, ctx.matches,
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
        0, d_dets, d_detCount, ctx.countBufferHost,
        ctx.tracks, ctx.trackCount, ctx.trackCountHost, ctx.nextTrackId, ctx.matches,
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

TEST_F(ObjectTrackerTest, MultiWarpReductionAndGhosting) {
    TrackerContext ctx;
    ctx.Init();

    // 1. Generate 40 tracks (spans across 2 warps to test cross-warp reduction)
    int num_tracks = 40;
    std::vector<TrackState> seedVec(ctx.maxTracks, TrackState{});
    for(int i = 0; i < num_tracks; ++i) {
        seedVec[i].id = i + 1;
        seedVec[i].age = 20; // Mature enough to be included in the mean
        seedVec[i].x = 100.f; seedVec[i].y = 100.f;
        seedVec[i].w = 10.f;  seedVec[i].h = 10.f;
        seedVec[i].vx = 2.0f; seedVec[i].vy = -1.5f;
    }

    ASSERT_CUDA_SUCCESS(ctx.tracks.assign(seedVec));
    ASSERT_CUDA_SUCCESS(ctx.trackCount.assign({num_tracks}));

    // 2. Input: NO real detections (Empty batch)
    BoundaryTypedBlock<DetectionRaw> d_dets;
    ASSERT_CUDA_SUCCESS(d_dets.resize(100)); // Room for 40 ghosts
    BoundaryBlock<int> d_detCount;
    ASSERT_CUDA_SUCCESS(d_detCount.assign({0}));

    // 3. Run Tracking
    ASSERT_CUDA_SUCCESS(TrackBatch(
        0, d_dets, d_detCount, ctx.countBufferHost,
        ctx.tracks, ctx.trackCount, ctx.trackCountHost, ctx.nextTrackId, ctx.matches,
        100, ctx.maxTracks, 1, 0.1f, 1024, 1024, 0
        ));

    // 4. Verify output (All 40 tracks should be valid and generate ghosts)
    std::vector<int> h_detCountOut;
    ASSERT_CUDA_SUCCESS(d_detCount.to_vector(h_detCountOut));
    EXPECT_EQ(h_detCountOut[0], num_tracks) << "All tracks should generate ghosts";

    std::vector<DetectionRaw> h_dets;
    ASSERT_CUDA_SUCCESS(d_dets.to_vector(h_dets));

    // Ghost position should coast by 1 frame's velocity (100 + 2.0 = 102.0)
    EXPECT_NEAR(h_dets[0].x, 102.0f, 0.1f);
    EXPECT_NEAR(h_dets[0].y, 98.5f, 0.1f);
}

TEST_F(ObjectTrackerTest, OutlierRejectionByMeanVelocity) {
    TrackerContext ctx;
    ctx.Init();

    int num_tracks = 33; // Just over 1 warp
    std::vector<TrackState> seedVec(ctx.maxTracks, TrackState{});

    // 1. Create 32 tracks moving smoothly to the right
    for(int i = 0; i < 32; ++i) {
        seedVec[i].id = i + 1;
        seedVec[i].age = 20;
        seedVec[i].x = 100.f; seedVec[i].y = 100.f;
        seedVec[i].w = 10.f;  seedVec[i].h = 10.f;
        seedVec[i].vx = 5.0f; seedVec[i].vy = 0.0f;
    }

    // 2. Create 1 Outlier track moving completely wrong
    seedVec[32].id = 33;
    seedVec[32].age = 20;
    seedVec[32].x = 100.f;  seedVec[32].y = 100.f;
    seedVec[32].w = 10.f;   seedVec[32].h = 10.f;
    seedVec[32].vx = -20.0f; // Wildly different velocity
    seedVec[32].vy = 0.0f;

    ASSERT_CUDA_SUCCESS(ctx.tracks.assign(seedVec));
    ASSERT_CUDA_SUCCESS(ctx.trackCount.assign({num_tracks}));

    // 3. Input: NO real detections
    BoundaryTypedBlock<DetectionRaw> d_dets;
    ASSERT_CUDA_SUCCESS(d_dets.resize(100));
    BoundaryBlock<int> d_detCount;
    ASSERT_CUDA_SUCCESS(d_detCount.assign({0}));

    // 4. Run Tracking
    ASSERT_CUDA_SUCCESS(TrackBatch(
        0, d_dets, d_detCount, ctx.countBufferHost,
        ctx.tracks, ctx.trackCount, ctx.trackCountHost, ctx.nextTrackId, ctx.matches,
        100, ctx.maxTracks, 1, 0.1f, 1024, 1024, 0
        ));

    // 5. Verify the Outlier was killed
    std::vector<TrackState> h_tracks;
    ASSERT_CUDA_SUCCESS(ctx.tracks.to_vector(h_tracks));

    EXPECT_EQ(h_tracks[32].age, -999) << "The outlier track should be marked as dead (-999)";
    EXPECT_EQ(h_tracks[0].age, 20) << "Normal tracks should remain alive and age up";

    std::vector<int> h_detCountOut;
    ASSERT_CUDA_SUCCESS(d_detCount.to_vector(h_detCountOut));
    EXPECT_EQ(h_detCountOut[0], 32) << "Only the 32 valid tracks should generate ghosts";
}

TEST_F(ObjectTrackerTest, GridStrideHandlesMoreThan1024Tracks) {
    TrackerContext ctx;
    // 1. Expand context beyond the 1024 limit
    ctx.maxTracks = 2500;
    ctx.Init();

    int num_tracks = 2000;
    std::vector<TrackState> seedVec(ctx.maxTracks, TrackState{});

    // 2. Create 2000 tracks moving smoothly to the right
    for(int i = 0; i < num_tracks; ++i) {
        seedVec[i].id = i + 1;
        seedVec[i].age = 20;
        seedVec[i].x = 100.f; seedVec[i].y = 100.f;
        seedVec[i].w = 10.f;  seedVec[i].h = 10.f;
        seedVec[i].vx = 5.0f; seedVec[i].vy = 0.0f;
    }

    // 3. Set an outlier well beyond the 1024 hardware thread limit
    seedVec[1500].vx = -20.0f;

    ASSERT_CUDA_SUCCESS(ctx.tracks.assign(seedVec));
    ASSERT_CUDA_SUCCESS(ctx.trackCount.assign({num_tracks}));

    // 4. Input: NO real detections (Forces Ghosting Phase)
    BoundaryTypedBlock<DetectionRaw> d_dets;
    // Must resize output to hold up to 3000 ghosts
    ASSERT_CUDA_SUCCESS(d_dets.resize(3000));
    BoundaryBlock<int> d_detCount;
    ASSERT_CUDA_SUCCESS(d_detCount.assign({0}));

    // Ensure host buffer has space
    ctx.countBufferHost.resize(1, 0);

    // 5. Run Tracking (Stride is bumped to 3000 to hold all ghosts)
    ASSERT_CUDA_SUCCESS(TrackBatch(
        0, d_dets, d_detCount, ctx.countBufferHost,
        ctx.tracks, ctx.trackCount, ctx.trackCountHost, ctx.nextTrackId, ctx.matches,
        3000, ctx.maxTracks, 1, 0.1f, 1024, 1024, 0
        ));

    // 6. Verification
    std::vector<TrackState> h_tracks;
    ASSERT_CUDA_SUCCESS(ctx.tracks.to_vector(h_tracks));

    // If the grid stride loop in Phase 5 works, Track 1500 will be killed
    EXPECT_EQ(h_tracks[1500].age, -999) << "The outlier track > 1024 should be marked as dead (-999)";
    // Track 1999 should have been processed and survived
    EXPECT_EQ(h_tracks[1999].age, 20) << "Normal tracks > 1024 should remain alive";

    std::vector<int> h_detCountOut;
    ASSERT_CUDA_SUCCESS(d_detCount.to_vector(h_detCountOut));
    // We started with 2000, killed 1 outlier. Exactly 1999 ghosts should be spawned.
    EXPECT_EQ(h_detCountOut[0], 1999) << "Exactly 1999 valid tracks should generate ghosts";
}

TEST_F(ObjectTrackerTest, FiniteGridStrideLoop) {
    TrackerContext ctx;
    ctx.maxTracks = 3000;
    ctx.Init();

    int num_tracks = 2049;
    std::vector<TrackState> seedVec(ctx.maxTracks, TrackState{});

    // Initialize all to dead to avoid noise
    for(int i = 0; i < ctx.maxTracks; i++) {
        seedVec[i].age = -999;
    }

    // 1. Track 0: Normal (Thread 0 succeeds, sidx becomes 1024)
    seedVec[0].id = 1;
    seedVec[0].age = 20;
    seedVec[0].x = 100.f; seedVec[0].y = 100.f;
    seedVec[0].w = 10.f;  seedVec[0].h = 10.f;
    seedVec[0].vx = 5.0f; seedVec[0].vy = 0.0f;
    seedVec[0].missedFrames = 0;

    // 2. Track 1024: STALE (Thread 0 hits 'continue', spins 1000x, and dies)
    seedVec[1024].id = 2;
    seedVec[1024].age = 70;
    seedVec[1024].x = 100.f; seedVec[1024].y = 100.f;
    seedVec[1024].w = 10.f;  seedVec[1024].h = 10.f;
    seedVec[1024].vx = 5.0f; seedVec[1024].vy = 0.0f;
    seedVec[1024].missedFrames = TRACKER_MAX_MISSED_FRAMES + 1;

    // 3. Track 2048: Normal (Thread 0 SHOULD process this, but it's dead!)
    seedVec[2048].id = 3;
    seedVec[2048].age = 20;
    seedVec[2048].x = 200.f; seedVec[2048].y = 200.f;
    seedVec[2048].w = 10.f;  seedVec[2048].h = 10.f;
    seedVec[2048].vx = 5.0f; seedVec[2048].vy = 0.0f;
    seedVec[2048].missedFrames = 0;

    ASSERT_CUDA_SUCCESS(ctx.tracks.assign(seedVec));
    ASSERT_CUDA_SUCCESS(ctx.trackCount.assign({num_tracks}));

    BoundaryTypedBlock<DetectionRaw> d_dets;
    ASSERT_CUDA_SUCCESS(d_dets.resize(100));
    BoundaryBlock<int> d_detCount;
    ASSERT_CUDA_SUCCESS(d_detCount.assign({0}));
    ctx.countBufferHost.resize(1, 0);

    // Run Tracking
    ASSERT_CUDA_SUCCESS(TrackBatch(
        0, d_dets, d_detCount, ctx.countBufferHost,
        ctx.tracks, ctx.trackCount, ctx.trackCountHost, ctx.nextTrackId, ctx.matches,
        100, ctx.maxTracks, 1, 0.1f, 1024, 1024, 0
        ));

    // Verification
    std::vector<TrackState> h_tracks;
    ASSERT_CUDA_SUCCESS(ctx.tracks.to_vector(h_tracks));

    // If the buggy kernel is used, Thread 0 dies before evaluating index 2048.
    // If the fixed kernel is used, Thread 0 survives and evaluates 2048 successfully.
    EXPECT_EQ(h_tracks[1024].age, -999) << "The stale track should be marked as dead";
    EXPECT_EQ(h_tracks[2048].age, 20) << "Track 2048 must survive. If this fails, Thread 0 died prematurely!";

    std::vector<int> h_detCountOut;
    ASSERT_CUDA_SUCCESS(d_detCount.to_vector(h_detCountOut));

    // Track 0 and Track 2048 should generate ghosts.
    // If the 'continue' bug is present, count will be 1. If fixed, count will be 2.
    EXPECT_EQ(h_detCountOut[0], 2) << "Both normal tracks (0 and 2048) must generate ghosts. If count is 1, the stride loop aborted early.";
}

} // namespace cropandweed
