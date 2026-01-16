#pragma once
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <filesystem>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include "FFmpegSource.h"
#include "FFmpegSourceKernels.h"
#include "BatchData.h"
#include "helpers.h"
#include "Block.h"

namespace cropandweed {

namespace fs = std::filesystem;

// Ensure macro is available
#ifndef ASSERT_CUDA_SUCCESS
#define ASSERT_CUDA_SUCCESS(err) ASSERT_FALSE(CudaError::IsFailure(err)) << (err).Text()
#endif

class FFmpegSourceTest : public ::testing::Test {
protected:
    fs::path video_path_;
    bool video_available_ = false;

    void SetUp() override {
        // 1. Force CUDA Context Initialization and clear sticky errors
        cudaFree(0);
        cudaGetLastError(); // Clear any previous error state

        // 2. Resolve Path with Priority: Current Dir -> Relative Dir
        fs::path filenames[] = {
            "Moving.mp4",
            fs::path("..") / "video" / "Moving.mp4"
        };

        for (const auto& p : filenames) {
            if (fs::exists(p)) {
                try {
                    video_path_ = fs::canonical(p);
                    video_available_ = true;
                    break;
                } catch (...) { continue; }
            }
        }

        if (!video_available_) {
            std::cerr << "[Test Setup] Warning: 'Moving.mp4' not found in current or parent/video directories." << std::endl;
        }
    }

    void TearDown() override {
        // Ensure errors from a failed test don't bleed into the next one
        cudaGetLastError();
    }
};

// ==========================================
// 1. FFmpegSource Class Tests
// ==========================================

TEST_F(FFmpegSourceTest, HandleInvalidURI) {
    std::unique_ptr<ISource> source;
    CudaError err = FFmpegSource::Create(source, "invalid_video_file.mp4");

    EXPECT_TRUE(CudaError::IsFailure(err));
    EXPECT_NE(err.Text().find("Could not open input"), std::string::npos);
}

TEST_F(FFmpegSourceTest, CreateUniquePtr) {
    if (!video_available_) GTEST_SKIP() << "Moving.mp4 not found.";

    std::unique_ptr<ISource> source;
    ASSERT_CUDA_SUCCESS(FFmpegSource::Create(source, video_path_.string()));
    ASSERT_NE(source, nullptr);
}

TEST_F(FFmpegSourceTest, GetNextBatchBasic) {
    if (!video_available_) GTEST_SKIP() << "Moving.mp4 not found.";

    std::unique_ptr<ISource> source;
    ASSERT_CUDA_SUCCESS(FFmpegSource::Create(source, video_path_.string()));

    source->SetOutputSize(300, 300);

    BatchData batch;
    size_t requestSize = 4;
    bool process = false;

    ASSERT_CUDA_SUCCESS(source->GetNextBatch(batch, requestSize, process));

    ASSERT_TRUE(process);
    EXPECT_EQ(batch.width, 300);
    EXPECT_EQ(batch.height, 300);
    EXPECT_LE(batch.batchSize, requestSize);
    EXPECT_GT(batch.batchSize, 0);

    size_t expectedElements = batch.batchSize * 300 * 300 * 3;
    EXPECT_EQ(batch.deviceData.size(), expectedElements);

    ASSERT_NE(batch.readyEvent, nullptr);
    cudaError_t syncErr = cudaEventSynchronize(*batch.readyEvent);
    EXPECT_EQ(syncErr, cudaSuccess);
}

TEST_F(FFmpegSourceTest, VerifyFrameCountAndEOS) {
    if (!video_available_) GTEST_SKIP() << "Moving.mp4 not found.";

    std::unique_ptr<ISource> source;
    ASSERT_CUDA_SUCCESS(FFmpegSource::Create(source, video_path_.string()));

    BatchData batch;
    size_t batchSize = 10;
    bool process = true;
    int total_frames_read = 0;

    while (process) {
        ASSERT_CUDA_SUCCESS(source->GetNextBatch(batch, batchSize, process));

        if (process) {
            total_frames_read += batch.batchSize;
            if (batch.readyEvent) {
                cudaEventSynchronize(*batch.readyEvent);
            }
        }
    }

    EXPECT_FALSE(process);
    EXPECT_NEAR(total_frames_read, 120, 2);
}

TEST_F(FFmpegSourceTest, VerifyRGBContent) {
    if (!video_available_) GTEST_SKIP() << "Moving.mp4 not found.";

    std::unique_ptr<ISource> source;
    ASSERT_CUDA_SUCCESS(FFmpegSource::Create(source, video_path_.string()));

    int w = 64;
    int h = 64;
    source->SetOutputSize(w, h);

    BatchData batch;
    bool process = false;
    ASSERT_CUDA_SUCCESS(source->GetNextBatch(batch, 1, process));

    if (process) {
        cudaEventSynchronize(*batch.readyEvent);

        std::vector<float> hostData(w * h * 3);
        cudaMemcpy(hostData.data(), batch.deviceData.data(), hostData.size() * sizeof(float), cudaMemcpyDeviceToHost);

        float sum = 0.0f;
        for(float v : hostData) sum += v;

        EXPECT_GT(sum, 1.0f) << "Frame data was completely zero (black), check decoding kernel.";
    }
}

// ==========================================
// 2. Kernel Tests (Direct Invocation)
// ==========================================

class FFmpegSourceKernelTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Clear any sticky errors before kernel tests
        cudaGetLastError();
    }

    void RunKernelTest(bool use_texture_obj) {
        int srcW = 2;
        int srcH = 2;

        // 1. Determine Valid Texture Pitch
        // CUDA textures require the pitch to be aligned (e.g., 256 or 512 bytes).
        // If we don't align it, cudaCreateTextureObject returns cudaErrorInvalidValue.
        size_t texAlignment = 0;
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        texAlignment = prop.texturePitchAlignment;

        // Ensure pitch is at least srcW but multiple of alignment
        size_t srcPitch = (srcW < texAlignment) ? texAlignment : (srcW + texAlignment - 1) / texAlignment * texAlignment;

        // Host Data (2x2)
        std::vector<uint8_t> h_Y = { 16, 235, 128, 128 }; // Y Plane
        std::vector<uint8_t> h_UV = { 128, 128 };         // UV Plane (interleaved)

        // 2. Allocate GPU Memory using aligned Pitch
        // We use Block to manage the memory, but we must allocate enough for padding.
        Block<uint8_t> d_Y;
        Block<uint8_t> d_UV;
        Block<float> d_Dst;

        ASSERT_CUDA_SUCCESS(d_Y.resize(srcPitch * srcH));
        ASSERT_CUDA_SUCCESS(d_UV.resize(srcPitch * (srcH / 2))); // UV is half height

        // Output: 2x2 image, 3 channels.
        size_t dstElements = srcW * srcH * 3;
        ASSERT_CUDA_SUCCESS(d_Dst.resize(dstElements));
        cudaMemset(d_Dst.data(), 0, dstElements * sizeof(float));

        // 3. Copy Data with Pitch (Host: Tight, Device: Aligned)
        // Use cudaMemcpy2D to handle the stride difference automatically
        ASSERT_EQ(cudaMemcpy2D(d_Y.data(), srcPitch, h_Y.data(), srcW, srcW, srcH, cudaMemcpyHostToDevice), cudaSuccess);

        // UV height is srcH/2. UV width is srcW (interleaved bytes).
        ASSERT_EQ(cudaMemcpy2D(d_UV.data(), srcPitch, h_UV.data(), srcW, srcW, srcH / 2, cudaMemcpyHostToDevice), cudaSuccess);

        // 4. Run Kernel
        CudaError err = NV12ToRGBPlanar(
            d_Y.data(), d_UV.data(), (int)srcPitch, // Pass aligned pitch!
            d_Dst.data(), 0,
            srcW, srcH,
            srcW, srcH,
            use_texture_obj,
            0
            );

        ASSERT_CUDA_SUCCESS(err);
        cudaDeviceSynchronize();

        // Download Result
        std::vector<float> h_Dst(dstElements);
        cudaMemcpy(h_Dst.data(), d_Dst.data(), dstElements * sizeof(float), cudaMemcpyDeviceToHost);

        // Verification Logic
        float tolerance = 0.05f;

        // Pixel 0 (0,0): Y=16. Expect ~0.0
        EXPECT_NEAR(h_Dst[0], 0.0f, tolerance) << "Pixel 0 Red channel mismatch";

        // Pixel 1 (1,0): Y=235. Expect Bright (~0.85 - 1.0)
        EXPECT_GT(h_Dst[1], 0.8f) << "Pixel 1 Red channel should be bright";
    }
};

TEST_F(FFmpegSourceKernelTest, TextureObjectImplementation) {
    RunKernelTest(true);
}

TEST_F(FFmpegSourceKernelTest, GlobalMemoryImplementation) {
    RunKernelTest(false);
}

} // namespace cropandweed
