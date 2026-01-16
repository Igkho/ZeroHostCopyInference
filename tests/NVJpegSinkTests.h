#pragma once
#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <vector>
#include <string>
#include <cuda_runtime.h>

#include "NVJpegSink.h"
#include "NVJpegSinkKernels.h"
#include "BatchData.h"
#include "BatchDetections.h"
#include "Block.h"
#include "helpers.h"

namespace cropandweed {

namespace fs = std::filesystem;

// Local assertion macro
#ifndef ASSERT_CUDA_SUCCESS
#define ASSERT_CUDA_SUCCESS(err) ASSERT_FALSE(CudaError::IsFailure(err)) << (err).Text()
#endif

class NVJpegSinkTest : public ::testing::Test {
protected:
    fs::path output_dir_;

    void SetUp() override {
        // Unique output directory for this test run
        output_dir_ = fs::current_path() / "test_output_sink";
        if (fs::exists(output_dir_)) {
            fs::remove_all(output_dir_);
        }
    }

    void TearDown() override {
        // Cleanup produced files
        if (fs::exists(output_dir_)) {
            fs::remove_all(output_dir_);
        }
    }
};

// ==========================================
// 1. Kernel Tests (FloatToUint8)
// ==========================================

TEST_F(NVJpegSinkTest, KernelFloatToUint8Conversion) {
    // Test values covering: Underflow, Zero, Mid-range, One, Overflow
    std::vector<float> h_src = { -0.5f, 0.0f, 0.5f, 1.0f, 1.5f };
    int count = (int)h_src.size();

    Block<float> d_src;
    Block<uint8_t> d_dst;

    ASSERT_CUDA_SUCCESS(d_src.resize(count));
    ASSERT_CUDA_SUCCESS(d_dst.resize(count));

    // Copy to GPU
    cudaMemcpy(d_src.data(), h_src.data(), count * sizeof(float), cudaMemcpyHostToDevice);

    // Run Kernel
    ASSERT_CUDA_SUCCESS(FloatToUint8(d_src.data(), d_dst.data(), count));
    
    // Sync (Kernel runs on default stream)
    cudaDeviceSynchronize();

    // Copy back
    std::vector<uint8_t> h_dst(count);
    cudaMemcpy(h_dst.data(), d_dst.data(), count * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    // Verify
    // -0.5 -> 0
    EXPECT_EQ(h_dst[0], 0);
    // 0.0 -> 0
    EXPECT_EQ(h_dst[1], 0);
    // 0.5 -> 128 (approx 127 or 128 depending on rounding 0.5*255 = 127.5)
    // The kernel uses `val * 255.0f + 0.5f` (rounding to nearest)
    // 0.5 * 255 = 127.5 + 0.5 = 128.0 -> 128
    EXPECT_EQ(h_dst[2], 128);
    // 1.0 -> 255
    EXPECT_EQ(h_dst[3], 255);
    // 1.5 -> 255
    EXPECT_EQ(h_dst[4], 255);
}

// ==========================================
// 2. Sink Lifecycle Tests
// ==========================================

TEST_F(NVJpegSinkTest, CreateAndInit) {
    std::unique_ptr<ISink> sink;
    // Factory should create the directory if it doesn't exist
    ASSERT_CUDA_SUCCESS(NVJpegSink::Create(sink, output_dir_.string()));
    
    ASSERT_NE(sink, nullptr);
    EXPECT_TRUE(fs::exists(output_dir_));
    EXPECT_TRUE(fs::is_directory(output_dir_));
}

// ==========================================
// 3. Functional Save Tests
// ==========================================

TEST_F(NVJpegSinkTest, SaveBatchToJpeg) {
    // 1. Setup Sink
    std::unique_ptr<ISink> sink;
    ASSERT_CUDA_SUCCESS(NVJpegSink::Create(sink, output_dir_.string()));

    // 2. Prepare Batch Data (2 images, 64x64)
    int batchSize = 2;
    int w = 64;
    int h = 64;
    
    std::shared_ptr<BatchData> data;
    ASSERT_CUDA_SUCCESS(BatchData::Create(data, 0, batchSize, w, h));
    
    // Fill image with a gradient pattern to ensure valid JPEG encoding
    // Planar RGB: RRR... GGG... BBB...
    std::vector<float> hostImg(batchSize * w * h * 3);
    for (int b = 0; b < batchSize; ++b) {
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                int idx = (b * w * h * 3) + (y * w + x);
                // Red Channel (gradient x)
                hostImg[idx] = (float)x / w;
                // Green Channel (gradient y)
                hostImg[idx + w * h] = (float)y / h;
                // Blue Channel (static)
                hostImg[idx + 2 * w * h] = 0.5f;
            }
        }
    }
    cudaMemcpy(data->deviceData.data(), hostImg.data(), hostImg.size() * sizeof(float), cudaMemcpyHostToDevice);
    
    // Set IDs
    data->sourceIdentifiers = {"img_A", "img_B"};
    
    // 3. Prepare Batch Detections (Empty is fine, ObjectTracker handles it)
    std::shared_ptr<BatchDetections> results;
    ASSERT_CUDA_SUCCESS(BatchDetections::Create(results, batchSize));
    
    // Need to initialize counts to 0 to prevent tracker reading garbage
    cudaMemset(results->counts.data(), 0, batchSize * sizeof(int));

    // Signal events (simulating pipeline completion)
    cudaEventRecord(*data->readyEvent, 0);
    cudaEventRecord(*results->readyEvent, 0);

    // 4. Run Save
    ASSERT_CUDA_SUCCESS(sink->Save(*data, *results));

    // 5. Verification
    fs::path file1 = output_dir_ / "frame_img_A.jpg";
    fs::path file2 = output_dir_ / "frame_img_B.jpg";

    EXPECT_TRUE(fs::exists(file1)) << "Missing " << file1;
    EXPECT_TRUE(fs::exists(file2)) << "Missing " << file2;

    if (fs::exists(file1)) {
        EXPECT_GT(fs::file_size(file1), 100) << "JPEG file is suspiciously small";
    }
    if (fs::exists(file2)) {
        EXPECT_GT(fs::file_size(file2), 100) << "JPEG file is suspiciously small";
    }
}

TEST_F(NVJpegSinkTest, HandleLargeImages) {
    // Verify memory allocation handling for larger resolutions
    std::unique_ptr<ISink> sink;
    ASSERT_CUDA_SUCCESS(NVJpegSink::Create(sink, output_dir_.string()));

    int w = 1920;
    int h = 1080;
    std::shared_ptr<BatchData> data;
    ASSERT_CUDA_SUCCESS(BatchData::Create(data, 1, 1, w, h));
    
    // Initialize to grey
    cudaMemset(data->deviceData.data(), 0, data->deviceData.size() * sizeof(float));

    std::shared_ptr<BatchDetections> results;
    ASSERT_CUDA_SUCCESS(BatchDetections::Create(results, 1));
    cudaMemset(results->counts.data(), 0, sizeof(int));

    // Signal events
    cudaEventRecord(*data->readyEvent, 0);
    cudaEventRecord(*results->readyEvent, 0);

    ASSERT_CUDA_SUCCESS(sink->Save(*data, *results));

    // Check output
    // If IDs are missing, default naming is used: frame_{ID}.jpg
    // BatchId=1, Index=0 -> frame_1.jpg (since 1 * 1 + 0 = 1)
    fs::path file = output_dir_ / "frame_1.jpg"; 
    EXPECT_TRUE(fs::exists(file));
}

} // namespace cropandweed