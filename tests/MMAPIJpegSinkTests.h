#pragma once
#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <vector>
#include <string>
#include <cuda_runtime.h>

#include "MMAPIJpegSink.h"
#include "SinkKernels.h"
#include "BatchData.h"
#include "BatchDetections.h"
#include "Block.h"
#include "helpers.h"

namespace cropandweed {

namespace fs = std::filesystem;

#ifndef ASSERT_CUDA_SUCCESS
#define ASSERT_CUDA_SUCCESS(err) ASSERT_FALSE(CudaError::IsFailure(err)) << (err).Text()
#endif

class MMAPIJpegSinkTest : public ::testing::Test {
protected:
    fs::path output_dir_;

    void SetUp() override {
        // Unique output directory for this test run
        output_dir_ = fs::current_path() / "test_mmapi_output_sink";
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
// 1. Sink Lifecycle Tests
// ==========================================

TEST_F(MMAPIJpegSinkTest, CreateAndInit) {
    std::unique_ptr<ISink> sink;
    int batchSize = 4;

    // Factory should create the directory if it doesn't exist
    ASSERT_CUDA_SUCCESS(MMAPIJpegSink::Create(sink, output_dir_.string(),
                                              {1024, 1024, 3, {"", "", ""}}, batchSize));

    ASSERT_NE(sink, nullptr);
    EXPECT_TRUE(fs::exists(output_dir_));
    EXPECT_TRUE(fs::is_directory(output_dir_));
}

// ==========================================
// 2. Functional Save Tests
// ==========================================

TEST_F(MMAPIJpegSinkTest, SaveBatchToJpeg) {
    size_t w = 64;
    size_t h = 64;
    int batchSize = 2;

    // 1. Setup Sink
    std::unique_ptr<ISink> sink;
    ASSERT_CUDA_SUCCESS(MMAPIJpegSink::Create(sink, output_dir_.string(),
                                              {w, h, 3, {"", "", ""}}, batchSize));

    // 2. Prepare Batch Data
    std::shared_ptr<BatchData> data;
    ASSERT_CUDA_SUCCESS(BatchData::Create(data, 0, batchSize, w, h));

    // Fill image with a gradient pattern to ensure valid JPEG encoding
    std::vector<float> hostImg(batchSize * w * h * 3);
    for (int b = 0; b < batchSize; ++b) {
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                int idx = (b * w * h * 3) + (y * w + x);
                hostImg[idx] = (float)x / w;
                hostImg[idx + w * h] = (float)y / h;
                hostImg[idx + 2 * w * h] = 0.5f;
            }
        }
    }
    ASSERT_CUDA_SUCCESS(data->deviceData.assign(hostImg));

    // Set IDs (MMAPIJpegSink sets output to `frame_0000{id}.jpg` due to setw pad, but if
    // ID exceeds 4 chars, it prints the ID as-is. So "img_A" -> "frame_img_A.jpg").
    data->sourceIdentifiers = {"img_A", "img_B"};

    // 3. Prepare Batch Detections
    std::shared_ptr<BatchDetections> results;
    ASSERT_CUDA_SUCCESS(BatchDetections::Create(results, batchSize));
    ASSERT_CUDA_SUCCESS(results->counts.fill(0, 0));

    // Signal events (simulating pipeline completion)
    cudaEventRecord(*data->readyEvent, 0);
    cudaEventRecord(*results->readyEvent, 0);

    // 4. Run Save
    ASSERT_CUDA_SUCCESS(sink->Save(*data, *results));
    ASSERT_CUDA_SUCCESS(sink->Close());

    // 5. Verification
    fs::path file1 = output_dir_ / "frame_img_A.jpg";
    fs::path file2 = output_dir_ / "frame_img_B.jpg";

    EXPECT_TRUE(fs::exists(file1)) << "Missing " << file1;
    EXPECT_TRUE(fs::exists(file2)) << "Missing " << file2;

    if (fs::exists(file1)) {
        EXPECT_GT(fs::file_size(file1), 100) << "JPEG file is suspiciously small";
    }
}

TEST_F(MMAPIJpegSinkTest, HandleLargeImages) {
    size_t w = 1920;
    size_t h = 1080;
    int batchSize = 1;

    std::unique_ptr<ISink> sink;
    ASSERT_CUDA_SUCCESS(MMAPIJpegSink::Create(sink, output_dir_.string(),
                                              {w, h, 3, {"", "", ""}}, batchSize));

    std::shared_ptr<BatchData> data;
    ASSERT_CUDA_SUCCESS(BatchData::Create(data, 1, 1, w, h));
    ASSERT_CUDA_SUCCESS(data->deviceData.fill(0, 0));

    std::shared_ptr<BatchDetections> results;
    ASSERT_CUDA_SUCCESS(BatchDetections::Create(results, 1));
    ASSERT_CUDA_SUCCESS(results->counts.fill(0, 0));

    cudaEventRecord(*data->readyEvent, 0);
    cudaEventRecord(*results->readyEvent, 0);

    ASSERT_CUDA_SUCCESS(sink->Save(*data, *results));
    ASSERT_CUDA_SUCCESS(sink->Close());

    // Fallback logic uses batchId * batchSize + frame_idx
    // 1 * 1 + 0 = 1 -> formatted with setw(4) -> "0001"
    fs::path file = output_dir_ / "frame_0001.jpg";
    EXPECT_TRUE(fs::exists(file));
}

// ==========================================
// 3. Advanced Architecture & State Machine Tests
// ==========================================

TEST_F(MMAPIJpegSinkTest, VerifiesDoubleBufferedDeferral) {
    size_t w = 64;
    size_t h = 64;
    int batchSize = 2;

    std::unique_ptr<ISink> sink;
    ASSERT_CUDA_SUCCESS(MMAPIJpegSink::Create(sink, output_dir_.string(), {w, h, 3, {"", "", ""}}, batchSize));

    std::shared_ptr<BatchDetections> results;
    ASSERT_CUDA_SUCCESS(BatchDetections::Create(results, 1));
    ASSERT_CUDA_SUCCESS(results->counts.fill(0, 0));
    cudaEventRecord(*results->readyEvent, 0);

    // 1. Send BATCH 1 (Uses Buffer 0)
    std::shared_ptr<BatchData> data1;
    ASSERT_CUDA_SUCCESS(BatchData::Create(data1, 1, 1, w, h));
    ASSERT_CUDA_SUCCESS(data1->deviceData.fill(0, 0));
    data1->sourceIdentifiers = {"batch1_img"};
    cudaEventRecord(*data1->readyEvent, 0);
    ASSERT_CUDA_SUCCESS(sink->Save(*data1, *results));

    // 2. Send BATCH 2 (Uses Buffer 1 - Runs concurrently with Buffer 0!)
    std::shared_ptr<BatchData> data2;
    ASSERT_CUDA_SUCCESS(BatchData::Create(data2, 2, 1, w, h));
    ASSERT_CUDA_SUCCESS(data2->deviceData.fill(0, 0));
    data2->sourceIdentifiers = {"batch2_img"};
    cudaEventRecord(*data2->readyEvent, 0);
    ASSERT_CUDA_SUCCESS(sink->Save(*data2, *results));

    // 3. Send BATCH 3 (Uses Buffer 0)
    // Because it reuses Buffer 0, it MUST synchronize and flush Batch 1!
    std::shared_ptr<BatchData> data3;
    ASSERT_CUDA_SUCCESS(BatchData::Create(data3, 3, 1, w, h));
    ASSERT_CUDA_SUCCESS(data3->deviceData.fill(0, 0));
    data3->sourceIdentifiers = {"batch3_img"};
    cudaEventRecord(*data3->readyEvent, 0);
    ASSERT_CUDA_SUCCESS(sink->Save(*data3, *results));

    // Verify Batch 1 is guaranteed to be fully written now
    fs::path file1 = output_dir_ / "frame_batch1_img.jpg";
    EXPECT_TRUE(fs::exists(file1)) << "Batch 3 failed to synchronize the async IO threads for Batch 1.";

    // 4. Explicit Close forces synchronization of the trailing Batch 2 and Batch 3
    ASSERT_CUDA_SUCCESS(sink->Close());

    fs::path file2 = output_dir_ / "frame_batch2_img.jpg";
    fs::path file3 = output_dir_ / "frame_batch3_img.jpg";
    EXPECT_TRUE(fs::exists(file2)) << "Close() failed to flush Batch 2.";
    EXPECT_TRUE(fs::exists(file3)) << "Close() failed to flush Batch 3.";
}

TEST_F(MMAPIJpegSinkTest, DestructorFallbackSavesUnflushedData) {
    size_t w = 64;
    size_t h = 64;
    int batchSize = 1;

    std::shared_ptr<BatchData> data;
    ASSERT_CUDA_SUCCESS(BatchData::Create(data, 1, 1, w, h));
    ASSERT_CUDA_SUCCESS(data->deviceData.fill(0, 0));
    data->sourceIdentifiers = {"destructor_test"};
    cudaEventRecord(*data->readyEvent, 0);

    std::shared_ptr<BatchDetections> results;
    ASSERT_CUDA_SUCCESS(BatchDetections::Create(results, 1));
    ASSERT_CUDA_SUCCESS(results->counts.fill(0, 0));
    cudaEventRecord(*results->readyEvent, 0);

    fs::path expected_file = output_dir_ / "frame_destructor_test.jpg";

    {
        // Scope the sink so we can force its destructor to run early
        std::unique_ptr<ISink> sink;
        ASSERT_CUDA_SUCCESS(MMAPIJpegSink::Create(sink, output_dir_.string(), {w, h, 3, {"", "", ""}}, batchSize));

        ASSERT_CUDA_SUCCESS(sink->Save(*data, *results));

        // Data is encoding asynchronously. We do NOT explicitly call Close().
        // The sink goes out of scope here, triggering the destructor.
    }

    // Verify the destructor caught the unflushed future tasks and joined them safely
    EXPECT_TRUE(fs::exists(expected_file)) << "Destructor failed to join async threads and flush remaining data!";
    EXPECT_GT(fs::file_size(expected_file), 100);
}

} // namespace cropandweed
