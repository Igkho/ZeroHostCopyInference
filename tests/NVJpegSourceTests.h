#pragma once
#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <vector>
#include <string>
#include <memory>
#include <iomanip>
#include <sstream>
#include <cuda_runtime.h>

#include "NVJpegSource.h"
#include "BatchData.h"
#include "helpers.h"

namespace cropandweed {

namespace fs = std::filesystem;

#ifndef ASSERT_CUDA_SUCCESS
#define ASSERT_CUDA_SUCCESS(err) ASSERT_FALSE(CudaError::IsFailure(err)) << (err).Text()
#endif

class NVJpegSourceTest : public ::testing::Test {
protected:
    fs::path test_dir_;

    // A minimal valid 8x8 grayscale JPEG byte array to avoid external file dependencies.
    const std::vector<uint8_t> valid_jpeg_bytes_ = {
        0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10, 0x4a, 0x46, 0x49, 0x46, 0x00, 0x01,
        0x01, 0x01, 0x00, 0x48, 0x00, 0x48, 0x00, 0x00, 0xff, 0xdb, 0x00, 0x43,
        0x00, 0x03, 0x02, 0x02, 0x02, 0x02, 0x02, 0x03, 0x02, 0x02, 0x02, 0x03,
        0x03, 0x03, 0x03, 0x04, 0x06, 0x04, 0x04, 0x04, 0x04, 0x04, 0x08, 0x06,
        0x06, 0x05, 0x06, 0x09, 0x08, 0x0a, 0x0a, 0x09, 0x08, 0x09, 0x09, 0x0a,
        0x0c, 0x0f, 0x0c, 0x0a, 0x0b, 0x0e, 0x0b, 0x09, 0x09, 0x0d, 0x11, 0x0d,
        0x0e, 0x0f, 0x10, 0x10, 0x11, 0x10, 0x0a, 0x0c, 0x12, 0x13, 0x12, 0x10,
        0x13, 0x0f, 0x10, 0x10, 0x10, 0xff, 0xc0, 0x00, 0x0b, 0x08, 0x00, 0x08,
        0x00, 0x08, 0x01, 0x01, 0x11, 0x00, 0xff, 0xc4, 0x00, 0x1f, 0x00, 0x00,
        0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
        0x09, 0x0a, 0x0b, 0xff, 0xc4, 0x00, 0xb5, 0x10, 0x00, 0x02, 0x01, 0x03,
        0x03, 0x02, 0x04, 0x03, 0x05, 0x05, 0x04, 0x04, 0x00, 0x00, 0x01, 0x7d,
        0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06,
        0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08,
        0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0, 0x24, 0x33, 0x62, 0x72,
        0x82, 0x09, 0x0a, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28,
        0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45,
        0x46, 0x47, 0x48, 0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
        0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6a, 0x73, 0x74, 0x75,
        0x76, 0x77, 0x78, 0x79, 0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
        0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3,
        0xa4, 0xa5, 0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6,
        0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9,
        0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2,
        0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf1, 0xf2, 0xf3, 0xf4,
        0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa, 0xff, 0xda, 0x00, 0x08, 0x01, 0x01,
        0x00, 0x00, 0x3f, 0x00, 0x3f, 0xff, 0xd9
    };

    void SetUp() override {
        cudaGetLastError(); // Clear sticky errors
        test_dir_ = fs::current_path() / "test_nvjpeg_source";
        if (fs::exists(test_dir_)) fs::remove_all(test_dir_);
        fs::create_directory(test_dir_);
    }

    void TearDown() override {
        if (fs::exists(test_dir_)) fs::remove_all(test_dir_);
        cudaGetLastError();
    }

    // Ensures zero-padded sorting (e.g., 000_good.jpg, 001_bad.jpg)
    void CreateValidJpeg(int index, const std::string& suffix = "good") {
        std::stringstream ss;
        ss << std::setw(3) << std::setfill('0') << index << "_" << suffix << ".jpg";
        fs::path file_path = test_dir_ / ss.str();
        std::ofstream out(file_path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(valid_jpeg_bytes_.data()), valid_jpeg_bytes_.size());
    }

    // Generates a deterministic corrupt file
    void CreateCorruptJpeg(int index, const std::string& suffix = "bad") {
        std::stringstream ss;
        ss << std::setw(3) << std::setfill('0') << index << "_" << suffix << ".jpg";
        fs::path file_path = test_dir_ / ss.str();
        std::ofstream out(file_path, std::ios::binary);
        out.write("This is a truncated/invalid JPEG header", 39);
    }

    // Legacy helper for standard batch tests
    void CreateDummyJpegs(int count) {
        for (int i = 0; i < count; ++i) {
            CreateValidJpeg(i);
        }
    }
};

// ==========================================
// 1. Initialization & Path Validation
// ==========================================

TEST_F(NVJpegSourceTest, InitFailsOnInvalidDirectory) {
    std::unique_ptr<ISource> source;
    CudaError err = NVJpegSource::Create(source, "/path/to/nowhere/that/does/not/exist", 224, 224);

    EXPECT_TRUE(CudaError::IsFailure(err));
    EXPECT_NE(err.Text().find("Invalid folder path"), std::string::npos);
}

TEST_F(NVJpegSourceTest, InitSucceedsOnEmptyDirectory) {
    std::unique_ptr<ISource> source;
    ASSERT_CUDA_SUCCESS(NVJpegSource::Create(source, test_dir_.string(), 224, 224));
    ASSERT_NE(source, nullptr);

    BatchData batch;
    bool process = true;
    ASSERT_CUDA_SUCCESS(source->GetNextBatch(batch, 4, process));
    EXPECT_FALSE(process) << "Should cleanly signal EOF on an empty directory";
}

// ==========================================
// 2. Batch Decoding Logic
// ==========================================

TEST_F(NVJpegSourceTest, DecodesSingleBatchExactly) {
    int targetW = 224;
    int targetH = 224;
    int batchSize = 4;
    CreateDummyJpegs(batchSize);

    std::unique_ptr<ISource> source;
    ASSERT_CUDA_SUCCESS(NVJpegSource::Create(source, test_dir_.string(), targetW, targetH));

    BatchData batch;
    bool process = false;

    ASSERT_CUDA_SUCCESS(source->GetNextBatch(batch, batchSize, process));

    EXPECT_TRUE(process);
    EXPECT_EQ(batch.batchSize, batchSize);
    EXPECT_EQ(batch.width, targetW);
    EXPECT_EQ(batch.height, targetH);
    EXPECT_EQ(batch.sourceIdentifiers.size(), batchSize);

    size_t expectedElements = batchSize * targetW * targetH * 3;
    EXPECT_EQ(batch.deviceData.size(), expectedElements);

    ASSERT_NE(batch.readyEvent, nullptr);
    EXPECT_EQ(cudaEventSynchronize(*batch.readyEvent), cudaSuccess);
}

TEST_F(NVJpegSourceTest, DecodesPartialBatchAtEOF) {
    CreateDummyJpegs(3);

    std::unique_ptr<ISource> source;
    ASSERT_CUDA_SUCCESS(NVJpegSource::Create(source, test_dir_.string(), 64, 64));

    BatchData batch;
    bool process = false;

    ASSERT_CUDA_SUCCESS(source->GetNextBatch(batch, 8, process));
    EXPECT_TRUE(process);
    EXPECT_EQ(batch.batchSize, 3);

    ASSERT_CUDA_SUCCESS(source->GetNextBatch(batch, 8, process));
    EXPECT_FALSE(process);
}

// ==========================================
// 3. Robust Error Handling (Skips)
// ==========================================

// Replaces the old CorruptImagePropagatesErrorSafely
TEST_F(NVJpegSourceTest, CorruptImageIsSkippedSafely) {
    // Generate: 000_good.jpg, 001_bad.jpg, 002_good.jpg
    CreateValidJpeg(0);
    CreateCorruptJpeg(1);
    CreateValidJpeg(2);

    std::unique_ptr<ISource> source;
    ASSERT_CUDA_SUCCESS(NVJpegSource::Create(source, test_dir_.string(), 128, 128));

    BatchData batch;
    bool process = false;

    // Request 4. Source sees 3 files. It parses them, finds 1 is bad.
    ASSERT_CUDA_SUCCESS(source->GetNextBatch(batch, 4, process));

    EXPECT_TRUE(process);
    EXPECT_EQ(batch.batchSize, 2) << "Must discard the corrupt image and yield the 2 valid ones";

    // Explicitly verify the missed line regression from the previous code review
    EXPECT_EQ(batch.width, 128) << "Target width must be assigned even on partial successes";
    EXPECT_EQ(batch.height, 128) << "Target height must be assigned even on partial successes";
    EXPECT_EQ(batch.sourceIdentifiers.size(), 2);
}

// New test verifying recursive fetching logic
TEST_F(NVJpegSourceTest, EntireCorruptBatchAdvancesToNextValidBatch) {
    // Request size is 2.
    // Chunk 1: 000_bad, 001_bad (Total Failure)
    // Chunk 2: 002_good (Partial Success / EOF)
    CreateCorruptJpeg(0);
    CreateCorruptJpeg(1);
    CreateValidJpeg(2);

    std::unique_ptr<ISource> source;
    ASSERT_CUDA_SUCCESS(NVJpegSource::Create(source, test_dir_.string(), 64, 64));

    BatchData batch;
    bool process = false;

    ASSERT_CUDA_SUCCESS(source->GetNextBatch(batch, 2, process));

    EXPECT_TRUE(process) << "Must recursively fetch the next chunk if the current chunk completely fails";
    EXPECT_EQ(batch.batchSize, 1) << "Must return the 1 valid image from Chunk 2";
    EXPECT_EQ(batch.width, 64);
    EXPECT_EQ(batch.height, 64);
}

// New test verifying that terminal corruptions don't cause infinite loops
TEST_F(NVJpegSourceTest, CorruptImagesAtEOFYieldsNoProcess) {
    CreateCorruptJpeg(0);
    CreateCorruptJpeg(1);

    std::unique_ptr<ISource> source;
    ASSERT_CUDA_SUCCESS(NVJpegSource::Create(source, test_dir_.string(), 64, 64));

    BatchData batch;
    bool process = true; // Initialize to true to verify it correctly flips

    ASSERT_CUDA_SUCCESS(source->GetNextBatch(batch, 4, process));

    EXPECT_FALSE(process) << "Pipeline must cleanly signal EOF if remaining images are corrupt";
    EXPECT_EQ(batch.batchSize, 0);
}

// Restored critical double-buffering state test
TEST_F(NVJpegSourceTest, ExercisesDoubleBufferingWrapAround) {
    // Double buffering uses 2 internal states. We need to process at least 3 batches
    // to force a wrap-around and ensure the synchronizations don't deadlock or corrupt memory.
    int batchSize = 2;
    CreateDummyJpegs(6); // 3 full batches: (0,1), (2,3), (4,5)

    std::unique_ptr<ISource> source;
    ASSERT_CUDA_SUCCESS(NVJpegSource::Create(source, test_dir_.string(), 128, 128));

    BatchData batch;
    bool process = true;
    int batchesProcessed = 0;

    while (process) {
        ASSERT_CUDA_SUCCESS(source->GetNextBatch(batch, batchSize, process));
        if (process) {
            batchesProcessed++;
            EXPECT_EQ(batch.batchSize, batchSize);

            // Simulate downstream pipeline delay (Wait for DMA)
            ASSERT_NE(batch.readyEvent, nullptr);
            cudaEventSynchronize(*batch.readyEvent);
        }
    }

    EXPECT_EQ(batchesProcessed, 3) << "Failed to process all 3 batches. Wrap-around DMA sync logic may be deadlocked.";
}

// Test verifying strict engine capacity and zero-padding on partial batches
TEST_F(NVJpegSourceTest, PartialBatchMaintainsStrictEngineCapacityAndZeroPads) {
    // Generate: 000_good.jpg, 001_good.jpg, 002_bad.jpg
    CreateValidJpeg(0);
    CreateValidJpeg(1);
    CreateCorruptJpeg(2);

    int targetW = 64;
    int targetH = 64;
    int reqBatchSize = 4;

    std::unique_ptr<ISource> source;
    ASSERT_CUDA_SUCCESS(NVJpegSource::Create(source, test_dir_.string(), targetW, targetH));

    BatchData batch;
    bool process = false;

    // Request 4 frames. It should find 3 files, 1 is bad. Valid count = 2.
    ASSERT_CUDA_SUCCESS(source->GetNextBatch(batch, reqBatchSize, process));

    EXPECT_TRUE(process);
    EXPECT_EQ(batch.batchSize, 2) << "Must report exactly 2 valid frames";

    // 1. Verify Strict Engine Capacity
    size_t expectedTotalElements = reqBatchSize * targetW * targetH * 3;
    EXPECT_EQ(batch.deviceData.size(), expectedTotalElements)
        << "Buffer must remain sized for the requested batchSize (4) to prevent TRT/ONNX shape errors";

    // 2. Verify Zero-Padding of the tail
    if (batch.readyEvent) {
        cudaEventSynchronize(*batch.readyEvent);
    }

    std::vector<float> hostData;
    ASSERT_CUDA_SUCCESS(batch.deviceData.to_vector(hostData));

    size_t validElements = batch.batchSize * targetW * targetH * 3;

    // Sum the padding area to ensure it's completely empty
    double paddingSum = 0.0;
    for (size_t i = validElements; i < expectedTotalElements; ++i) {
        paddingSum += hostData[i];
    }

    EXPECT_DOUBLE_EQ(paddingSum, 0.0) << "Padding frames must be zero-filled to prevent NaN poisoning";
}

} // namespace cropandweed
