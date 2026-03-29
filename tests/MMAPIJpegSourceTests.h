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

#include "MMAPIJpegSource.h"
#include "BatchData.h"
#include "helpers.h"

// Use the MMAPI Hardware Encoder to synthesize the dummy test files!
#include "NvJpegEncoder.h"
#include "nvbufsurface.h"

namespace cropandweed {

namespace fs = std::filesystem;

#ifndef ASSERT_CUDA_SUCCESS
#define ASSERT_CUDA_SUCCESS(err) ASSERT_FALSE(CudaError::IsFailure(err)) << (err).Text()
#endif

class MMAPIJpegSourceTest : public ::testing::Test {
protected:
    fs::path test_dir_;

    void SetUp() override {
        cudaGetLastError(); // Clear sticky errors
        test_dir_ = fs::current_path() / "test_mmapi_source";
        if (fs::exists(test_dir_)) fs::remove_all(test_dir_);
        fs::create_directory(test_dir_);
    }

    void TearDown() override {
        if (fs::exists(test_dir_)) fs::remove_all(test_dir_);
        cudaGetLastError();
    }

    // Dynamically generates a valid 64x64 NV12 JPEG using the MMAPI Hardware Encoder.
    // This perfectly bypasses the Tegra vs CUDA linker collision!
    void CreateValidJpeg(int index, const std::string& suffix = "good") {
        std::unique_ptr<NvJPEGEncoder> encoder(NvJPEGEncoder::createJPEGEncoder("test_enc"));
        if (!encoder) return;

        NvBufSurface* surf = nullptr;
        NvBufSurfaceCreateParams params{};
        params.gpuId = 0;
        params.width = 64;
        params.height = 64;
        params.size = 0;
        params.colorFormat = NVBUF_COLOR_FORMAT_NV12;
        params.layout = NVBUF_LAYOUT_PITCH;
        params.memType = NVBUF_MEM_SURFACE_ARRAY;

        if (NvBufSurfaceCreate(&surf, 1, &params) != 0) return;

        unsigned long encode_size = 64 * 64 * 2; // Max possible size
        std::vector<uint8_t> out_buf(encode_size);
        unsigned char* out_ptr = out_buf.data();

        int fd = surf->surfaceList[0].bufferDesc;

        // Encode the blank hardware surface to JPEG bytes
        if (encoder->encodeFromFd(fd, JCS_YCbCr, &out_ptr, encode_size, 95) >= 0) {
            std::stringstream ss;
            ss << std::setw(3) << std::setfill('0') << index << "_" << suffix << ".jpg";
            fs::path file_path = test_dir_ / ss.str();
            std::ofstream out(file_path, std::ios::binary);
            out.write(reinterpret_cast<const char*>(out_ptr), encode_size);
        }

        NvBufSurfaceDestroy(surf);
    }

    // Generates a deterministic corrupt file (bypasses hardware initialization
    // to verify the skip logic works without triggering the SMMU)
    void CreateCorruptJpeg(int index, const std::string& suffix = "bad") {
        std::stringstream ss;
        ss << std::setw(3) << std::setfill('0') << index << "_" << suffix << ".jpg";
        fs::path file_path = test_dir_ / ss.str();
        std::ofstream out(file_path, std::ios::binary);
        out.write("This is a truncated/invalid JPEG header", 39);
    }

    void CreateDummyJpegs(int count) {
        for (int i = 0; i < count; ++i) {
            CreateValidJpeg(i);
        }
    }
};

// ==========================================
// 1. Initialization & Path Validation
// ==========================================

TEST_F(MMAPIJpegSourceTest, InitFailsOnInvalidDirectory) {
    std::unique_ptr<ISource> source;
    CudaError err = MMAPIJpegSource::Create(source, "/path/to/nowhere/that/does/not/exist", 64, 64, 4);

    EXPECT_TRUE(CudaError::IsFailure(err));
    EXPECT_NE(err.Text().find("Invalid folder path"), std::string::npos);
}

TEST_F(MMAPIJpegSourceTest, InitSucceedsOnEmptyDirectory) {
    std::unique_ptr<ISource> source;
    ASSERT_CUDA_SUCCESS(MMAPIJpegSource::Create(source, test_dir_.string(), 64, 64, 4));
    ASSERT_NE(source, nullptr);

    BatchData batch;
    bool process = true;
    ASSERT_CUDA_SUCCESS(source->GetNextBatch(batch, 4, process));
    EXPECT_FALSE(process) << "Should cleanly signal EOF on an empty directory";
}

// ==========================================
// 2. Batch Decoding Logic
// ==========================================

TEST_F(MMAPIJpegSourceTest, DecodesSingleBatchExactly) {
    int targetW = 64;
    int targetH = 64;
    int batchSize = 4;
    CreateDummyJpegs(batchSize);

    std::unique_ptr<ISource> source;
    ASSERT_CUDA_SUCCESS(MMAPIJpegSource::Create(source, test_dir_.string(), targetW, targetH, batchSize));

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

TEST_F(MMAPIJpegSourceTest, DecodesPartialBatchAtEOF) {
    CreateDummyJpegs(3);

    std::unique_ptr<ISource> source;
    ASSERT_CUDA_SUCCESS(MMAPIJpegSource::Create(source, test_dir_.string(), 64, 64, 8));

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

TEST_F(MMAPIJpegSourceTest, CorruptImageIsSkippedSafely) {
    // Generate: 000_good.jpg, 001_bad.jpg, 002_good.jpg
    CreateValidJpeg(0);
    CreateCorruptJpeg(1);
    CreateValidJpeg(2);

    std::unique_ptr<ISource> source;
    ASSERT_CUDA_SUCCESS(MMAPIJpegSource::Create(source, test_dir_.string(), 64, 64, 4));

    BatchData batch;
    bool process = false;

    // Request 4. Source sees 3 files. It parses them, finds 1 is bad.
    ASSERT_CUDA_SUCCESS(source->GetNextBatch(batch, 4, process));

    EXPECT_TRUE(process);
    EXPECT_EQ(batch.batchSize, 2) << "Must discard the corrupt image and yield the 2 valid ones";
    EXPECT_EQ(batch.width, 64);
    EXPECT_EQ(batch.height, 64);
    EXPECT_EQ(batch.sourceIdentifiers.size(), 2);
}

TEST_F(MMAPIJpegSourceTest, EntireCorruptBatchAdvancesToNextValidBatch) {
    CreateCorruptJpeg(0);
    CreateCorruptJpeg(1);
    CreateValidJpeg(2);

    std::unique_ptr<ISource> source;
    ASSERT_CUDA_SUCCESS(MMAPIJpegSource::Create(source, test_dir_.string(), 64, 64, 2));

    BatchData batch;
    bool process = false;

    int retries = 0;
    while (!process && retries < 5) {
        ASSERT_CUDA_SUCCESS(source->GetNextBatch(batch, 2, process));
        retries++;
    }

    EXPECT_TRUE(process) << "Must successfully fetch the next valid chunk after skipping bad chunk";
    EXPECT_EQ(batch.batchSize, 1) << "Must return the 1 valid image from Chunk 2";
    EXPECT_EQ(batch.width, 64);
    EXPECT_EQ(batch.height, 64);
}

TEST_F(MMAPIJpegSourceTest, CorruptImagesAtEOFYieldsNoProcess) {
    // Two completely bad images right at the end of the directory
    CreateCorruptJpeg(0);
    CreateCorruptJpeg(1);

    std::unique_ptr<ISource> source;
    ASSERT_CUDA_SUCCESS(MMAPIJpegSource::Create(source, test_dir_.string(), 64, 64, 4));

    BatchData batch;
    bool process = true; // Initialize to true to verify it correctly flips to false

    // Try to fetch. The hardware will skip both, find 0 valid frames, and return.
    ASSERT_CUDA_SUCCESS(source->GetNextBatch(batch, 4, process));

    // The next call will hit the EOF condition (frames_in_buffer_ == 0)
    ASSERT_CUDA_SUCCESS(source->GetNextBatch(batch, 4, process));

    EXPECT_FALSE(process) << "Pipeline must cleanly signal EOF if remaining images are corrupt";
    EXPECT_EQ(batch.batchSize, 0);
}

TEST_F(MMAPIJpegSourceTest, ExercisesDoubleBufferingWrapAround) {
    int batchSize = 2;
    CreateDummyJpegs(6); // 3 full batches: (0,1), (2,3), (4,5)

    std::unique_ptr<ISource> source;
    ASSERT_CUDA_SUCCESS(MMAPIJpegSource::Create(source, test_dir_.string(), 64, 64, batchSize));

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

TEST_F(MMAPIJpegSourceTest, PartialBatchMaintainsStrictEngineCapacityAndZeroPads) {
    CreateValidJpeg(0);
    CreateValidJpeg(1);
    CreateCorruptJpeg(2);

    int targetW = 64;
    int targetH = 64;
    int reqBatchSize = 4;

    std::unique_ptr<ISource> source;
    ASSERT_CUDA_SUCCESS(MMAPIJpegSource::Create(source, test_dir_.string(), targetW, targetH, reqBatchSize));

    BatchData batch;
    bool process = false;

    ASSERT_CUDA_SUCCESS(source->GetNextBatch(batch, reqBatchSize, process));

    EXPECT_TRUE(process);
    EXPECT_EQ(batch.batchSize, 2);

    // Verify Strict Engine Capacity
    size_t expectedTotalElements = reqBatchSize * targetW * targetH * 3;
    EXPECT_EQ(batch.deviceData.size(), expectedTotalElements);

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
