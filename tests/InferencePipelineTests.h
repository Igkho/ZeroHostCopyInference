#pragma once
#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <atomic>
#include <set>
#include <mutex>
#include <thread>
#include "InferencePipeline.h"
#include "Interfaces.h"
#include "BatchData.h"
#include "BatchDetections.h"
#include "helpers.h"

namespace cropandweed {

// Shared context to verify results after the Pipeline takes ownership of the pointers
struct TestContext {
    std::atomic<int> sourceCalls{0};
    std::atomic<int> detectorCalls{0};
    std::atomic<int> sinkCalls{0};

    // For recycling verification
    std::mutex mtx;
    std::set<float*> seenPointers;
    int reuseCount = 0;

    int maxBatches = 10;
    int batchSize = 2;
    int width = 128;
    int height = 128;

    // Simulation timing
    int sourceDelayMs = 0;
};

// --- MOCK COMPONENTS ---

class MockSource : public ISource {
public:
    explicit MockSource(std::shared_ptr<TestContext> ctx) : ctx_(ctx) {}

    CudaError GetNextBatch(BatchData& outBatch, size_t batchSize, bool &process) override {
        // Simulate decoding latency (critical for recycling test)
        if (ctx_->sourceDelayMs > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(ctx_->sourceDelayMs));
        }

        int current = ctx_->sourceCalls.fetch_add(1);

        if (current >= ctx_->maxBatches) {
            process = false;
            return CudaError();
        }

        // Simulate Frame Data
        // If outBatch comes from the RecycleQueue, Init() will check capacity.
        // If it's sufficient, it won't realloc.
        CUDA_TRY(outBatch.Init(ctx_->batchSize, ctx_->width, ctx_->height));
        outBatch.batchId = current;

        // Track the GPU pointer to verify recycling
        {
            std::lock_guard<std::mutex> lock(ctx_->mtx);
            float* ptr = outBatch.deviceData.data();
            if (ptr) {
                if (ctx_->seenPointers.count(ptr)) {
                    ctx_->reuseCount++;
                } else {
                    ctx_->seenPointers.insert(ptr);
                }
            }
        }

        // Simulate IDs
        outBatch.sourceIdentifiers.clear();
        for(size_t i=0; i<outBatch.batchSize; ++i) {
            outBatch.sourceIdentifiers.push_back("mock_" + std::to_string(current) + "_" + std::to_string(i));
        }

        // Record event (Simulate GPU work done)
        if (outBatch.readyEvent) {
            cudaEventRecord(*outBatch.readyEvent, 0);
        }

        process = true;
        return CudaError();
    }

    void SetOutputSize(size_t width, size_t height) override {
        ctx_->width = (int)width;
        ctx_->height = (int)height;
    }

private:
    std::shared_ptr<TestContext> ctx_;
};

class MockDetector : public IDetector {
public:
    explicit MockDetector(std::shared_ptr<TestContext> ctx) : ctx_(ctx) {}

    CudaError Detect(const BatchData& input, BatchDetections &output) override {
        ctx_->detectorCalls++;

        // Initialize Output
        CUDA_TRY(output.Init(input.batchSize));

        // Wait for input (Pipeline logic check)
        if (input.readyEvent) {
            cudaEventSynchronize(*input.readyEvent);
        }

        // Signal Output Ready
        if (output.readyEvent) {
            cudaEventRecord(*output.readyEvent, 0);
        }

        return CudaError();
    }

    std::pair<size_t, size_t> GetInputSize() const override {
        return { (size_t)ctx_->width, (size_t)ctx_->height };
    }

private:
    std::shared_ptr<TestContext> ctx_;
};

class MockSink : public ISink {
public:
    explicit MockSink(std::shared_ptr<TestContext> ctx) : ctx_(ctx) {}

    CudaError Save(const BatchData& batch, const BatchDetections& results) override {
        ctx_->sinkCalls++;

        // Verify Sync logic
        if (results.readyEvent) {
            cudaEventSynchronize(*results.readyEvent);
        }

        return CudaError();
    }

private:
    std::shared_ptr<TestContext> ctx_;
};

// --- TEST SUITE ---

class InferencePipelineTest : public ::testing::Test {
protected:
    std::shared_ptr<TestContext> ctx_;

    void SetUp() override {
        ctx_ = std::make_shared<TestContext>();
        // Default Settings
        ctx_->maxBatches = 20;
        ctx_->batchSize = 4;
        ctx_->sourceDelayMs = 0;
    }

    void TearDown() override {
        cudaGetLastError();
    }
};

TEST_F(InferencePipelineTest, EndToEndFlow) {
    // 1. Create Mocks with Shared Context
    auto src = std::make_unique<MockSource>(ctx_);
    auto det = std::make_unique<MockDetector>(ctx_);
    auto sink = std::make_unique<MockSink>(ctx_);

    // 2. Create Pipeline
    InferencePipeline pipeline(std::move(src), std::move(det), std::move(sink), ctx_->batchSize);

    // 3. Run
    CudaError err = pipeline.Run();

    // 4. Verify
    EXPECT_FALSE(CudaError::IsFailure(err));

    // Source is called 21 times (20 valid + 1 returning false)
    EXPECT_EQ(ctx_->sourceCalls.load(), 21);
    EXPECT_EQ(ctx_->detectorCalls.load(), 20);
    EXPECT_EQ(ctx_->sinkCalls.load(), 20);
}

TEST_F(InferencePipelineTest, BufferRecyclingWorks) {
    ctx_->maxBatches = 50;

    // Slow down the source slightly (1 ms) to allow
    // downstream threads (Detector/Sink) to finish and recycle buffers.
    // Without this, the unchecked producer fills memory before recycling occurs.
    ctx_->sourceDelayMs = 1;

    auto src = std::make_unique<MockSource>(ctx_);
    auto det = std::make_unique<MockDetector>(ctx_);
    auto sink = std::make_unique<MockSink>(ctx_);

    InferencePipeline pipeline(std::move(src), std::move(det), std::move(sink), ctx_->batchSize);
    pipeline.Run();

    // Verify Reuse
    {
        std::lock_guard<std::mutex> lock(ctx_->mtx);
        size_t uniqueAllocations = ctx_->seenPointers.size();

        std::cout << "[Test] Unique GPU Buffers allocated: " << uniqueAllocations
                  << " for " << ctx_->maxBatches << " batches." << std::endl;

        // With steady state recycling, we expect ~3-5 buffers max.
        // We set 15 as a very safe upper bound.
        EXPECT_LT(uniqueAllocations, 15);
        EXPECT_GT(ctx_->reuseCount, 30);
    }
}

TEST_F(InferencePipelineTest, HandlesEmptySource) {
    ctx_->maxBatches = 0; // Immediate EOF

    auto src = std::make_unique<MockSource>(ctx_);
    auto det = std::make_unique<MockDetector>(ctx_);
    auto sink = std::make_unique<MockSink>(ctx_);

    InferencePipeline pipeline(std::move(src), std::move(det), std::move(sink), ctx_->batchSize);

    pipeline.Run();

    EXPECT_EQ(ctx_->sourceCalls.load(), 1); // 1 call -> returns false
    EXPECT_EQ(ctx_->detectorCalls.load(), 0);
    EXPECT_EQ(ctx_->sinkCalls.load(), 0);
}

} // namespace cropandweed
