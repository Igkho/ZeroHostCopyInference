#pragma once
#include <gtest/gtest.h>
#include <vector>
#include <numeric>
#include <algorithm>
#include "helpers.h"
#include "Block.h"

namespace cropandweed {

// Helper to assert CUDA success within GTest
#ifndef ASSERT_CUDA_SUCCESS
#define ASSERT_CUDA_SUCCESS(err) ASSERT_FALSE(CudaError::IsFailure(err)) << (err).Text()
#endif

class BlockTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Optional: Reset device or verify state if needed
    }

    void TearDown() override {
        // Optional: Cleanup
    }
};

// --- 1. Basic Lifecycle & Factories ---

TEST_F(BlockTest, CreateEmpty) {
    std::unique_ptr<Block<float>> block;
    // Create size 0
    ASSERT_CUDA_SUCCESS(Block<float>::Create(block, 0));

    ASSERT_NE(block, nullptr);
    EXPECT_EQ(block->size(), 0);
    EXPECT_EQ(block->capacity(), 0);
    EXPECT_TRUE(block->empty());
    // Data pointer might be null or valid but empty, usually null for cap 0
    EXPECT_EQ(block->data(), nullptr);
}

TEST_F(BlockTest, CreateWithSize) {
    std::unique_ptr<Block<int>> block;
    size_t size = 100;
    ASSERT_CUDA_SUCCESS(Block<int>::Create(block, size));

    ASSERT_NE(block, nullptr);
    EXPECT_EQ(block->size(), size);
    EXPECT_GE(block->capacity(), size);
    EXPECT_FALSE(block->empty());
    EXPECT_NE(block->data(), nullptr);
}

TEST_F(BlockTest, CreateWithValue) {
    std::unique_ptr<Block<uint8_t>> block;
    size_t size = 10;
    int val = 0xFF; // 255
    ASSERT_CUDA_SUCCESS(Block<uint8_t>::Create(block, size, val));

    // verify on host
    std::vector<uint8_t> host_vec;
    ASSERT_CUDA_SUCCESS(block->to_vector(host_vec));

    ASSERT_EQ(host_vec.size(), size);
    for (uint8_t v : host_vec) {
        EXPECT_EQ(v, 255);
    }
}

TEST_F(BlockTest, CreateFromVector) {
    std::vector<float> input = {1.1f, 2.2f, 3.3f, 4.4f, 5.5f};
    std::unique_ptr<Block<float>> block;

    ASSERT_CUDA_SUCCESS(Block<float>::Create(block, input));

    EXPECT_EQ(block->size(), input.size());

    // Round trip back to check data integrity
    std::vector<float> output;
    ASSERT_CUDA_SUCCESS(block->to_vector(output));

    EXPECT_EQ(input, output);
}

// --- 2. Move Semantics ---

TEST_F(BlockTest, MoveConstructor) {
    std::vector<int> data = {1, 2, 3};
    std::unique_ptr<Block<int>> original;
    ASSERT_CUDA_SUCCESS(Block<int>::Create(original, data));

    // Perform Move
    Block<int> moved_block(std::move(*original));

    // Verify Original is empty/null
    EXPECT_EQ(original->size(), 0);
    EXPECT_EQ(original->data(), nullptr);

    // Verify Moved has data
    EXPECT_EQ(moved_block.size(), 3);

    std::vector<int> check;
    ASSERT_CUDA_SUCCESS(moved_block.to_vector(check));
    EXPECT_EQ(check, data);
}

TEST_F(BlockTest, MoveAssignment) {
    std::vector<int> data = {10, 20};
    std::unique_ptr<Block<int>> source;
    ASSERT_CUDA_SUCCESS(Block<int>::Create(source, data));

    std::unique_ptr<Block<int>> dest;
    ASSERT_CUDA_SUCCESS(Block<int>::Create(dest, 100)); // Dest has existing garbage

    *dest = std::move(*source);

    EXPECT_EQ(source->size(), 0); // Source stripped
    EXPECT_EQ(source->data(), nullptr);
    EXPECT_EQ(dest->size(), 2);   // Dest took ownership

    std::vector<int> check;
    ASSERT_CUDA_SUCCESS(dest->to_vector(check));
    EXPECT_EQ(check, data);
}

// --- 3. Data Transfer & Assignment ---

TEST_F(BlockTest, AssignFromVector) {
    std::unique_ptr<Block<int>> block;
    ASSERT_CUDA_SUCCESS(Block<int>::Create(block, 5)); // Initial size 5

    std::vector<int> larger_data(20, 7); // Vector size 20

    // Assign larger vector (trigger reallocation)
    ASSERT_CUDA_SUCCESS(block->assign(larger_data));

    EXPECT_EQ(block->size(), 20);

    std::vector<int> out;
    ASSERT_CUDA_SUCCESS(block->to_vector(out));
    EXPECT_EQ(out, larger_data);
}

TEST_F(BlockTest, AssignFromBlock) {
    std::vector<int> data = {1, 2, 3, 4};
    std::unique_ptr<Block<int>> src;
    std::unique_ptr<Block<int>> dst;

    ASSERT_CUDA_SUCCESS(Block<int>::Create(src, data));
    ASSERT_CUDA_SUCCESS(Block<int>::Create(dst, 1)); // tiny dest

    ASSERT_CUDA_SUCCESS(dst->assign(*src));

    EXPECT_EQ(dst->size(), 4);

    std::vector<int> out;
    ASSERT_CUDA_SUCCESS(dst->to_vector(out));
    EXPECT_EQ(out, data);
}

// --- 4. Modifiers (Resize, Reserve, Clear) ---

TEST_F(BlockTest, Reserve) {
    std::unique_ptr<Block<double>> block;
    ASSERT_CUDA_SUCCESS(Block<double>::Create(block, 0));

    // Reserve shouldn't change size, only capacity
    ASSERT_CUDA_SUCCESS(block->reserve(100));
    EXPECT_EQ(block->size(), 0);
    EXPECT_GE(block->capacity(), 100);

    // Ensure we can fill it without reallocation (logic check only)
    ASSERT_CUDA_SUCCESS(block->resize(50));
    EXPECT_EQ(block->size(), 50);
    EXPECT_GE(block->capacity(), 100);
}

TEST_F(BlockTest, ResizeShrink) {
    std::vector<int> data = {10, 20, 30, 40, 50};
    std::unique_ptr<Block<int>> block;
    ASSERT_CUDA_SUCCESS(Block<int>::Create(block, data));

    // Shrink to 3
    ASSERT_CUDA_SUCCESS(block->resize(3));
    EXPECT_EQ(block->size(), 3);

    std::vector<int> out;
    ASSERT_CUDA_SUCCESS(block->to_vector(out));

    std::vector<int> expected = {10, 20, 30};
    EXPECT_EQ(out, expected);
}

TEST_F(BlockTest, ResizeGrowNoVal) {
    std::vector<int> data = {1, 2};
    std::unique_ptr<Block<int>> block;
    ASSERT_CUDA_SUCCESS(Block<int>::Create(block, data));

    // Grow to 4. New elements are uninitialized (we can't strictly check their value,
    // but we check old data is preserved).
    ASSERT_CUDA_SUCCESS(block->resize(4));
    EXPECT_EQ(block->size(), 4);

    std::vector<int> out;
    ASSERT_CUDA_SUCCESS(block->to_vector(out));

    EXPECT_EQ(out[0], 1);
    EXPECT_EQ(out[1], 2);
}

TEST_F(BlockTest, ResizeGrowWithVal) {
    std::vector<uint8_t> data = {100};
    std::unique_ptr<Block<uint8_t>> block;
    ASSERT_CUDA_SUCCESS(Block<uint8_t>::Create(block, data));

    // Grow to 4, fill new with 0xFF (255)
    ASSERT_CUDA_SUCCESS(block->resize(4, 0xFF));

    std::vector<uint8_t> out;
    ASSERT_CUDA_SUCCESS(block->to_vector(out));

    std::vector<uint8_t> expected = {100, 255, 255, 255};
    EXPECT_EQ(out, expected);
}

TEST_F(BlockTest, Clear) {
    std::unique_ptr<Block<int>> block;
    ASSERT_CUDA_SUCCESS(Block<int>::Create(block, 10));

    size_t cap = block->capacity();
    block->clear();

    EXPECT_EQ(block->size(), 0);
    EXPECT_EQ(block->capacity(), cap); // Capacity should remain
    EXPECT_TRUE(block->empty());
}

TEST_F(BlockTest, Swap) {
    std::unique_ptr<Block<int>> b1, b2;
    std::vector<int> v1 = {1, 1, 1};
    std::vector<int> v2 = {2, 2};

    ASSERT_CUDA_SUCCESS(Block<int>::Create(b1, v1));
    ASSERT_CUDA_SUCCESS(Block<int>::Create(b2, v2));

    b1->swap(*b2);

    EXPECT_EQ(b1->size(), 2);
    EXPECT_EQ(b2->size(), 3);

    std::vector<int> out1, out2;
    b1->to_vector(out1);
    b2->to_vector(out2);

    EXPECT_EQ(out1, v2);
    EXPECT_EQ(out2, v1);
}

// --- 5. Types & Memory Templates (Edge Cases) ---

TEST_F(BlockTest, LargeAllocation_Uint8) {
    // Testing a larger block of bytes (e.g., image buffer)
    std::unique_ptr<Block<uint8_t>> block;
    // Allocate ~1MB
    size_t size = 1024 * 1024;
    // Fill with 0x00 is safe for memset
    ASSERT_CUDA_SUCCESS(Block<uint8_t>::Create(block, size, static_cast<uint8_t>(0)));

    EXPECT_EQ(block->size(), size);
    EXPECT_EQ(block->byte_size(), size);

    // Sample check last element
    std::vector<uint8_t> out(1);
    // Copy just the last byte to host manually to verify pointer arithmetic access works internally
    cudaMemcpy(out.data(), block->data() + size - 1, 1, cudaMemcpyDeviceToHost);
    EXPECT_EQ(out[0], 0);
}

TEST_F(BlockTest, PinnedMemoryInstantiation) {
    // Verify Pinned memory template works
    std::unique_ptr<Block<uint8_t, MemoryType::Pinned>> block;
    // Use value 5. Since sizeof(uint8_t) == 1, memset(..., 5) results in value 5.
    ASSERT_CUDA_SUCCESS((Block<uint8_t, MemoryType::Pinned>::Create(block, 10, 5)));

    EXPECT_EQ(block->size(), 10);

    std::vector<uint8_t> out;
    ASSERT_CUDA_SUCCESS(block->to_vector(out));
    EXPECT_EQ(out[0], 5);
}

TEST_F(BlockTest, ZeroCopyInstantiation) {
    // Added: ZeroCopy verification for uint8_t
    std::unique_ptr<Block<uint8_t, MemoryType::ZeroCopy>> block;
    ASSERT_CUDA_SUCCESS((Block<uint8_t, MemoryType::ZeroCopy>::Create(block, 10, 128)));

    EXPECT_EQ(block->size(), 10);

    std::vector<uint8_t> out;
    ASSERT_CUDA_SUCCESS(block->to_vector(out));
    EXPECT_EQ(out[0], 128);
}

// --- 6. Edge Case: Self-Assignment ---

TEST_F(BlockTest, AssignSelf) {
    // While move-self is protected in the code, verify logic holds for standard assign
    // (The API takes a const reference, so copy_from handles the logic)
    std::vector<int> data = {1, 2, 3};
    std::unique_ptr<Block<int>> block;
    ASSERT_CUDA_SUCCESS(Block<int>::Create(block, data));

    // Assign block to itself
    ASSERT_CUDA_SUCCESS(block->assign(*block));

    EXPECT_EQ(block->size(), 3);
    std::vector<int> out;
    block->to_vector(out);
    EXPECT_EQ(out, data);
}

// --- 7. Async Stream Operations ---

TEST_F(BlockTest, AsyncCreateAndFill) {
    // 1. Setup Stream
    std::unique_ptr<CudaStream> streamPtr;
    ASSERT_CUDA_SUCCESS(CudaStream::Create(streamPtr));
    cudaStream_t stream = *streamPtr;

    // 2. Async Creation
    std::unique_ptr<Block<int>> block;
    // Create size 100, fill with 77, on specific stream
    ASSERT_CUDA_SUCCESS(Block<int>::Create(block, 100, 77, stream));

    // 3. Verify
    std::vector<int> out;
    // to_vector syncs internally, ensuring we wait for the async fill
    ASSERT_CUDA_SUCCESS(block->to_vector(out, stream));

    ASSERT_EQ(out.size(), 100);
    for (int v : out) {
        EXPECT_EQ(v, 77);
    }
}

TEST_F(BlockTest, AsyncResizePreservesData) {
    // This tests the critical path: Malloc -> Async Copy Old -> Async Memset New -> Swap
    std::unique_ptr<CudaStream> streamPtr;
    ASSERT_CUDA_SUCCESS(CudaStream::Create(streamPtr));
    cudaStream_t stream = *streamPtr;

    std::vector<int> initial_data = {10, 20, 30};
    std::unique_ptr<Block<int>> block;
    ASSERT_CUDA_SUCCESS(Block<int>::Create(block, initial_data, stream));

    // Resize to 10, fill new slots with 99
    ASSERT_CUDA_SUCCESS(block->resize(10, 99, stream));

    std::vector<int> out;
    ASSERT_CUDA_SUCCESS(block->to_vector(out, stream));

    EXPECT_EQ(out.size(), 10);
    // Check old data preserved
    EXPECT_EQ(out[0], 10);
    EXPECT_EQ(out[1], 20);
    EXPECT_EQ(out[2], 30);
    // Check new data filled
    for (size_t i = 3; i < 10; ++i) {
        EXPECT_EQ(out[i], 99);
    }
}

TEST_F(BlockTest, AsyncBlockToBlockCopy) {
    std::unique_ptr<CudaStream> streamPtr;
    ASSERT_CUDA_SUCCESS(CudaStream::Create(streamPtr));
    cudaStream_t stream = *streamPtr;

    std::vector<float> data(1000, 123.456f);
    std::unique_ptr<Block<float>> src;
    std::unique_ptr<Block<float>> dst;

    ASSERT_CUDA_SUCCESS(Block<float>::Create(src, data, stream));
    ASSERT_CUDA_SUCCESS(Block<float>::Create(dst, 0, stream)); // Empty dst

    // Async assignment between device blocks
    ASSERT_CUDA_SUCCESS(dst->assign(*src, stream));

    std::vector<float> out;
    ASSERT_CUDA_SUCCESS(dst->to_vector(out, stream));

    EXPECT_EQ(out, data);
}

TEST_F(BlockTest, AsyncPinnedMemoryCompatibility) {
    // Pinned memory often falls back to synchronous CPU memset/memcpy.
    // We must ensure passing a stream doesn't break compilation or runtime logic.
    std::unique_ptr<CudaStream> streamPtr;
    ASSERT_CUDA_SUCCESS(CudaStream::Create(streamPtr));
    cudaStream_t stream = *streamPtr;

    std::unique_ptr<Block<uint8_t, MemoryType::Pinned>> block;
    ASSERT_CUDA_SUCCESS((Block<uint8_t, MemoryType::Pinned>::Create(block, 50, stream)));

    // Resize with value on stream
    ASSERT_CUDA_SUCCESS(block->resize(100, 0xAA, stream));

    std::vector<uint8_t> out;
    ASSERT_CUDA_SUCCESS(block->to_vector(out, stream));

    EXPECT_EQ(out.size(), 100);
    EXPECT_EQ(out[99], 0xAA);
}

// Added explicit coverage for the TypedBlock<T> type-safe adapter
TEST_F(BlockTest, TypedBlock_LifecycleAndSizing) {
    TypedBlock<float> tb;
    EXPECT_EQ(tb.size(), 0);
    EXPECT_EQ(tb.capacity(), 0);

    // Resize by element count, not bytes
    ASSERT_CUDA_SUCCESS(tb.resize(100));
    EXPECT_EQ(tb.size(), 100);
    EXPECT_EQ(tb.byte_size(), 100 * sizeof(float));
    EXPECT_GE(tb.capacity(), 100);

    // Reserve additional capacity
    ASSERT_CUDA_SUCCESS(tb.reserve(200));
    EXPECT_EQ(tb.size(), 100);
    EXPECT_GE(tb.capacity(), 200);
}

// Verify type-safe Host <-> Device data transfer logic
TEST_F(BlockTest, TypedBlock_DataTransfer) {
    std::vector<int> host_in = {42, 73, 108};
    TypedBlock<int> tb;

    ASSERT_CUDA_SUCCESS(tb.assign(host_in));
    EXPECT_EQ(tb.size(), 3);
    EXPECT_NE(tb.data(), nullptr);

    std::vector<int> host_out;
    ASSERT_CUDA_SUCCESS(tb.to_vector(host_out)); // Implicitly synchronizes
    EXPECT_EQ(host_in, host_out);
}

// Guarantee move semantics don't leak or leave dangling raw pointers
TEST_F(BlockTest, TypedBlock_MoveSemantics) {
    std::vector<double> data = {1.1, 2.2, 3.3};
    TypedBlock<double> src;
    ASSERT_CUDA_SUCCESS(src.assign(data));

    double* original_ptr = src.data();

    // Move construct
    TypedBlock<double> dst(std::move(src));
    EXPECT_EQ(src.size(), 0);
    EXPECT_EQ(src.data(), nullptr);
    EXPECT_EQ(dst.size(), 3);
    EXPECT_EQ(dst.data(), original_ptr);

    // Move assignment
    TypedBlock<double> dst2;
    dst2 = std::move(dst);
    EXPECT_EQ(dst.data(), nullptr);
    EXPECT_EQ(dst2.data(), original_ptr);
}

// Validate raw byte-block access for specialized casting scenarios
TEST_F(BlockTest, TypedBlock_RawBlockAccess) {
    TypedBlock<int> tb;
    ASSERT_CUDA_SUCCESS(tb.resize(10));

    Block<uint8_t>& raw = tb.raw();
    EXPECT_EQ(raw.size(), 10 * sizeof(int));
    EXPECT_EQ((void*)raw.data(), (void*)tb.data());
}

// Test previously uncovered iterator and element access operators
TEST_F(BlockTest, Block_IteratorsAndElementAccess) {
    std::vector<int> data = {10, 20, 30};
    std::unique_ptr<Block<int, MemoryType::Pinned>> pinned_block;
    ASSERT_CUDA_SUCCESS((Block<int, MemoryType::Pinned>::Create(pinned_block, data)));

    // Since it's pinned, host can safely access the pointers directly
    EXPECT_EQ((*pinned_block)[0], 10);
    EXPECT_EQ((*pinned_block)[2], 30);

    // Test Iterator arithmetic
    EXPECT_EQ(std::distance(pinned_block->begin(), pinned_block->end()), 3);
    EXPECT_EQ(std::distance(pinned_block->cbegin(), pinned_block->cend()), 3);

    int sum = 0;
    for (auto it = pinned_block->begin(); it != pinned_block->end(); ++it) {
        sum += *it;
    }
    EXPECT_EQ(sum, 60);
}

// Verifies that resize with a value acts as a bulk fill
TEST_F(BlockTest, ResizeActsAsBulkFill) {
    std::unique_ptr<Block<int>> block;
    ASSERT_CUDA_SUCCESS(Block<int>::Create(block, 0));

    // Resize to 5 and fill with 42
    ASSERT_CUDA_SUCCESS(block->resize(5, 42));

    std::vector<int> out;
    ASSERT_CUDA_SUCCESS(block->to_vector(out));
    for (int v : out) {
        EXPECT_EQ(v, 42);
    }
}

// Verifies that host-accessible memory types can be accessed directly
TEST_F(BlockTest, DirectAccess_PinnedMemory) {
    std::unique_ptr<Block<float, MemoryType::Pinned>> block;
    ASSERT_CUDA_SUCCESS((Block<float, MemoryType::Pinned>::Create(block, 3, 0.0f)));

    // Directly access and mutate via host pointer
    float* host_ptr = block->data();
    host_ptr[1] = 3.14f;

    // Verify via the const accessor
    EXPECT_FLOAT_EQ((*block)[1], 3.14f);
}

TEST_F(BlockTest, TypedBlock_FillZero) {
    std::unique_ptr<CudaStream> streamPtr;
    ASSERT_CUDA_SUCCESS(CudaStream::Create(streamPtr));
    cudaStream_t stream = *streamPtr;

    std::vector<float> initial_data = {1.5f, 2.5f, 3.5f, 4.5f};
    TypedBlock<float> tb;
    ASSERT_CUDA_SUCCESS(tb.assign(initial_data, stream));

    // Execute zero-fill
    ASSERT_CUDA_SUCCESS(tb.fill_zero(stream));

    std::vector<float> out;
    ASSERT_CUDA_SUCCESS(tb.to_vector(out, stream));

    EXPECT_EQ(out.size(), 4);
    for (float v : out) {
        EXPECT_FLOAT_EQ(v, 0.0f);
    }
}

} // namespace cropandweed
