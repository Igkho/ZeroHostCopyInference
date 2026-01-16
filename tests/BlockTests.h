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
    ASSERT_CUDA_SUCCESS(Block<uint8_t>::Create(block, size, 0));

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

} // namespace cropandweed
