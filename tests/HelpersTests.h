#pragma once
#include <gtest/gtest.h>
#include <memory>
#include <cuda_runtime.h>
#include "helpers.h"

namespace cropandweed {

// --- Macro Definition ---
// Define the assertion macro locally for this test suite
#ifndef ASSERT_CUDA_SUCCESS
#define ASSERT_CUDA_SUCCESS(err) ASSERT_FALSE(CudaError::IsFailure(err)) << (err).Text()
#endif

// --- 1. CudaError Tests ---

TEST(CudaErrorTest, SuccessDoesNotReportFailure) {
    CudaError err;
    EXPECT_FALSE(CudaError::IsFailure(err));
    EXPECT_EQ(err.Text(), "No GPU errors");
}

TEST(CudaErrorTest, ReportsCudaFailure) {
    // Simulate a CUDA error
    cudaError_t raw_err = cudaErrorMemoryAllocation;
    CudaError err("TestLocation", raw_err);
    
    EXPECT_TRUE(CudaError::IsFailure(err));
    EXPECT_NE(err.Text().find("TestLocation"), std::string::npos);
    EXPECT_NE(err.Text().find("out of memory"), std::string::npos);
}

TEST(CudaErrorTest, ReportsNvjpegFailure) {
    nvjpegStatus_t raw_err = NVJPEG_STATUS_INTERNAL_ERROR;
    EXPECT_TRUE(CudaError::IsFailure(raw_err));
    EXPECT_EQ(CudaError::GetErrorString(raw_err), "Internal Error");
}

TEST(CudaErrorTest, ErrorChaining) {
    CudaError root("Root", cudaErrorInitializationError);
    CudaError chained("Chain", root);

    EXPECT_TRUE(CudaError::IsFailure(chained));
    std::string text = chained.Text();
    // Check that both levels of the stack are present
    EXPECT_NE(text.find("Root"), std::string::npos);
    EXPECT_NE(text.find("Chain"), std::string::npos);
    EXPECT_NE(text.find("initialization error"), std::string::npos);
}

TEST(CudaErrorTest, ReportsStringFailure) {
    // Test the logic used by CUDA_GENERAL_ERROR
    std::string msg = "Logical failure occurred";
    CudaError err("TestLoc", msg);
    
    EXPECT_TRUE(CudaError::IsFailure(err));
    EXPECT_NE(err.Text().find(msg), std::string::npos);
}

// --- 2. CudaStream Tests ---

TEST(CudaStreamTest, CreateUniquePtr) {
    std::unique_ptr<CudaStream> stream;
    CudaError err = CudaStream::Create(stream);
    
    EXPECT_FALSE(CudaError::IsFailure(err));
    EXPECT_NE(stream, nullptr);
    EXPECT_NE((cudaStream_t)*stream, nullptr);
}

TEST(CudaStreamTest, CreateSharedPtr) {
    std::shared_ptr<CudaStream> stream;
    CudaError err = CudaStream::Create(stream);
    
    EXPECT_FALSE(CudaError::IsFailure(err));
    EXPECT_NE(stream, nullptr);
    EXPECT_NE((cudaStream_t)*stream, nullptr);
}

TEST(CudaStreamTest, CreateWithFlags) {
    std::unique_ptr<CudaStream> stream;
    // Create non-blocking stream
    CudaError err = CudaStream::Create(stream, cudaStreamNonBlocking);
    
    EXPECT_FALSE(CudaError::IsFailure(err));
    EXPECT_NE(stream, nullptr);
    
    // Verify validity by doing a dummy operation
    cudaStream_t raw = *stream;
    EXPECT_EQ(cudaStreamQuery(raw), cudaSuccess);
}

TEST(CudaStreamTest, MoveSemantics) {
    std::unique_ptr<CudaStream> stream1;
    CudaStream::Create(stream1);
    cudaStream_t raw1 = *stream1;

    // Move stream1 to stream2 (manual move just to test class mechanics, 
    // usually we move the unique_ptr itself)
    CudaStream stream2(std::move(*stream1));
    
    EXPECT_EQ((cudaStream_t)stream2, raw1);
    // stream1's internal handle should be null now (though unique_ptr is still valid object)
    EXPECT_EQ((cudaStream_t)*stream1, nullptr);
}

TEST(CudaStreamTest, MoveAssignment) {
    // Create two valid streams
    std::unique_ptr<CudaStream> s1, s2;
    ASSERT_CUDA_SUCCESS(CudaStream::Create(s1));
    ASSERT_CUDA_SUCCESS(CudaStream::Create(s2));

    cudaStream_t handle1 = *s1;
    cudaStream_t handle2 = *s2;

    // Assign s1 to s2. 
    // s2's original stream (handle2) should be destroyed.
    // s2 should now hold handle1.
    // s1 should be null/empty.
    *s2 = std::move(*s1);

    EXPECT_EQ((cudaStream_t)*s2, handle1);
    EXPECT_EQ((cudaStream_t)*s1, nullptr);
    
    // We can't easily verify handle2 was destroyed without mocking, 
    // but we can verify handle1 is still valid by using it.
    EXPECT_EQ(cudaStreamQuery(*s2), cudaSuccess);
}

// --- 3. CudaEvent Tests ---

TEST(CudaEventTest, CreateUniquePtr) {
    std::unique_ptr<CudaEvent> event;
    CudaError err = CudaEvent::Create(event);
    
    EXPECT_FALSE(CudaError::IsFailure(err));
    EXPECT_NE(event, nullptr);
    EXPECT_NE((cudaEvent_t)*event, nullptr);
}

TEST(CudaEventTest, CreateSharedPtr) {
    // Parity with CudaStreamTest.CreateSharedPtr
    std::shared_ptr<CudaEvent> event;
    ASSERT_CUDA_SUCCESS(CudaEvent::Create(event));
    
    EXPECT_NE(event, nullptr);
    EXPECT_NE((cudaEvent_t)*event, nullptr);
}

TEST(CudaEventTest, CreateWithFlags) {
    // Parity with CudaStreamTest.CreateWithFlags
    std::unique_ptr<CudaEvent> event;
    // Use cudaEventBlockingSync to ensure flags are respected
    ASSERT_CUDA_SUCCESS(CudaEvent::Create(event, cudaEventBlockingSync));
    
    EXPECT_NE(event, nullptr);
    EXPECT_NE((cudaEvent_t)*event, nullptr);
}

TEST(CudaEventTest, MoveConstructor) {
    // Parity with CudaStreamTest.MoveSemantics
    std::unique_ptr<CudaEvent> e1;
    ASSERT_CUDA_SUCCESS(CudaEvent::Create(e1));
    cudaEvent_t raw1 = *e1;

    // Perform explicit Move Construction
    CudaEvent e2(std::move(*e1));
    
    EXPECT_EQ((cudaEvent_t)e2, raw1);
    EXPECT_EQ((cudaEvent_t)*e1, nullptr);
}

TEST(CudaEventTest, MoveAssignment) {
    // 1. Create two valid events
    std::unique_ptr<CudaEvent> e1, e2;
    ASSERT_CUDA_SUCCESS(CudaEvent::Create(e1));
    ASSERT_CUDA_SUCCESS(CudaEvent::Create(e2));

    cudaEvent_t handle1 = *e1;
    // handle2 (inside e2) will be destroyed during the move assignment below

    // 2. Assign e1 to e2. 
    // e2's original event resource is destroyed.
    // e2 takes ownership of handle1.
    // e1 is left in a null/empty state.
    *e2 = std::move(*e1);

    EXPECT_EQ((cudaEvent_t)*e2, handle1);
    EXPECT_EQ((cudaEvent_t)*e1, nullptr);
    
    // 3. Verify handle1 is still valid by using it via e2
    // cudaEventQuery returns cudaSuccess (done) or cudaErrorNotReady (pending), 
    // but ensures the handle itself is valid.
    cudaError_t status = cudaEventQuery(*e2);
    EXPECT_TRUE(status == cudaSuccess || status == cudaErrorNotReady);
}

TEST(CudaEventTest, RecordAndQuery) {
    std::unique_ptr<CudaStream> stream;
    CudaStream::Create(stream);
    
    std::unique_ptr<CudaEvent> event;
    CudaEvent::Create(event);

    cudaEventRecord(*event, *stream);
    cudaError_t status = cudaEventQuery(*event);
    
    // Status should be either Success (done) or NotReady (pending), but not an error
    EXPECT_TRUE(status == cudaSuccess || status == cudaErrorNotReady);
}

// --- 4. KernelGrid Tests ---

TEST(KernelGridTest, Calculate1D) {
    unsigned int total_threads = 1000;
    unsigned int block_size = 256;
    
    KernelGrid grid(total_threads, block_size);
    
    EXPECT_EQ(grid.bsize().x, 256);
    EXPECT_EQ(grid.bsize().y, 1);
    EXPECT_EQ(grid.bsize().z, 1);
    
    // ceil(1000 / 256) = 4
    EXPECT_EQ(grid.gsize().x, 4);
    EXPECT_EQ(grid.gsize().y, 1);
    EXPECT_EQ(grid.gsize().z, 1);
}

TEST(KernelGridTest, Calculate2D) {
    dim3 size(100, 100);
    dim3 block(10, 10);
    
    KernelGrid grid(size, block);
    
    EXPECT_EQ(grid.gsize().x, 10);
    EXPECT_EQ(grid.gsize().y, 10);
    EXPECT_EQ(grid.gsize().z, 1);
}

TEST(KernelGridTest, HandlesZeroInput) {
    // If user passes 0 size, it should clamp to at least 1 grid to avoid launch failures
    // or behave rationally. The current implementation uses max(1u, size.x).
    
    KernelGrid grid(0, 256);
    EXPECT_EQ(grid.gsize().x, 1); // (0 + 256 - 1) / 256 = 0 integer division? 
                                  // Wait, implementation: (max(1, 0) + 255) / 256 = 256/256 = 1.
    EXPECT_EQ(grid.bsize().x, 256);
}

TEST(KernelGridTest, ThrowsOnLimitExceeded) {
    // Attempt to create a grid that exceeds Y-dimension hardware limit (65535)
    dim3 huge_size(1, 70000 * 1); // 70000 items
    dim3 small_block(1, 1);       // 1 thread per block -> 70000 blocks in Y
    
    EXPECT_THROW({
        KernelGrid grid(huge_size, small_block);
    }, std::runtime_error);
}

TEST(KernelGridTest, RespectsXLimit) {
    // X limit is very large (2^31 - 1), so this should pass if within limits
    dim3 large_x(100000, 1, 1);
    dim3 block(1, 1, 1);
    
    EXPECT_NO_THROW({
        KernelGrid grid(large_x, block);
        EXPECT_EQ(grid.gsize().x, 100000);
    });
}

TEST(KernelGridTest, HandlesZeroBlockSize) {
    // Edge case: Block size is 0. Should be clamped to 1 to prevent division by zero.
    // Grid calc: (100 + 1 - 1) / 1 = 100
    KernelGrid grid(100, 0); 
    
    EXPECT_EQ(grid.bsize().x, 1); 
    EXPECT_EQ(grid.gsize().x, 100);
}

TEST(KernelGridTest, ExactBoundaryConditions) {
    // Y/Z Max is 65535.
    
    // Case 1: Exactly at limit (Should Pass)
    dim3 exact_limit(1, 65535, 1);
    dim3 block(1, 1, 1);
    EXPECT_NO_THROW({
        KernelGrid grid(exact_limit, block);
        EXPECT_EQ(grid.gsize().y, 65535);
    });

    // Case 2: One over limit (Should Fail)
    dim3 over_limit(1, 65536, 1);
    EXPECT_THROW({
        KernelGrid grid(over_limit, block);
    }, std::runtime_error);
}

TEST(KernelGridTest, 3DCalculation) {
    // Verify Z dimension specifically since most tests checked X/Y
    dim3 size(10, 10, 100);
    dim3 block(1, 1, 10);
    
    KernelGrid grid(size, block);
    EXPECT_EQ(grid.gsize().z, 10); // 100 / 10 = 10
}

} // namespace cropandweed
