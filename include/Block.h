#pragma once
#include <vector>
#include <cuda_runtime_api.h>
#include <memory>
#include <cstring>
#include "helpers.h"

namespace cropandweed {

enum class MemoryType {
    Device,     // Standard VRAM (fastest for heavy GPU math)
    Pinned,     // Host Pinned (staging for transfers on RTX)
    ZeroCopy    // Mapped Memory (CPU/GPU share pointer - Best for Jetson I/O)
};

// --- Architectural Memory Intent ---
// BoundaryMemType:
// Data bridging the compute boundary. Primarily generated and used by the GPU,
// but requires lightweight or occasional reads by the CPU (e.g., detection counts, bounding boxes).
//
// HostStagingMemType:
// Data whose gravity sits with the CPU for heavy I/O operations (e.g., writing JPEGs to disk).
// Provides a high-bandwidth lane for the GPU to receive or deliver bulk data.

#ifdef PLATFORM_JETSON

// --- Jetson Implementation (Unified Memory Architecture) ---
// Because the ARM CPU and GPU physically share the same RAM chips,
// MemoryType::ZeroCopy (cudaHostAllocMapped) is the optimal realization for both intents.
// It maps physical pages into both virtual address spaces, allowing direct pointer
// dereferencing by either processor without triggering redundant memory-to-memory copies.

constexpr MemoryType BoundaryMemType = MemoryType::ZeroCopy;
constexpr MemoryType HostStagingMemType = MemoryType::ZeroCopy;

#else

// --- PC Implementation (Discrete GPU over PCIe Bus) ---
// The CPU and GPU possess physically isolated memory banks separated by the PCIe bus.

// Pure VRAM is the fastest realization for GPU math. The data stays isolated
// on the GPU until the CPU explicitly requests a transfer across the boundary.
constexpr MemoryType BoundaryMemType = MemoryType::Device;

// Page-locked system RAM is the optimal realization for heavy CPU I/O.
// It prevents OS memory swapping, allowing the GPU's DMA controller to saturate
// the PCIe bus bandwidth during bulk transfers.
constexpr MemoryType HostStagingMemType = MemoryType::Pinned;

#endif

/**
 * @brief A class for storing consecutive elements of type T in the device memory.
 *
 * This class provides functionality similar to `std::vector` but is designed
 * to manage memory directly on a CUDA-enabled device. It allows for efficient
 * allocation, deallocation, and manipulation of data in GPU memory.
 *
 * @tparam T The type of elements to store in the block.
 */
template <class T, MemoryType MemType = MemoryType::Device>
class Block {
public:
    // --- 1. Constructors & Destructors ---

    //! The default constructor. Constructs an empty block
    Block() noexcept;

    //! The default destructor
    ~Block() noexcept;

    //! The move constructor. Constructs a block with the contents of other using move semantics.
    //! The data is moved from other into this container. other is empty afterwards
    Block(Block<T, MemType> &&other) noexcept;

    //! Move assignment operator. Replaces the contents with those of other using move semantics
    //! (i.e. the data in other is moved from other into this container). other is empty afterwards
    Block<T, MemType> &operator=(Block<T, MemType> &&other) noexcept;

    //! Copy semantics deleted to prevent implicit expensive GPU operations.
    //! Use assign() or copy_from() instead.
    Block(const Block<T, MemType> &) = delete;
    Block<T, MemType> &operator=(const Block<T, MemType> &) = delete;

    // --- 2. Factory Methods ---

    //! Factory method. Creates a new block of size elements length (memory is not initialized).
    static CudaError Create(std::unique_ptr<Block<T, MemType>>& out, size_t size, cudaStream_t stream = 0);

    //! Factory method. Creates a new block of size elements length and fills all bytes of memory with val.
    static CudaError Create(std::unique_ptr<Block<T, MemType>>& out, size_t size, T val, cudaStream_t stream = 0);

    //! Factory method. Creates a new block from the host vector.
    static CudaError Create(std::unique_ptr<Block<T, MemType>>& out, const std::vector<T>& data, cudaStream_t stream = 0);

    // --- 3. Data Transfer (Resizes + Copies) ---

    //! Replaces the data of the block with the host vector data.
    //! Discards previous content, resizing if necessary.
    CudaError assign(const std::vector<T> &other, cudaStream_t stream = 0);

    //! Replaces the data of the block with another block's data.
    //! Discards previous content, resizing if necessary.
    CudaError assign(const Block<T, MemType> &other, cudaStream_t stream = 0);

    //! Copies the contents of the block into the provided host vector.
    //! Resizes the output vector to match the block size.
    CudaError to_vector(std::vector<T> &out, cudaStream_t stream = 0) const;

    // --- 4. Modifiers ---

    //! Increase the capacity of the block (the total number of elements that the block can hold
    //! without requiring reallocation) to a value that's greater or equal to new_cap.
    //! If new_cap is greater than the current capacity(), new storage is allocated,
    //! otherwise the function does nothing. reserve() does not change the size of the block.
    CudaError reserve(size_t new_cap, cudaStream_t stream = 0);

    //! Resizes the block to contain count elements, does nothing if new_size == size().
    //! If the current size is greater than new_size, the block is reduced to its first new_size elements.
    //! If the current size is less than new_size, then additional not initialized elements are appended
    CudaError resize(size_t new_size, cudaStream_t stream = 0);

    //! Resizes the block to contain count elements, does nothing if new_size == size().
    //! If the current size is greater than new_size, the block is reduced to its first new_size elements.
    //! If the current size is less than new_size, then additional elements are appended.
    //! Every byte of memory for the appended elements is filled with val
    CudaError resize(size_t new_size, T val, cudaStream_t stream = 0);

    //! Fills all initialized elements (from 0 to size()) with the given value.
    CudaError fill(T val, cudaStream_t stream = 0);

    //! Fills elements from `offset` up to `size_`
    CudaError fill_back(size_t offset, T val, cudaStream_t stream);

    //! Erases all elements from the container. After this call, size() returns zero.
    //! Leaves the capacity() of the block unchanged
    __host__ __device__ void clear() noexcept;

    //! Exchanges the contents and capacity of the container with those of other
    void swap(Block<T, MemType> &other);


    // --- 5. Accessors ---

    //! Helper to check type at runtime if needed
    static constexpr MemoryType type() { return MemType; }

    //! Returns the element at specified location pos
    __host__ __device__ const T &operator [](size_t pos) const;

    //! Returns a pointer to the underlying array serving as element storage
    __host__ __device__ T *data() noexcept;

    //! Returns a const pointer to the underlying array serving as element storage
    __host__ __device__ const T *data() const noexcept;

    //! Returns a pointer to the first element of a block
    __host__ __device__ T *begin() noexcept;

    //! Returns a const pointer to the first element of a block
    __host__ __device__ const T *begin() const noexcept;

    //! Returns a const pointer to the first element of a block
    __host__ __device__ const T *cbegin() const noexcept;

    //! Returns a pointer to the element of a block following the last element
    __host__ __device__ T *end() noexcept;

    //! Returns a const pointer to the element of a block following the last element
    __host__ __device__ const T *end() const noexcept;

    //! Returns a const pointer to the element of a block following the last element
    __host__ __device__ const T *cend() const noexcept;

    //! Checks if the block has no elements. Returns true if the block is empty, false otherwise
    __host__ __device__ bool empty() const noexcept;

    //! Returns the number of elements in the block
    __host__ __device__ size_t size() const;

    //! Returns the size of memory used by the block elements in bytes
    __host__ __device__ size_t byte_size() const;

    //! Returns the number of elements that the block has currently allocated space for
    __host__ __device__ size_t capacity() const;

private:
    //! A pointer to internal data storage
    T *ptr_;
    //! Size and capacity values
    size_t size_, capacity_;

    //! Frees the underlying memory. After this call, size() and capacity() return zero.
    void free() noexcept;

    //! Allocates the memory for new_cap elements. Updates size and capacity to new_cap.
    //! Returns CudaError if allocation fails.
    CudaError malloc_impl(size_t new_cap);

    //! If possible (allocated memory is enough) copies the data from other to this block.
    //! No memory allocations are made. Assumes capacity is sufficient.
    CudaError copy_from(const Block<T, MemType> &other, cudaStream_t stream);

    //! If possible (allocated memory is enough) copies the data from other host vector to this block.
    //! No memory allocations are made. Assumes capacity is sufficient.
    CudaError copy_from(const std::vector<T> &other, cudaStream_t stream);

    //! If possible (the size of a host vector is enough) copies the data from this block to other host vector.
    //! No memory allocations are made. Assumes output vector is already resized.
    CudaError copy_to(std::vector<T> &other, cudaStream_t stream) const;
};

// Header-only adapter for type safety without template instantiation
template <typename T, MemoryType MemType = MemoryType::Device>
class TypedBlock {
private:
    Block<uint8_t, MemType> raw_; // Underlying storage

public:
    TypedBlock() = default;
    ~TypedBlock() = default;

    // Move Semantics
    TypedBlock(TypedBlock&& other) noexcept : raw_(std::move(other.raw_)) {}
    TypedBlock& operator=(TypedBlock&& other) noexcept {
        raw_ = std::move(other.raw_);
        return *this;
    }

    // No Copy
    TypedBlock(const TypedBlock&) = delete;
    TypedBlock& operator=(const TypedBlock&) = delete;

    // Type-Safe Resize (Count based, not Byte based)
    CudaError resize(size_t count, cudaStream_t stream = 0) {
        return raw_.resize(count * sizeof(T), stream);
    }

    // Type-Safe Reserve
    CudaError reserve(size_t count, cudaStream_t stream = 0) {
        return raw_.reserve(count * sizeof(T), stream);
    }

    CudaError fill_zero(cudaStream_t stream = 0) {
        return raw_.fill(0, stream);
    }

    // Assign from host vector
    CudaError assign(const std::vector<T>& other, cudaStream_t stream = 0) {
        size_t totalBytes = other.size() * sizeof(T);
        CUDA_TRY(raw_.resize(totalBytes, stream));
        if (totalBytes > 0) {
            if constexpr (MemType == MemoryType::Device) {
                CUDA_TRY(cudaMemcpyAsync(raw_.data(), other.data(),
                                         totalBytes, cudaMemcpyHostToDevice, stream));
                CUDA_TRY(cudaStreamSynchronize(stream));
            } else {
                CUDA_TRY(cudaStreamSynchronize(stream));
                std::memcpy(raw_.data(), other.data(), totalBytes);
            }
        }
        return CudaError();
    }

    // Copy to host vector
    CudaError to_vector(std::vector<T>& out, cudaStream_t stream = 0) const {
        size_t count = size();
        out.resize(count);
        if (count > 0) {
            size_t totalBytes = count * sizeof(T);
            if constexpr (MemType == MemoryType::Device) {
                CUDA_TRY(cudaMemcpyAsync(out.data(), raw_.data(),
                                         totalBytes, cudaMemcpyDeviceToHost, stream));
                CUDA_TRY(cudaStreamSynchronize(stream));
            } else {
                CUDA_TRY(cudaStreamSynchronize(stream));
                std::memcpy(out.data(), raw_.data(), totalBytes);
            }
        }
        return CudaError();
    }

    // Accessors
    __host__ __device__ T* data() noexcept { return reinterpret_cast<T*>(raw_.data()); }
    __host__ __device__ const T* data() const noexcept { return reinterpret_cast<const T*>(raw_.data()); }

    __host__ __device__ size_t size() const { return raw_.size() / sizeof(T); }
    __host__ __device__ size_t byte_size() const { return raw_.byte_size(); }
    __host__ __device__ bool empty() const { return raw_.empty(); }
    __host__ __device__ size_t capacity() const { return raw_.capacity() / sizeof(T); }
    // For specialized cases needing the raw byte block
    Block<uint8_t, MemType>& raw() { return raw_; }
};

template <typename T>
using BoundaryBlock = Block<T, BoundaryMemType>;

template <typename T>
using BoundaryTypedBlock = TypedBlock<T, BoundaryMemType>;

template <typename T>
using HostStagingBlock = Block<T, HostStagingMemType>;

} // namespace cropandweed
