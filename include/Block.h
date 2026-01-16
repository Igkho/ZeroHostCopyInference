#pragma once
#include <vector>
#include <cuda_runtime_api.h>
#include <memory>
#include "helpers.h"

namespace cropandweed {

enum class MemoryType {
    Device,     // Standard VRAM (fastest for heavy GPU math)
    Pinned,     // Host Pinned (staging for transfers on RTX)
    ZeroCopy    // Mapped Memory (CPU/GPU share pointer - Best for Jetson I/O)
};

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
    static CudaError Create(std::unique_ptr<Block<T, MemType>>& out, size_t size);

    //! Factory method. Creates a new block of size elements length and fills all bytes of memory with val.
    static CudaError Create(std::unique_ptr<Block<T, MemType>>& out, size_t size, int val);

    //! Factory method. Creates a new block from the host vector.
    static CudaError Create(std::unique_ptr<Block<T, MemType>>& out, const std::vector<T>& data);

    // --- 3. Data Transfer (Resizes + Copies) ---

    //! Replaces the data of the block with the host vector data.
    //! Discards previous content, resizing if necessary.
    CudaError assign(const std::vector<T> &other);

    //! Replaces the data of the block with another block's data.
    //! Discards previous content, resizing if necessary.
    CudaError assign(const Block<T, MemType> &other);

    //! Copies the contents of the block into the provided host vector.
    //! Resizes the output vector to match the block size.
    CudaError to_vector(std::vector<T> &out) const;

    // --- 4. Modifiers ---

    //! Increase the capacity of the block (the total number of elements that the block can hold
    //! without requiring reallocation) to a value that's greater or equal to new_cap.
    //! If new_cap is greater than the current capacity(), new storage is allocated,
    //! otherwise the function does nothing. reserve() does not change the size of the block.
    CudaError reserve(size_t new_cap);

    //! Resizes the block to contain count elements, does nothing if new_size == size().
    //! If the current size is greater than new_size, the block is reduced to its first new_size elements.
    //! If the current size is less than new_size, then additional not initialized elements are appended
    CudaError resize(size_t new_size);

    //! Resizes the block to contain count elements, does nothing if new_size == size().
    //! If the current size is greater than new_size, the block is reduced to its first new_size elements.
    //! If the current size is less than new_size, then additional elements are appended.
    //! Every byte of memory for the appended elements is filled with val
    CudaError resize(size_t new_size, int val);

    //! Erases all elements from the container. After this call, size() returns zero.
    //! Leaves the capacity() of the block unchanged
    void clear() noexcept;

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
    CudaError copy_from(const Block<T, MemType> &other);

    //! If possible (allocated memory is enough) copies the data from other host vector to this block.
    //! No memory allocations are made. Assumes capacity is sufficient.
    CudaError copy_from(const std::vector<T> &other);

    //! If possible (the size of a host vector is enough) copies the data from this block to other host vector.
    //! No memory allocations are made. Assumes output vector is already resized.
    CudaError copy_to(std::vector<T> &other) const;
};

} // namespace cropandweed
