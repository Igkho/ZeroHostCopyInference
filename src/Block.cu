#include "Block.h"
#include <cstring>
#include <iostream>

namespace cropandweed {

namespace {

// Kernel for element-wise filling of non-byte types
template <class T>
__global__ void FillKernel(T* __restrict__ ptr, int val, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        ptr[idx] = static_cast<T>(val);
    }
}

}

// --- Constructors / Destructors ---

template <class T, MemoryType MemType>
Block<T, MemType>::Block() noexcept : ptr_(nullptr), size_(0), capacity_(0) {}

template <class T, MemoryType MemType>
Block<T, MemType>::Block(Block<T, MemType> &&other) noexcept : Block() {
    swap(other);
}

template <class T, MemoryType MemType>
Block<T, MemType> &Block<T, MemType>::operator=(Block<T, MemType> &&other) noexcept {
    if (this != &other) {
        free();
        ptr_ = other.ptr_;
        capacity_ = other.capacity_;
        size_ = other.size_;
        other.ptr_ = nullptr;
        other.capacity_ = other.size_ = 0;
    }
    return *this;
}

template <class T, MemoryType MemType>
Block<T, MemType>::~Block() noexcept {
    free();
}

// --- Factories ---

template <class T, MemoryType MemType>
CudaError Block<T, MemType>::Create(std::unique_ptr<Block<T, MemType>>& out, size_t size, cudaStream_t stream) {
    out = std::make_unique<Block<T, MemType>>();
    CUDA_TRY(out->resize(size, stream));
    return CudaError();
}

template <class T, MemoryType MemType>
CudaError Block<T, MemType>::Create(std::unique_ptr<Block<T, MemType>>& out, size_t size, int val, cudaStream_t stream) {
    out = std::make_unique<Block<T, MemType>>();
    CUDA_TRY(out->resize(size, val, stream));
    return CudaError();
}

template <class T, MemoryType MemType>
CudaError Block<T, MemType>::Create(std::unique_ptr<Block<T, MemType>>& out, const std::vector<T>& data, cudaStream_t stream) {
    out = std::make_unique<Block<T, MemType>>();
    CUDA_TRY(out->assign(data, stream));
    return CudaError();
}

// --- Public Data Transfer ---

template <class T, MemoryType MemType>
CudaError Block<T, MemType>::assign(const std::vector<T> &other, cudaStream_t stream) {
    if (capacity_ < other.size()) {
        CUDA_TRY(malloc_impl(other.size()));
    }
    size_ = other.size();
    CUDA_TRY(copy_from(other, stream)); // Uses private primitive
    return CudaError();
}

template <class T, MemoryType MemType>
CudaError Block<T, MemType>::assign(const Block<T, MemType> &other, cudaStream_t stream) {
    if (capacity_ < other.size()) {
        CUDA_TRY(malloc_impl(other.size()));
    }
    size_ = other.size();
    CUDA_TRY(copy_from(other, stream));
    return CudaError();
}

template <class T, MemoryType MemType>
CudaError Block<T, MemType>::to_vector(std::vector<T> &out, cudaStream_t stream) const {
    out.resize(size_);
    CUDA_TRY(copy_to(out, stream));
    return CudaError();
}

// --- Modifiers ---

template <class T, MemoryType MemType>
CudaError Block<T, MemType>::resize(size_t new_size, cudaStream_t stream) {
    if (capacity_ >= new_size) {
        size_ = new_size;
        return CudaError();
    }
    // Standard Resize: Must preserve old data
    Block<T, MemType> other;
    CUDA_TRY(other.malloc_impl(new_size));

    // Copy what we have (preserving data)
    CUDA_TRY(other.copy_from(*this, stream));

    other.size_ = new_size;
    swap(other);
    return CudaError();
}

template <class T, MemoryType MemType>
CudaError Block<T, MemType>::resize(size_t new_size, int val, cudaStream_t stream) {
    size_t old_size = size_;
    CUDA_TRY(resize(new_size, stream));

    // Fill only the new part
    if (new_size > old_size) {
        size_t elements_to_fill = new_size - old_size;
        size_t bytes_to_fill = (new_size - old_size) * sizeof(T);
        if constexpr (MemType == MemoryType::Device) {
            if (sizeof(T) == 1 || val == 0) {
                CUDA_TRY(cudaMemsetAsync(ptr_ + old_size, val, bytes_to_fill, stream));
            } else {
                KernelGrid grid(elements_to_fill);
                FillKernel<<<grid.gsize(), grid.bsize(), 0, stream>>>(ptr_ + old_size,
                                                                      val,
                                                                      elements_to_fill);
                CUDA_TRY(cudaGetLastError());
            }
        } else {
            T* fill_ptr = ptr_ + old_size;
            if (sizeof(T) == 1 || val == 0) {
                std::memset(fill_ptr, val, elements_to_fill * sizeof(T));
            } else {
                std::fill(fill_ptr, fill_ptr + elements_to_fill, static_cast<T>(val));
            }
        }
    }
    return CudaError();
}

template <class T, MemoryType MemType>
CudaError Block<T, MemType>::reserve(size_t new_cap, cudaStream_t stream) {
    if (capacity_ >= new_cap) return CudaError();

    Block<T, MemType> other;
    CUDA_TRY(other.malloc_impl(new_cap));
    CUDA_TRY(other.copy_from(*this, stream)); // Preserve data async

    other.size_ = size_;
    swap(other);
    return CudaError();
}

// --- PRIVATE Primitives ---

template <class T, MemoryType MemType>
CudaError Block<T, MemType>::copy_from(const Block<T, MemType> &other, cudaStream_t stream) {
    if (other.size() > 0) {
        // We trust 'assign' or 'resize' set the size/capacity correctly before calling this.
        // We copy min(capacity, other.size) just to be safe from overflows.
        size_t copy_amount = (capacity_ < other.size()) ? capacity_ : other.size();

        if constexpr (MemType != MemoryType::Device) {
            std::memcpy(ptr_, other.data(), copy_amount * sizeof(T));
        } else {
            CUDA_TRY(cudaMemcpyAsync(ptr_, other.data(),
                                     copy_amount * sizeof(T), cudaMemcpyDeviceToDevice, stream));
        }
    }
    return CudaError();
}

template <class T, MemoryType MemType>
CudaError Block<T, MemType>::copy_from(const std::vector<T> &other, cudaStream_t stream) {
    if (other.size() > 0) {
        size_t copy_amount = (capacity_ < other.size()) ? capacity_ : other.size();

        if constexpr (MemType == MemoryType::Device) {
            CUDA_TRY(cudaMemcpyAsync(ptr_, other.data(),
                                     copy_amount * sizeof(T), cudaMemcpyHostToDevice, stream));
        } else {
            std::memcpy(ptr_, other.data(), copy_amount * sizeof(T));
        }
    }
    return CudaError();
}

template <class T, MemoryType MemType>
CudaError Block<T, MemType>::copy_to(std::vector<T> &other, cudaStream_t stream) const {
    if (size_ > 0) {
        size_t copy_amount = (other.size() < size_) ? other.size() : size_;

        if constexpr (MemType == MemoryType::Device) {
            // Note: If 'other.data()' is not pinned memory, this might effectively be synchronous,
            // but explicitly asking for Async gives the driver the chance to optimize.
            CUDA_TRY(cudaMemcpyAsync(other.data(), ptr_,
                                     copy_amount * sizeof(T), cudaMemcpyDeviceToHost, stream));
            // SAFETY: Since we are copying to a std::vector (pageable memory),
            // we often need to sync before the CPU reads it.
            // However, typical async patterns usually sync via Events later.
            // For a helper like `to_vector` which returns a CPU object, the user expects data to be ready.
            // So we Sync here.
            CUDA_TRY(cudaStreamSynchronize(stream));
        } else {
            std::memcpy(other.data(), ptr_, copy_amount * sizeof(T));
        }
    }
    return CudaError();
}

// --- Internals ---

template <class T, MemoryType MemType>
void Block<T, MemType>::swap(Block<T, MemType> &other) {
    std::swap(ptr_, other.ptr_);
    std::swap(capacity_, other.capacity_);
    std::swap(size_, other.size_);
}

template <class T, MemoryType MemType>
void Block<T, MemType>::free() noexcept {
    if (ptr_ != nullptr) {
        if constexpr (MemType == MemoryType::Device) {
            CUDA_CALL_NO_THROW(cudaFree((void *)ptr_));
        } else {
            CUDA_CALL_NO_THROW(cudaFreeHost((void *)ptr_));
        }
        ptr_ = nullptr;
    }
    capacity_ = size_ = 0;
}

template <class T, MemoryType MemType>
CudaError Block<T, MemType>::malloc_impl(size_t new_cap) {
    if (capacity_ >= new_cap) return CudaError();
    free();
    if constexpr (MemType == MemoryType::Device) {
        CUDA_TRY(cudaMalloc((void **)&ptr_, new_cap * sizeof(T)));
    } else if constexpr (MemType == MemoryType::Pinned) {
        CUDA_TRY(cudaMallocHost((void **)&ptr_, new_cap * sizeof(T)));
    } else if constexpr (MemType == MemoryType::ZeroCopy) {
        CUDA_TRY(cudaHostAlloc((void **)&ptr_, new_cap * sizeof(T),
                               cudaHostAllocMapped | cudaHostAllocPortable));
    }
    capacity_ = size_ = new_cap;
    return CudaError();
}

// --- Accessors ---
template <class T, MemoryType MemType>
T *Block<T, MemType>::data() noexcept { return ptr_; }
template <class T, MemoryType MemType>
const T *Block<T, MemType>::data() const noexcept { return ptr_; }
template <class T, MemoryType MemType>
T *Block<T, MemType>::begin() noexcept { return ptr_; }
template <class T, MemoryType MemType>
const T *Block<T, MemType>::begin() const noexcept { return ptr_; }
template <class T, MemoryType MemType>
const T *Block<T, MemType>::cbegin() const noexcept { return ptr_; }
template <class T, MemoryType MemType>
T *Block<T, MemType>::end() noexcept { return ptr_ + size_; }
template <class T, MemoryType MemType>
const T *Block<T, MemType>::end() const noexcept { return ptr_ + size_; }
template <class T, MemoryType MemType>
const T *Block<T, MemType>::cend() const noexcept { return ptr_ + size_; }
template <class T, MemoryType MemType>
bool Block<T, MemType>::empty() const noexcept { return !size_; }
template <class T, MemoryType MemType>
size_t Block<T, MemType>::size() const { return size_; }
template <class T, MemoryType MemType>
size_t Block<T, MemType>::byte_size() const { return size_ * sizeof(T); }
template <class T, MemoryType MemType>
size_t Block<T, MemType>::capacity() const { return capacity_; }
template <class T, MemoryType MemType>
void Block<T, MemType>::clear() noexcept { size_ = 0; }
template <class T, MemoryType MemType>
const T &Block<T, MemType>::operator[](size_t pos) const { return ptr_[pos]; }

// Explicit Instantiation
template class Block<double>;
template class Block<float>;
template class Block<int>;
template class Block<unsigned long long>;
template class Block<uint8_t>;
template class Block<uint8_t, MemoryType::Pinned>;
template class Block<uint8_t, MemoryType::ZeroCopy>;

} // namespace cropandweed
