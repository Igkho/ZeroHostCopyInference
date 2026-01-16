#include "Block.h"
#include <cstring>
#include <iostream>

namespace cropandweed {

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
CudaError Block<T, MemType>::Create(std::unique_ptr<Block<T, MemType>>& out, size_t size) {
    out = std::make_unique<Block<T, MemType>>();
    CUDA_TRY(out->resize(size));
    return CudaError();
}

template <class T, MemoryType MemType>
CudaError Block<T, MemType>::Create(std::unique_ptr<Block<T, MemType>>& out, size_t size, int val) {
    out = std::make_unique<Block<T, MemType>>();
    CUDA_TRY(out->resize(size, val));
    return CudaError();
}

template <class T, MemoryType MemType>
CudaError Block<T, MemType>::Create(std::unique_ptr<Block<T, MemType>>& out, const std::vector<T>& data) {
    out = std::make_unique<Block<T, MemType>>();
    CUDA_TRY(out->assign(data));
    return CudaError();
}

// --- Public Data Transfer ---

template <class T, MemoryType MemType>
CudaError Block<T, MemType>::assign(const std::vector<T> &other) {
    // OPTIMIZATION: Do not use resize() here.
    // If we need more space, just malloc new space. We don't care about preserving old data.
    if (capacity_ < other.size()) {
        CUDA_TRY(malloc_impl(other.size()));
    }
    size_ = other.size();
    CUDA_TRY(copy_from(other)); // Uses private primitive
    return CudaError();
}

template <class T, MemoryType MemType>
CudaError Block<T, MemType>::assign(const Block<T, MemType> &other) {
    // OPTIMIZATION: Same as above, discard old data.
    if (capacity_ < other.size()) {
        CUDA_TRY(malloc_impl(other.size()));
    }
    size_ = other.size();
    CUDA_TRY(copy_from(other));
    return CudaError();
}

template <class T, MemoryType MemType>
CudaError Block<T, MemType>::to_vector(std::vector<T> &out) const {
    out.resize(size_);
    CUDA_TRY(copy_to(out));
    return CudaError();
}

// --- Modifiers ---

template <class T, MemoryType MemType>
CudaError Block<T, MemType>::resize(size_t new_size) {
    if (capacity_ >= new_size) {
        size_ = new_size;
        return CudaError();
    }
    // Standard Resize: Must preserve old data
    Block<T, MemType> other;
    CUDA_TRY(other.malloc_impl(new_size));

    // Copy what we have (preserving data)
    CUDA_TRY(other.copy_from(*this));

    other.size_ = new_size;
    swap(other);
    return CudaError();
}

template <class T, MemoryType MemType>
CudaError Block<T, MemType>::resize(size_t new_size, int val) {
    size_t old_size = size_;
    CUDA_TRY(resize(new_size));

    // Fill only the new part
    if (new_size > old_size) {
        size_t bytes_to_fill = (new_size - old_size) * sizeof(T);
        if constexpr (MemType == MemoryType::Device) {
            CUDA_TRY(cudaMemset(ptr_ + old_size, val, bytes_to_fill));
        } else {
            std::memset(ptr_ + old_size, val, bytes_to_fill);
        }
    }
    return CudaError();
}

template <class T, MemoryType MemType>
CudaError Block<T, MemType>::reserve(size_t new_cap) {
    if (capacity_ >= new_cap) return CudaError();

    Block<T, MemType> other;
    CUDA_TRY(other.malloc_impl(new_cap));
    CUDA_TRY(other.copy_from(*this)); // Preserve data

    other.size_ = size_;
    swap(other);
    return CudaError();
}

// --- PRIVATE Primitives ---

template <class T, MemoryType MemType>
CudaError Block<T, MemType>::copy_from(const Block<T, MemType> &other) {
    if (other.size() > 0) {
        // We trust 'assign' or 'resize' set the size/capacity correctly before calling this.
        // We copy min(capacity, other.size) just to be safe from overflows.
        size_t copy_amount = (capacity_ < other.size()) ? capacity_ : other.size();

        if constexpr (MemType != MemoryType::Device) {
            std::memcpy(ptr_, other.data(), copy_amount * sizeof(T));
        } else {
            CUDA_TRY(cudaMemcpy(ptr_, other.data(), copy_amount * sizeof(T), cudaMemcpyDeviceToDevice));
        }
    }
    return CudaError();
}

template <class T, MemoryType MemType>
CudaError Block<T, MemType>::copy_from(const std::vector<T> &other) {
    if (other.size() > 0) {
        size_t copy_amount = (capacity_ < other.size()) ? capacity_ : other.size();

        if constexpr (MemType == MemoryType::Device) {
            CUDA_TRY(cudaMemcpy(ptr_, other.data(), copy_amount * sizeof(T), cudaMemcpyHostToDevice));
        } else {
            std::memcpy(ptr_, other.data(), copy_amount * sizeof(T));
        }
    }
    return CudaError();
}

template <class T, MemoryType MemType>
CudaError Block<T, MemType>::copy_to(std::vector<T> &other) const {
    if (size_ > 0) {
        size_t copy_amount = (other.size() < size_) ? other.size() : size_;

        if constexpr (MemType == MemoryType::Device) {
            CUDA_TRY(cudaMemcpy(other.data(), ptr_, copy_amount * sizeof(T), cudaMemcpyDeviceToHost));
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
