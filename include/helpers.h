#pragma once
#include <iostream>
#include <string>
#include <sstream>
#include <algorithm>
#include <filesystem>
#include <vector>
#include <cuda_runtime_api.h>
#include <nvjpeg.h>
#include <memory>
#include <type_traits>

namespace cropandweed {

static inline std::string ErrorSource(const char *file, int line) {
    std::string fileName = std::filesystem::path(file).filename().string();
    return fileName + ":" + std::to_string(line);
}

// Macro maps to the namespace function below
#define ERROR_SOURCE cropandweed::ErrorSource(__FILE__, __LINE__)

class CudaError {
public:
    static inline std::string GetErrorString(cudaError_t err) {
        return cudaGetErrorString(err);
    }

    static inline std::string GetErrorString(nvjpegStatus_t err) {
        // nvJPEG often doesn't have a standardized string function in older toolkits
        switch(err) {
        case NVJPEG_STATUS_SUCCESS: return "Success";
        case NVJPEG_STATUS_NOT_INITIALIZED: return "Not Initialized";
        case NVJPEG_STATUS_INVALID_PARAMETER: return "Invalid Parameter";
        case NVJPEG_STATUS_BAD_JPEG: return "Bad JPEG";
        case NVJPEG_STATUS_JPEG_NOT_SUPPORTED: return "JPEG Not Supported";
        case NVJPEG_STATUS_ALLOCATOR_FAILURE: return "Allocator Failure";
        case NVJPEG_STATUS_EXECUTION_FAILED: return "Execution Failed";
        case NVJPEG_STATUS_ARCH_MISMATCH: return "Arch Mismatch";
        case NVJPEG_STATUS_INTERNAL_ERROR: return "Internal Error";
        default: return "Unknown NVJPEG Error (" + std::to_string(err) + ")";
        }
    }

    static inline std::string GetErrorString(const std::string &err) {
        return err;
    }

    static bool IsFailure(cudaError_t err) {
        return err != cudaSuccess;
    }

    static bool IsFailure(nvjpegStatus_t err) {
        return err != NVJPEG_STATUS_SUCCESS;
    }

    static bool IsFailure(const std::string &err) {
        return !err.empty();
    }

    static bool IsFailure(const CudaError& err) {
        return !err.call_stack_.empty();
    }

    CudaError() = default;

    template <class T>
    CudaError(const std::string &source, T cudaErrCode) {
        if (IsFailure(cudaErrCode)) {
            call_stack_.emplace_back(source, GetErrorString(cudaErrCode));
        }
    }

    CudaError(const std::string &source, const CudaError &other) {
        call_stack_ = other.call_stack_;
        call_stack_.emplace_back(source, "unwrap");
    }

    std::string Text() const {
        if (call_stack_.empty()) {
            return "No GPU errors";
        }
        std::stringstream ss;
        ss << "\n=== GPU ERROR TRACE ===\n";
        for (auto it = call_stack_.rbegin(); it != call_stack_.rend(); ++it) {
            ss << " >> " << it->first << ": " << it->second << "\n";
        }
        return ss.str();
    }

private:
    std::vector<std::pair<std::string, std::string> > call_stack_;
};

#define CUDA_CALL(f) { \
    auto _res = (f); \
    if (cropandweed::CudaError::IsFailure(_res)) { \
        cropandweed::CudaError _err(ERROR_SOURCE, _res); \
        std::cerr << _err.Text() << std::endl; \
        throw std::runtime_error(_err.Text()); \
} \
}

#define CUDA_CALL_NO_THROW(f) { \
auto _res = (f); \
    if (cropandweed::CudaError::IsFailure(_res)) { \
        cropandweed::CudaError _err(ERROR_SOURCE, _res); \
        std::cerr << _err.Text() << std::endl; \
} \
}

#define CUDA_TRY(f) { \
    auto _res = (f); \
    if (cropandweed::CudaError::IsFailure(_res)) { \
        return cropandweed::CudaError(ERROR_SOURCE, _res); \
} \
}

// Generic error generator for logical failures (non-CUDA APIs)
#define CUDA_GENERAL_ERROR(msg) { \
return cropandweed::CudaError(ERROR_SOURCE, std::string(msg)); \
}

// --- Helper Trait to Detect shared_ptr ---
// Primary template: defaults to false
template<typename T>
struct is_shared_ptr : std::false_type {};

// Template specialization for std::shared_ptr<T>: sets to true
template<typename T>
struct is_shared_ptr<std::shared_ptr<T>> : std::true_type {};

// Optional: Template specialization for const shared_ptr (as it's a common case)
template<typename T>
struct is_shared_ptr<const std::shared_ptr<T>> : std::true_type {};

// Helper constant for convenience (C++17 onwards)
template<typename T>
constexpr bool is_shared_ptr_v = is_shared_ptr<T>::value;

// RAII Wrapper for CUDA Streams
struct CudaStream {
private:
    struct Token {};
    cudaStream_t stream = nullptr;

public:
    CudaStream(Token) {}

    // Universal Factory: Supports unique_ptr AND shared_ptr
    template <typename SmartPtr>
    static CudaError Create(SmartPtr& out, unsigned int flags = cudaStreamDefault) {
        cudaStream_t rawStream;
        CUDA_TRY(cudaStreamCreateWithFlags(&rawStream, flags));
        if constexpr (is_shared_ptr_v<SmartPtr>) {
            out = std::make_shared<CudaStream>(Token{});
        }
        else {
            out = std::make_unique<CudaStream>(Token{});
        }
        out->stream = rawStream;
        return CudaError();
    }

    // Destructor automatically destroys it
    ~CudaStream() {
        if (stream) {
            CUDA_CALL_NO_THROW(cudaStreamDestroy(stream));
        }
    }

    // No Copying
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    // Allow Moving
    CudaStream(CudaStream&& other) noexcept : stream(other.stream) {
        other.stream = nullptr;
    }

    CudaStream& operator=(CudaStream&& other) noexcept {
        if (this != &other) {
            if (stream) {
                CUDA_CALL_NO_THROW(cudaStreamDestroy(stream));
            }
            stream = other.stream;
            other.stream = nullptr;
        }
        return *this;
    }

    // Implicit conversion to cudaStream_t
    operator cudaStream_t() const {
        return stream;
    }
};

// RAII Wrapper for CUDA Events
struct CudaEvent {
private:
    struct Token {};
    cudaEvent_t event = nullptr;

public:

    CudaEvent(Token) {}

    template <typename SmartPtr>
    static CudaError Create(SmartPtr& out, unsigned int flags = cudaEventDisableTiming) {
        cudaEvent_t rawEvent;
        CUDA_TRY(cudaEventCreateWithFlags(&rawEvent, flags));
        if constexpr (is_shared_ptr_v<SmartPtr>) {
            out = std::make_shared<CudaEvent>(Token{});
        }
        else {
            out = std::make_unique<CudaEvent>(Token{});
        }
        out->event = rawEvent;
        return CudaError();
    }

    ~CudaEvent() {
        if (event) {
            CUDA_CALL_NO_THROW(cudaEventDestroy(event));
        }
    }

    // No Copying
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;

    // Allow Moving
    CudaEvent(CudaEvent&& other) noexcept : event(other.event) {
        other.event = nullptr;
    }

    CudaEvent& operator=(CudaEvent&& other) noexcept {
        if (this != &other) {
            if (event) {
                CUDA_CALL_NO_THROW(cudaEventDestroy(event));
            }
            event = other.event;
            other.event = nullptr;
        }
        return *this;
    }

    operator cudaEvent_t() const { return event; }
};

class KernelGrid {
public:
    KernelGrid() = delete;
    KernelGrid(const KernelGrid &) = default;
    KernelGrid &operator =(const KernelGrid &) = default;
    ~KernelGrid() = default;

    KernelGrid(unsigned int size, unsigned int block = 256) {
        calculate(dim3{size, 1, 1}, dim3{block, 1, 1});
        check_limits();
    }

    KernelGrid(dim3 size, dim3 block) {
        calculate(size, block);
        check_limits();
    }

    dim3 gsize() const {
        return gsize_;
    }

    dim3 bsize() const {
        return bsize_;
    }

private:

    void calculate(const dim3 &size,
                   const dim3 &block) {
        bsize_ = dim3{std::max(1u, block.x), std::max(1u, block.y), std::max(1u, block.z)};
        gsize_ = dim3{(std::max(1u, size.x) + bsize_.x - 1) / bsize_.x,
                      (std::max(1u, size.y) + bsize_.y - 1) / bsize_.y,
                      (std::max(1u, size.z) + bsize_.z - 1) / bsize_.z};
    }

    // Safety check function
    void check_limits() const {
        // Standard CUDA limits for Compute Capability 3.0+
        // Grid X Max: 2,147,483,647
        // Grid Y/Z Max: 65,535
        const unsigned int MAX_Y_Z = 65535;
        const unsigned int MAX_X = 2147483647;

        if (gsize_.x > MAX_X || gsize_.y > MAX_Y_Z || gsize_.z > MAX_Y_Z) {
            std::string msg = "KernelGrid Error in " + ERROR_SOURCE + ": " +
                              "Grid dimension calculated " +
                              "(" + std::to_string(gsize_.x) + ", " +
                              std::to_string(gsize_.y) + ", " +
                              std::to_string(gsize_.z) + ")" +
                              " exceeds hardware limits (X:" + std::to_string(MAX_X) +
                              ", Y/Z:" + std::to_string(MAX_Y_Z) + ")";

            std::cerr << msg << std::endl;
            throw std::runtime_error(msg);
        }
    }

    dim3 gsize_, bsize_;
};

} // namespace cropandweed
