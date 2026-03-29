#pragma once
#include "Interfaces.h"
#include "Block.h"
#include "helpers.h"
#include <vector>
#include <string>
#include <future>
#include <memory>
#include <cuda_runtime_api.h>
#include "ObjectTracker.h"

class NvJPEGEncoder;
struct NvBufSurface;

namespace cropandweed {

struct EncodeResource {
    static constexpr int HW_ENCODERS_PER_BUFFER = 2;
    
    // Hardware Buffers
    NvBufSurface* pitch_linear_surf = nullptr;
    cudaGraphicsResource_t egl_cuda_resource = nullptr;

    // Pre-allocated CPU buffer to prevent heap fragmentation during encoding
    std::vector<uint8_t> compressed_buffer; 

    uint32_t width = 0;
    uint32_t height = 0;

    EncodeResource() = default;
    ~EncodeResource() noexcept;
    
    void DestroyHardwareSurfaces(); 

    EncodeResource(const EncodeResource&) = delete;
    EncodeResource& operator=(const EncodeResource&) = delete;
    EncodeResource(EncodeResource &&other) noexcept {
        pitch_linear_surf = other.pitch_linear_surf;
        egl_cuda_resource = other.egl_cuda_resource;
        compressed_buffer = std::move(other.compressed_buffer);
        width = other.width;
        height = other.height;

        // Nullify the source to prevent double-free
        other.pitch_linear_surf = nullptr;
        other.egl_cuda_resource = nullptr;
    }
    EncodeResource& operator=(EncodeResource&&) noexcept = delete;
};

struct MMAPIEncodeStatus {
    CudaError err;
    bool success = false;
    std::string filename;
};

class MMAPIJpegSink : public ISink {
private:
    struct Token {};

public:
    MMAPIJpegSink(Token, std::string output_folder, ModelProperties modelProps, size_t batch_size);

    ~MMAPIJpegSink() noexcept override;

    static CudaError Create(std::unique_ptr<ISink>& out, std::string output_folder,
                            ModelProperties modelProps, size_t batch_size);

    CudaError Save(BatchData& batch, BatchDetections& detections) override;
    CudaError Close() override;

private:
    CudaError Init();
    CudaError ConvertAndMap(EncodeResource& res, const float* batch_src, int batchIndex, cudaStream_t stream);

    std::string output_folder_;
    ModelProperties modelProps_;

    size_t batch_size_ = 0;

    std::unique_ptr<CudaStream> cuda_stream_;
    std::unique_ptr<ObjectTracker> tracker_;

    // Double-Buffered Hardware State Machine
    std::vector<EncodeResource> resource_pool_[2];
    std::vector<std::unique_ptr<NvJPEGEncoder>> hw_encoders_[2];
    std::unique_ptr<CudaEvent> dma_complete_event_[2];
    std::vector<std::future<std::vector<MMAPIEncodeStatus>>> pending_io_tasks_[2];
    int active_buffer_ = 0;
    bool is_closed_ = false;
};

} // namespace cropandweed
