#pragma once
#include "Interfaces.h"
#include "Block.h"
#include "helpers.h"
#include <vector>
#include <string>
#include <future>

// Forward declarations for Jetson MMAPI
class NvJPEGDecoder;
struct NvBufSurface;

namespace cropandweed {

struct DecodeResource {
    static constexpr int HW_DECODERS_PER_BUFFER = 8;
    std::vector<uint8_t> raw_buffer;
    int block_linear_fd = -1;
    NvBufSurface* pitch_linear_surf = nullptr;
    cudaGraphicsResource_t egl_cuda_resource = nullptr;

    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t pixfmt = 0;

    DecodeResource();
    ~DecodeResource();

    // Helper to safely clean up hardware resources when geometry changes
    void DestroyHardwareSurfaces();

    DecodeResource(const DecodeResource&) = delete;
    DecodeResource& operator=(const DecodeResource&) = delete;
    DecodeResource(DecodeResource&& other) noexcept {
        raw_buffer = std::move(other.raw_buffer);
        block_linear_fd = other.block_linear_fd;
        pitch_linear_surf = other.pitch_linear_surf;
        egl_cuda_resource = other.egl_cuda_resource;
        width = other.width;
        height = other.height;
        pixfmt = other.pixfmt;

        // NULLIFY THE SOURCE to prevent double-free!
        other.block_linear_fd = -1;
        other.pitch_linear_surf = nullptr;
        other.egl_cuda_resource = nullptr;
    }
    DecodeResource& operator=(DecodeResource&&) noexcept = delete;
};

struct MMAPIDecodeStatus {
    CudaError err;
    bool success = false;
    std::string filename;
    int batch_index = -1;

    // Reallocation flags
    bool geometry_changed = false;
    uint32_t new_width = 0;
    uint32_t new_height = 0;
    uint32_t new_pixfmt = 0;

    NvBufSurface* new_pitch_linear_surf = nullptr;
};

class MMAPIJpegSource : public ISource {
private:
    struct Token {};

public:
    MMAPIJpegSource(Token, std::string folderPath, int width, int height, size_t batch_size);

    ~MMAPIJpegSource() noexcept override;

    static CudaError Create(std::unique_ptr<ISource>& out, std::string folderPath,
                            int width, int height, size_t batch_size);

    CudaError GetNextBatch(BatchData& outBatch, size_t batchSize, bool &process) override;

private:
    CudaError Init();
    CudaError MapAndConvert(DecodeResource &res, float* batch_dst, int batchIndex, cudaStream_t stream);

    std::string folder_path_;
    std::vector<std::string> file_list_;
    size_t current_file_idx_ = 0;
    size_t frameCounter_ = 0;

    int targetW_ = 0;
    int targetH_ = 0;
    size_t batch_size_ = 0;

    std::unique_ptr<CudaStream> cuda_stream_;
    std::unique_ptr<CudaEvent> dma_complete_event_[2];
    int active_buffer_ = 0;

    std::vector<DecodeResource> resource_pool_[2];
    std::vector<std::unique_ptr<NvJPEGDecoder>> hw_decoders_[2];
    std::vector<std::future<std::vector<MMAPIDecodeStatus>>> futures_[2];
    int frames_in_buffer_[2] = {0, 0};

    // Helper to fire off background threads
    void DispatchAsyncBatch(int buf_idx);
};

} // namespace cropandweed
