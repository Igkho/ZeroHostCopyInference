#pragma once
#include "Interfaces.h"
#include "Block.h"
#include "helpers.h"
#include <nvjpeg.h>
#include <vector>
#include <string>
#include <future>

namespace cropandweed {

// Structure to capture parallel thread results safely
struct NVJpegDecodeStatus {
    CudaError err;
    bool success = false;
    std::string filename;
    int batch_index = -1; // Tell main thread where to put it
    int width = 0;
    int height = 0;
    int channels = 0;
};

// Resource tracking struct per frame for memory isolation
struct JpegDecodeResource {
    Block<uint8_t> decoded_pixels; // Native RGB bytes
    int width = 0;
    int height = 0;
    int channels = 0;
};


class NVJpegSource : public ISource {
private:
    struct Token {};
public:
    NVJpegSource(Token, std::string folderPath, int width, int height, size_t batch_size)
        : folder_path_(std::move(folderPath)), targetW_(width), targetH_(height),
          batch_size_(batch_size) {}

    ~NVJpegSource() override;

    static CudaError Create(std::unique_ptr<ISource>& out, std::string folderPath,
                            int width, int height, size_t batch_size);

    CudaError GetNextBatch(BatchData& outBatch, size_t batchSize, bool &process) override;

private:
    CudaError Init();

    // Helper method to fire off true background pre-fetching
    void DispatchAsyncBatch(int buf_idx);

    std::string folder_path_;
    std::vector<std::string> file_list_;
    size_t current_file_idx_ = 0;
    size_t frameCounter_ = 0;

    int targetW_ = 0;
    int targetH_ = 0;
    size_t batch_size_ = 0;

    std::unique_ptr<CudaStream> cuda_stream_;
    
    // Decoupled nvJPEG Resources
    nvjpegHandle_t nvjpeg_handle_ = nullptr;
    nvjpegJpegDecoder_t jpeg_decoder_ = nullptr;
    nvjpegDecodeParams_t decode_params_ = nullptr;

    // Fixed pool of logical decoders
    static constexpr int DECODERS_PER_BUFFER = 2;

    // Double Buffering at the GPU Boundary
    std::vector<nvjpegJpegState_t> decoupled_states_[2];
    std::vector<nvjpegJpegStream_t> jpeg_streams_[2];
    std::vector<nvjpegBufferPinned_t> pinned_buffers_[2];
    std::vector<nvjpegBufferDevice_t> device_buffers_[2];
    // Dedicated streams for async threads
    std::vector<std::unique_ptr<CudaStream>> decode_streams_[2];
    std::unique_ptr<CudaEvent> dma_complete_event_[2];
    int active_buffer_ = 0;

    // Resource pooling and futures mapping
    std::vector<JpegDecodeResource> resource_pool_[2];
    std::vector<std::future<std::vector<NVJpegDecodeStatus>>> futures_[2];
    int frames_in_buffer_[2] = {0, 0};
};

}
