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
struct DecodeResult {
    CudaError err;
    int width = 0;
    int height = 0;
    int channels = 0;
    bool success = false;
};

class NVJpegSource : public ISource {
private:
    struct Token {};
public:
    NVJpegSource(Token, std::string folderPath, int width, int height)
        : folder_path_(std::move(folderPath)), targetW_(width), targetH_(height) {}

    ~NVJpegSource() override;

    static CudaError Create(std::unique_ptr<ISource>& out, std::string folderPath, int width, int height);

    CudaError GetNextBatch(BatchData& outBatch, size_t batchSize, bool &process) override;

private:
    CudaError Init();

    std::string folder_path_;
    std::vector<std::string> file_list_;
    size_t current_file_idx_ = 0;
    size_t frameCounter_ = 0;

    int targetW_ = 0;
    int targetH_ = 0;

    std::unique_ptr<CudaStream> cuda_stream_;
    
    // Decoupled nvJPEG Resources
    nvjpegHandle_t nvjpeg_handle_ = nullptr;
    nvjpegJpegDecoder_t jpeg_decoder_ = nullptr;
    nvjpegDecodeParams_t decode_params_ = nullptr;

    // Double Buffering at the GPU Boundary
    std::vector<nvjpegJpegState_t> decoupled_states_[2];
    std::vector<nvjpegJpegStream_t> jpeg_streams_[2];
    std::vector<nvjpegBufferPinned_t> pinned_buffers_[2];
    std::vector<nvjpegBufferDevice_t> device_buffers_[2];
    std::unique_ptr<CudaEvent> dma_complete_event_[2];
    int active_buffer_ = 0;

    Block<uint8_t> device_decode_buffer_;             // Holds the raw decoded uint8_t pixels
};

}
