#pragma once
#include <string>
#include <vector>
#include <memory>
#include <nvjpeg.h>
#include <cuda_runtime.h>
#include "Interfaces.h"
#include "Block.h"
#include "helpers.h"
#include "ObjectTracker.h"

namespace cropandweed {

// Encapsulated state for Double Buffered SSD writes
struct EncodeState {
    HostStagingBlock<uint8_t> pinned_buffer;
    std::vector<size_t> lengths;
    std::vector<std::string> filenames;
    std::unique_ptr<CudaEvent> dma_complete_event;
    bool has_data = false;
    int batch_size = 0;
};

class NVJpegSink : public ISink {
private:
    struct Token {};
public:

    NVJpegSink(Token, std::string outputPath, ModelProperties props)
        : output_path_(std::move(outputPath)), modelProps_(std::move(props)) {}

    static CudaError Create(std::unique_ptr<ISink>& out, std::string outputPath, ModelProperties props) {
        auto sink = std::make_unique<NVJpegSink>(Token{}, std::move(outputPath), std::move(props));
        CUDA_TRY(sink->Init());
        out = std::move(sink);
        return CudaError();
    }

    ~NVJpegSink() override;

    CudaError Save(BatchData& data, BatchDetections &results) override;

    // Explicit pipeline termination method
    CudaError Close() override;

private:
    CudaError CheckNVJpegVersion() const;

    CudaError Init();

    // Encapsulated helper for parallel SSD flushing
    CudaError FlushBufferToDisk(EncodeState& buf);

    std::string output_path_;
    ModelProperties modelProps_;

    std::unique_ptr<CudaStream> cuda_stream_;
    std::unique_ptr<ObjectTracker> tracker_;

    nvjpegHandle_t nvjpeg_handle_ = nullptr;
    nvjpegEncoderParams_t encode_params_ = nullptr;
    std::vector<nvjpegEncoderState_t> encoder_states_;

    Block<uint8_t> device_decode_buffer_;

    // Double Buffering Execution State
    EncodeState staging_buffers_[2];
    int active_buffer_ = 0;
};

} // namespace cropandweed
