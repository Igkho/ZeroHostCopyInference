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

    CudaError Save(const BatchData& data, const BatchDetections &results) override;

private:
    CudaError CheckNVJpegVersion() const;

    CudaError Init();

    std::string output_path_;
    ModelProperties modelProps_;

    // nvJPEG resources
    nvjpegHandle_t nvjpeg_handle_ = nullptr;
    std::vector<nvjpegEncoderState_t> encoder_states_;
    nvjpegEncoderParams_t encode_params_ = nullptr;
    std::unique_ptr<CudaStream> cuda_stream_;
    std::unique_ptr<ObjectTracker> tracker_;
    Block<uint8_t> buffer_block_;
    Block<uint8_t, MemoryType::Pinned> pinned_buffer_;
};

} // namespace cropandweed
