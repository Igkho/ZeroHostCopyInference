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

    NVJpegSink(Token, std::string outputPath): output_path_(std::move(outputPath)) {}

    template <class SmartPtr>
    static CudaError Create(SmartPtr& out, std::string outputPath) {

        if constexpr (is_shared_ptr_v<SmartPtr>) {
            auto sink = std::make_shared<NVJpegSink>(Token{}, std::move(outputPath));
            CUDA_TRY(sink->Init());
            out = std::move(sink);
        } else {
            auto sink = std::unique_ptr<NVJpegSink>(new NVJpegSink(Token{}, std::move(outputPath)));
            CUDA_TRY(sink->Init());
            out = std::move(sink);
        }
        return CudaError();
    }

    ~NVJpegSink() override;

    CudaError Save(const BatchData& data, const BatchDetections &results) override;

private:
    CudaError PrintNVJpegVersion() const;

    CudaError Init();

    std::string output_path_;
    
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
