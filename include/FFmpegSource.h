#pragma once
#include "Interfaces.h"
#include "helpers.h"
#include "Block.h"
#include <string>
#include <memory>
#include <type_traits>

// Forward declarations for FFmpeg structs
struct AVFormatContext;
struct AVCodecContext;
struct AVBufferRef;
struct AVFrame;
struct AVPacket;

namespace cropandweed {

class FFmpegSource : public ISource {
private:
    struct Token {};
public:

    FFmpegSource(Token, std::string uri, int width, int height):
        uri_(std::move(uri)), targetW_(width), targetH_(height) {}

    ~FFmpegSource() override;

    // --- Strict Factory Method ---
    static CudaError Create(std::unique_ptr<ISource>& out, std::string uri, int width, int height) {
        auto ptr = std::make_unique<FFmpegSource>(Token{}, uri, width, height);
        CUDA_TRY(ptr->Init());
        out = std::move(ptr);
        return CudaError();
    }

    CudaError GetNextBatch(BatchData& outBatch, size_t batchSize, bool &process) override;

private:
    CudaError Init();
    void Cleanup();

    std::string uri_;
    size_t frameCounter_ = 0;
    size_t width_ = 0;
    size_t height_ = 0;
    size_t targetW_ = 0;
    size_t targetH_ = 0;

    std::unique_ptr<CudaStream> cuda_stream_;

    // FFmpeg State
    AVFormatContext* fmtCtx_ = nullptr;
    AVCodecContext* decCtx_ = nullptr;
    AVBufferRef* hwDeviceCtx_ = nullptr;
    AVFrame* gpuFrame_ = nullptr;
    AVPacket* pkt_ = nullptr;
    int streamIndex_ = -1;

    bool finished_ = false;
    bool flushing_ = false;
};

}
