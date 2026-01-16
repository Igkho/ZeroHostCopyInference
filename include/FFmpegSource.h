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

    FFmpegSource(Token, std::string uri): uri_(std::move(uri)) {};

    ~FFmpegSource() override;

    // --- Strict Factory Method ---
    static CudaError Create(std::unique_ptr<ISource>& out, std::string uri) {
        auto ptr = std::make_unique<FFmpegSource>(Token{}, uri);
        CUDA_TRY(ptr->Init());
        // Implicit move-conversion from unique_ptr<FFmpegSource> to unique_ptr<ISource>
        out = std::move(ptr);
        return CudaError();
    }

    CudaError GetNextBatch(BatchData& outBatch, size_t batchSize, bool &process) override;
    void SetOutputSize(size_t width, size_t height) override;

private:
    CudaError Init();
    void Cleanup();

    // DEBUG: Saves the GPU buffer to disk as PPM
//    CudaError SaveDebugPPM(float* gpuRGB, int w, int h, int frameId);

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
