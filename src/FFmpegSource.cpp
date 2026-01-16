#include "FFmpegSource.h"
#include "helpers.h"
#include "FFmpegSourceKernels.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <filesystem>
#include <cuda.h>


extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
}

namespace fs = std::filesystem;

namespace cropandweed {

FFmpegSource::~FFmpegSource() {
    Cleanup();
}

void FFmpegSource::Cleanup() {
    if (gpuFrame_) {
        av_frame_free(&gpuFrame_);
    }
    if (pkt_) {
        av_packet_free(&pkt_);
    }
    if (decCtx_) {
        avcodec_free_context(&decCtx_);
    }
    if (fmtCtx_) {
        avformat_close_input(&fmtCtx_);
    }
    if (hwDeviceCtx_) {
        av_buffer_unref(&hwDeviceCtx_);
    }
}

CudaError FFmpegSource::Init() {

    if (avformat_open_input(&fmtCtx_, uri_.c_str(), nullptr, nullptr) < 0) {
        return CudaError(ERROR_SOURCE, "Could not open input: " + uri_);
    }

    if (avformat_find_stream_info(fmtCtx_, nullptr) < 0) {
        return CudaError(ERROR_SOURCE, "Could not find stream info");
    }

// FFmpeg 5.0 (Libavformat 59) changed the API to require const
#if LIBAVFORMAT_VERSION_MAJOR >= 59
    const AVCodec* decoder = nullptr;
#else
    AVCodec* decoder = nullptr;
#endif
    streamIndex_ = av_find_best_stream(fmtCtx_, AVMEDIA_TYPE_VIDEO, -1, -1, &decoder, 0);
    if (streamIndex_ < 0) {
        return CudaError(ERROR_SOURCE, "No video stream found");
    }

    CUDA_TRY(cudaFree(0));

    CUcontext primary_ctx = nullptr;
    CUresult cuRes = cuCtxGetCurrent(&primary_ctx);
    if (cuRes != CUDA_SUCCESS || !primary_ctx) {
        return CudaError(ERROR_SOURCE, "Failed to get current CUDA context.");
    }

    hwDeviceCtx_ = av_hwdevice_ctx_alloc(AV_HWDEVICE_TYPE_CUDA);
    if (!hwDeviceCtx_) {
        return CudaError(ERROR_SOURCE, "Failed to alloc FFmpeg HW context wrapper");
    }

    AVCUDADeviceContext* internal_ctx =
        (AVCUDADeviceContext*)((AVHWDeviceContext*)hwDeviceCtx_->data)->hwctx;
    internal_ctx->cuda_ctx = primary_ctx;
    internal_ctx->stream = NULL;

    if (av_hwdevice_ctx_init(hwDeviceCtx_) < 0) {
        return CudaError(ERROR_SOURCE, "Failed to init FFmpeg HW context with shared CUDA context");
    }

    CUDA_TRY(CudaStream::Create(cuda_stream_, cudaStreamNonBlocking));

    decCtx_ = avcodec_alloc_context3(decoder);
    if (!decCtx_) {
        return CudaError(ERROR_SOURCE, "Failed to allocate codec context");
    }
    avcodec_parameters_to_context(decCtx_, fmtCtx_->streams[streamIndex_]->codecpar);
    decCtx_->hw_device_ctx = av_buffer_ref(hwDeviceCtx_);

    if (avcodec_open2(decCtx_, decoder, nullptr) < 0) {
        return CudaError(ERROR_SOURCE, "Failed to open codec");
    }

    pkt_ = av_packet_alloc();
    gpuFrame_ = av_frame_alloc();

    width_ = decCtx_->width;
    height_ = decCtx_->height;

    return CudaError();
}

void FFmpegSource::SetOutputSize(size_t width, size_t height) {
    targetW_ = width;
    targetH_ = height;
}

CudaError FFmpegSource::GetNextBatch(BatchData& outBatch, size_t batchSize, bool &process) {
    if (finished_ && !flushing_) {
        process = false;
        return CudaError();
    }

    int outW = (targetW_ > 0) ? targetW_ : width_;
    int outH = (targetH_ > 0) ? targetH_ : height_;

    outBatch.batchId = frameCounter_ / batchSize;
    outBatch.width = outW;
    outBatch.height = outH;
    outBatch.batchSize = 0;
    outBatch.sourceIdentifiers.clear();

    size_t totalFloats = (outW * outH * 3) * batchSize;

    CUDA_TRY(outBatch.deviceData.resize(totalFloats));

    int framesCollected = 0;

    // --- ROBUST LOOP: DRAIN FIRST, THEN READ ---
    while (framesCollected < batchSize) {

        // 1. Try to receive pending frames (Drain Decoder)
        int ret = avcodec_receive_frame(decCtx_, gpuFrame_);

        if (ret == 0) {
            // Got Frame
            if (gpuFrame_->format == AV_PIX_FMT_CUDA) {
                int actualW = gpuFrame_->width;
                int actualH = gpuFrame_->height;

                float* batchBasePtr = outBatch.deviceData.data();
 //               size_t frameOffsetFloats = (size_t)framesCollected * (outW * outH * 3);

                CUDA_TRY(cudaStreamSynchronize(0));

                CUDA_TRY(NV12ToRGBPlanar(gpuFrame_->data[0],
                                         gpuFrame_->data[1],
                                         gpuFrame_->linesize[0],
                                         batchBasePtr,
                                         framesCollected,
                                         actualW, actualH,
                                         outW, outH,
                                         false,
                                         *cuda_stream_
                                         ));

                outBatch.sourceIdentifiers.push_back(std::to_string(frameCounter_++));
                framesCollected++;
            } else {
                // FATAL: The decoder fell back to software!
                std::cerr << "[Error] Frame " << frameCounter_
                          << " is format " << gpuFrame_->format
                          << " (Software). Expected AV_PIX_FMT_CUDA." << std::endl;
                return CudaError(ERROR_SOURCE, "Hardware decoding failed/fallback occurred.");
            }
            av_frame_unref(gpuFrame_);
            continue; // Keep draining
        }
        else if (ret == AVERROR_EOF) {
            finished_ = true;
            flushing_ = false;
            break;
        }
        else if (ret != AVERROR(EAGAIN)) {
            std::cerr << "Decoder Error: " << ret << std::endl;
            finished_ = true;
            break;
        }

        // 2. Decoder Empty (EAGAIN) -> Read Packet
        if (flushing_) {
            finished_ = true;
            flushing_ = false;
            break;
        }

        int readRet = av_read_frame(fmtCtx_, pkt_);
        if (readRet < 0) {
            // EOF -> Start Flush
            flushing_ = true;
            avcodec_send_packet(decCtx_, nullptr);
            continue; // Go back to receive loop
        }

        if (pkt_->stream_index == streamIndex_) {
            int sendRet = avcodec_send_packet(decCtx_, pkt_);
            if (sendRet < 0 && sendRet != AVERROR(EAGAIN)) {
                // Warning: Send failed
                std::cerr << "[FFMpegSource] packet sending failed" << std::endl;
            }
        }
        av_packet_unref(pkt_);
    }

    if (framesCollected > 0) {
        if (!outBatch.readyEvent) {
            CUDA_TRY(CudaEvent::Create(outBatch.readyEvent));
        }
        // Try to record
        cudaError_t err = cudaEventRecord(*outBatch.readyEvent, *cuda_stream_);

        // If handle is invalid (stale recycled event), recreate and retry
        if (err == cudaErrorInvalidResourceHandle) {
            std::cerr << "[FFmpegSource] Recycled event handle was invalid. Recreating..." << std::endl;
            CUDA_TRY(CudaEvent::Create(outBatch.readyEvent)); // Replaces old event
            CUDA_TRY(cudaEventRecord(*outBatch.readyEvent, *cuda_stream_)); // Retry
        } else {
            if (err != cudaSuccess) {
            // Propagate other errors using your helper macro logic manually
                return CudaError(ERROR_SOURCE, err);
            }
        }
    }

    outBatch.batchSize = framesCollected;
    process = framesCollected > 0;
    return CudaError();
}

} // namespace
