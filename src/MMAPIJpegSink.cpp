#include "MMAPIJpegSink.h"
#include "SinkKernels.h"
#include <filesystem>
#include <fstream>
#include <iostream>

#include "NvJpegEncoder.h"
#include "nvbufsurface.h"
#include "nvbufsurftransform.h"
#include <cuda_egl_interop.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <unistd.h>

namespace fs = std::filesystem;

namespace cropandweed {

void EncodeResource::DestroyHardwareSurfaces() {
    if (egl_cuda_resource) {
        CUDA_CALL_NO_THROW(cudaGraphicsUnregisterResource(egl_cuda_resource));
        egl_cuda_resource = nullptr;
    }
    if (pitch_linear_surf) {
        NvBufSurfaceUnMapEglImage(pitch_linear_surf, 0); 
        NvBufSurfaceDestroy(pitch_linear_surf);
        pitch_linear_surf = nullptr;
    }
}

EncodeResource::~EncodeResource() noexcept {
    DestroyHardwareSurfaces();
}

// Explicit teardown that propagates the error to the pipeline
CudaError MMAPIJpegSink::Close() {
    // Prevent double-closing
    if (is_closed_) {
        return CudaError();
    }
    is_closed_ = true;

    CudaError final_err; // Defaults to success

    if (cuda_stream_) {
        // Manually capture the raw CUDA error to propagate
        cudaError_t err = cudaStreamSynchronize(*cuda_stream_);
        if (err != cudaSuccess) {
            CudaError stream_err(ERROR_SOURCE, std::string("Stream sync failed: ") +
                                                   cudaGetErrorString(err));
            if (!CudaError::IsFailure(final_err)) {
                final_err = stream_err;
            }
        }
    }

    // Strictly await all background threads across both buffers before tearing down pools
    for (int b = 0; b < 2; ++b) {
        for (auto& task : pending_io_tasks_[b]) {
            if (task.valid()) {
                try {
                    std::vector<MMAPIEncodeStatus> statuses = task.get();
                    for (const auto& status : statuses) {
                        // Latch the first failure we see, but keep looping to join the remaining threads!
                        if (!status.success && !CudaError::IsFailure(final_err)) {
                            final_err = status.err;
                        }
                    }
                } catch (const std::exception& e) {
                    if (!CudaError::IsFailure(final_err)) {
                        final_err = CudaError(ERROR_SOURCE, std::string("Async task exception: ") + e.what());
                    }
                } catch (...) {
                    if (!CudaError::IsFailure(final_err)) {
                        final_err = CudaError(ERROR_SOURCE, "Unknown async task exception");
                    }
                }
            }
        }
        pending_io_tasks_[b].clear();
    }
    // Propagate the latched error back to InferencePipeline::outputWorker
    return final_err;
}

MMAPIJpegSink::MMAPIJpegSink(Token, std::string output_folder, ModelProperties modelProps, size_t batch_size)
    : output_folder_(std::move(output_folder)),
    modelProps_(std::move(modelProps)),
    batch_size_(batch_size) {}

// Safe, silent destructor for RAII stack unrolling
MMAPIJpegSink::~MMAPIJpegSink() noexcept {
    if (!is_closed_) {
        try {
            CudaError err = Close();
            // If the destructor had to clean up and found an error, just log it.
            if (CudaError::IsFailure(err)) {
                std::cerr << "[MMAPIJpegSink] Destructor swallowed Close() error: " << err.Text() << std::endl;
            }
        } catch (const std::exception& e) {
            try {
                std::cerr << "[MMAPIJpegSink] Destructor caught exception: " << e.what() << std::endl;
            } catch (...) {} // Swallow OOM during logging
        } catch (...) {
            try {
                std::cerr << "[MMAPIJpegSink] Destructor caught unknown exception." << std::endl;
            } catch (...) {} // Swallow OOM during logging
        }
    }
}

CudaError MMAPIJpegSink::Create(std::unique_ptr<ISink>& out, std::string output_folder,
                                ModelProperties modelProps, size_t batch_size) {
    auto ptr = std::make_unique<MMAPIJpegSink>(Token{}, std::move(output_folder),
                                               std::move(modelProps), batch_size);
    CUDA_TRY(ptr->Init());
    out = std::move(ptr);
    return CudaError();
}

CudaError MMAPIJpegSink::Init() {
    if (!fs::exists(output_folder_)) {
        fs::create_directories(output_folder_);
    }

    CUDA_TRY(CudaStream::Create(cuda_stream_, cudaStreamNonBlocking));
    CUDA_TRY(ObjectTracker::Create(tracker_, modelProps_.numClasses, *cuda_stream_));

    // Maximum possible size for an NV12 frame (Width * Height * 1.5 bytes)
    size_t max_jpeg_size = modelProps_.inputWidth * modelProps_.inputHeight * 3 / 2;

    // Pre-allocate EVERYTHING for both ping-pong buffers
    for (int b = 0; b < 2; ++b) {
        CUDA_TRY(CudaEvent::Create(dma_complete_event_[b], cudaEventDisableTiming));
        resource_pool_[b].resize(batch_size_);

        hw_encoders_[b].resize(EncodeResource::HW_ENCODERS_PER_BUFFER);
        for(int t = 0; t < EncodeResource::HW_ENCODERS_PER_BUFFER; ++t) {
            hw_encoders_[b][t].reset(NvJPEGEncoder::createJPEGEncoder("jpegenc"));
        }

        for (size_t i = 0; i < batch_size_; ++i) {
            EncodeResource& res = resource_pool_[b][i];
            res.width = modelProps_.inputWidth;
            res.height = modelProps_.inputHeight;
            
            // Pre-allocate CPU vector to max possible size to prevent malloc during encode
            res.compressed_buffer.resize(max_jpeg_size);

            // A. Allocate Pitch Linear for CUDA Kernel
            NvBufSurfaceCreateParams pitch_params{};
            pitch_params.gpuId = 0;
            pitch_params.width = modelProps_.inputWidth;
            pitch_params.height = modelProps_.inputHeight;
            pitch_params.size = 0;
            pitch_params.colorFormat = NVBUF_COLOR_FORMAT_NV12;
            pitch_params.layout = NVBUF_LAYOUT_PITCH;
            pitch_params.memType = NVBUF_MEM_SURFACE_ARRAY;

            CUDA_TRY(NvBufSurfaceCreate(&res.pitch_linear_surf, 1, &pitch_params));
        }
    }

    return CudaError();
}

CudaError MMAPIJpegSink::ConvertAndMap(EncodeResource& res, const float* batch_src,
                                       int batchIndex, cudaStream_t stream) {

    // Explicitly acquire the hardware lock
    CUDA_TRY(NvBufSurfaceMapEglImage(res.pitch_linear_surf, 0));

    // Register to CUDA for this frame to acquire the write lock
    EGLImageKHR egl_image = res.pitch_linear_surf->surfaceList[0].mappedAddr.eglImage;
    CUDA_TRY(cudaGraphicsEGLRegisterImage(&res.egl_cuda_resource, egl_image,
                                          cudaGraphicsRegisterFlagsWriteDiscard));

    cudaEglFrame eglFrame;
    CUDA_TRY(cudaGraphicsResourceGetMappedEglFrame(&eglFrame, res.egl_cuda_resource, 0, 0));

    uint8_t* dstY = static_cast<uint8_t*>(eglFrame.frame.pPitch[0].ptr);
    uint8_t* dstUV = static_cast<uint8_t*>(eglFrame.frame.pPitch[1].ptr);
    int pitch = eglFrame.frame.pPitch[0].pitch;

    // Write data into the surface
    CUDA_TRY(RGBPlanarToNV12(batch_src, dstY, dstUV, pitch, batchIndex, res.width, res.height, stream));

    return CudaError();
}

CudaError MMAPIJpegSink::Save(BatchData& batch, BatchDetections& detections) {
    if (batch.batchSize == 0) return CudaError();

    int buf_idx = active_buffer_;

    // 1. Wait for Inference & Drawing to finish modifying the batch array
    if (batch.readyEvent) {
        CUDA_TRY(cudaStreamWaitEvent(*cuda_stream_, *batch.readyEvent, 0));
    }
    if (detections.readyEvent) {
        CUDA_TRY(cudaStreamWaitEvent(*cuda_stream_, *detections.readyEvent, 0));
    }

    int stride = BatchDetections::MAX_DETECTIONS_PER_FRAME;

    // 2. Run Tracking (Sequential per frame in batch)
    for (int i = 0; i < batch.batchSize; ++i) {
        CUDA_TRY(tracker_->ProcessBatch(
            i,
            detections.data,
            detections.counts,
            stride,
            (int)batch.width,
            (int)batch.height,
            *cuda_stream_
            ));
    }
    CUDA_TRY(tracker_->Compact(*cuda_stream_));

    // 3. Run Annotation
    CUDA_TRY(tracker_->Annotate(
        batch.deviceData,
        batch.batchSize,
        batch.width, batch.height,
        detections.data,
        detections.counts,
        *cuda_stream_
        ));

    // 4. Sink CPU sync: Ensure the background threads from the *previous* cycle of this specific buffer are done
    for (auto& task : pending_io_tasks_[buf_idx]) {
        if (task.valid()) {
            std::vector<MMAPIEncodeStatus> chunk_statuses = task.get();
            for (const auto& status : chunk_statuses) {
                // If the background thread returned an error, log it but DO NOT forward it up.
                // This prevents a single bad file write/encode from crashing the whole pipeline.
                if (!status.success) {
                    std::cerr << "\n[MMAPIJpegSink] Warning: Async hardware encode/IO failed for file:\n"
                              << status.filename << "':\n" << status.err.Text() << std::endl;
                }
            }
        }
    }
    pending_io_tasks_[buf_idx].clear();

    // 5. Queue the colorspace conversion kernels
    for (size_t i = 0; i < batch.batchSize; ++i) {
        CUDA_TRY(ConvertAndMap(resource_pool_[buf_idx][i], batch.deviceData.data(), i, *cuda_stream_));
    }

    // // We must wait for CUDA to finish, then release the EGL locks in the SAME thread that mapped them!
    CUDA_TRY(cudaStreamSynchronize(*cuda_stream_));

    for (size_t i = 0; i < batch.batchSize; ++i) {
        auto& res = resource_pool_[buf_idx][i];
        if (res.egl_cuda_resource) {
            CUDA_TRY(cudaGraphicsUnregisterResource(res.egl_cuda_resource));
            res.egl_cuda_resource = nullptr;
        }
       CUDA_TRY(NvBufSurfaceUnMapEglImage(res.pitch_linear_surf, 0));
    }

    // 6. Record event
    CUDA_TRY(cudaEventRecord(*dma_complete_event_[buf_idx], *cuda_stream_));

    // 7. Dispatch Chunked Async CPU Threads to handle Hardware Encoding & SSD Writes
    int num_threads = std::min((int)batch.batchSize, EncodeResource::HW_ENCODERS_PER_BUFFER);
    int frames_per_thread = (batch.batchSize + num_threads - 1) / num_threads;

    for (int t = 0; t < num_threads; ++t) {
        pending_io_tasks_[buf_idx].push_back(std::async(std::launch::async,
            [this, t, frames_per_thread, batch_size = batch.batchSize, batch_id = batch.batchId,
             identifiers = batch.sourceIdentifiers, buf_idx]() -> std::vector<MMAPIEncodeStatus> {

            std::vector<MMAPIEncodeStatus> thread_statuses;

            // A. Await GPU completion inside the background thread!
            // If the event sync fails, the whole chunk is invalid
            cudaError_t sync_err = cudaEventSynchronize(*dma_complete_event_[buf_idx]);
            if (sync_err != cudaSuccess) {
                MMAPIEncodeStatus err_status;
                err_status.err = CudaError(ERROR_SOURCE, sync_err);
                thread_statuses.push_back(err_status);
                return thread_statuses;
            }

            int start_idx = t * frames_per_thread;
            int end_idx = std::min(start_idx + frames_per_thread, (int)batch_size);

            // Fetch the fixed hardware encoder dedicated to this specific thread
            auto& hw_encoder = hw_encoders_[buf_idx][t];

            for (int i = start_idx; i < end_idx; ++i) {
                // Isolate errors to this specific frame
                MMAPIEncodeStatus frame_status = [&, frame_idx = i]() -> MMAPIEncodeStatus {
                    MMAPIEncodeStatus status;
                    // status.filename = output_folder_ + "/out_" + identifiers[frame_idx] + ".jpg";
                    std::string id = (frame_idx < identifiers.size() && !identifiers[frame_idx].empty())
                                         ? identifiers[frame_idx]
                                         : std::to_string(batch_id * batch_size + frame_idx);

                    std::stringstream ss;
                    ss << "frame_" << std::setw(4) << std::setfill('0') << id << ".jpg";
                    status.filename = output_folder_ + "/" + ss.str();

                    auto& res = resource_pool_[buf_idx][frame_idx];

                    // C. Hardware Encode
                    unsigned long encode_size = res.compressed_buffer.capacity();
                    unsigned char* out_buf_ptr = res.compressed_buffer.data();
                    int pitch_fd = res.pitch_linear_surf->surfaceList[0].bufferDesc;

                    CUDA_TRY_LAMBDA(hw_encoder->encodeFromFd(pitch_fd, JCS_YCbCr, &out_buf_ptr,
                                                             encode_size, 95), status);

                    // D. Flush sequentially to SSD
                    std::ofstream outfile(status.filename, std::ios::out | std::ios::binary);
                    if (!outfile) {
                        status.err = CudaError(ERROR_SOURCE, "Failed to open output file: " +
                                                                 status.filename);
                        return status;
                    }
                    outfile.write(reinterpret_cast<const char*>(out_buf_ptr), encode_size);
                    if (outfile.fail()) {
                        status.err = CudaError(ERROR_SOURCE,
                                    "Failed to write encoded JPEG data to SSD. Disk full? File: \n" +
                                                   status.filename);
                        return status;
                    }
                    status.success = true;
                    return status;
                }();

                thread_statuses.push_back(frame_status);
            }
            return thread_statuses;
        }));
    }

    // std::cout << "Batch is encoded: " << batch.batchId << std::endl;

    // Swap active buffer index for the next Save() call
    active_buffer_ = (active_buffer_ + 1) & 0x01;

    return CudaError();
}

} // namespace cropandweed
