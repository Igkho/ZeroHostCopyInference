#include "MMAPIJpegSource.h"
#include "SourceKernels.h"
#include <filesystem>
#include <algorithm>
#include <fstream>
#include <iostream>

#include "NvJpegDecoder.h"
#include "nvbufsurface.h"
#include "nvbufsurftransform.h"
#include <cuda_egl_interop.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <unistd.h>

namespace fs = std::filesystem;

namespace cropandweed {

// Global EGL Initialization Helper
static void InitializeEGL() {
    static bool egl_initialized = false;
    if (!egl_initialized) {
        EGLDisplay egl_display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
        eglInitialize(egl_display, nullptr, nullptr);
        egl_initialized = true;
    }
}

// --- DecodeResource Implementation ---
DecodeResource::DecodeResource() {
    // Pre-reserve 10MB to completely avoid CPU heap reallocations during file IO
    raw_buffer.reserve(10 * 1024 * 1024);
}

void DecodeResource::DestroyHardwareSurfaces() {
    if (egl_cuda_resource) {
        cudaGraphicsUnregisterResource(egl_cuda_resource);
        egl_cuda_resource = nullptr;
    }
    if (pitch_linear_surf) {
        NvBufSurfaceUnMapEglImage(pitch_linear_surf, 0); // Unmap before destroy
        NvBufSurfaceDestroy(pitch_linear_surf);
        pitch_linear_surf = nullptr;
    }
    // if (block_linear_fd != -1) {
    //     close(block_linear_fd);
    //     block_linear_fd = -1;
    // }
    // Do not manually close block_linear_fd! NvJPEGDecoder owns it.
    block_linear_fd = -1;
}

DecodeResource::~DecodeResource() noexcept {
    DestroyHardwareSurfaces();
}

MMAPIJpegSource::MMAPIJpegSource(Token, std::string folderPath, int width, int height, size_t batch_size)
    : folder_path_(std::move(folderPath)), targetW_(width), targetH_(height), batch_size_(batch_size) {}

MMAPIJpegSource::~MMAPIJpegSource() noexcept {
    if (cuda_stream_) {
        CUDA_CALL_NO_THROW(cudaStreamSynchronize(*cuda_stream_));
    }
    // resource_pool_ will automatically trigger ~DecodeResource() and clean up all FDs.
}

CudaError MMAPIJpegSource::Create(std::unique_ptr<ISource>& out, std::string folderPath, int width, int height, size_t batch_size) {
    auto ptr = std::make_unique<MMAPIJpegSource>(Token{}, std::move(folderPath), width, height, batch_size);
    CUDA_TRY(ptr->Init());
    out = std::move(ptr);
    return CudaError();
}

CudaError MMAPIJpegSource::Init() {
    if (!fs::exists(folder_path_) || !fs::is_directory(folder_path_)) {
        return CudaError(ERROR_SOURCE, "Invalid folder path: " + folder_path_);
    }

    InitializeEGL();

    for (const auto& entry : fs::directory_iterator(folder_path_)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".jpg" || ext == ".jpeg") {
                file_list_.push_back(entry.path().string());
            }
        }
    }
    
    std::sort(file_list_.begin(), file_list_.end());

    CUDA_TRY(CudaStream::Create(cuda_stream_, cudaStreamNonBlocking));

    for (int b = 0; b < 2; ++b) {
        // Pre-allocate resource vectors
        resource_pool_[b].resize(batch_size_);
        futures_[b].resize(DecodeResource::HW_DECODERS_PER_BUFFER);
        hw_decoders_[b].resize(DecodeResource::HW_DECODERS_PER_BUFFER);
        for(int t = 0; t < DecodeResource::HW_DECODERS_PER_BUFFER; ++t) {
            hw_decoders_[b][t].reset(NvJPEGDecoder::createJPEGDecoder("jpegdec"));
        }
        // Create and PRE-TRIGGER the events so the first async threads don't deadlock
        CUDA_TRY(CudaEvent::Create(dma_complete_event_[b], cudaEventDisableTiming));
        CUDA_TRY(cudaEventRecord(*dma_complete_event_[b], *cuda_stream_));
        // Fire the background pre-fetching
        DispatchAsyncBatch(b);
    }
    return CudaError();
}

CudaError MMAPIJpegSource::MapAndConvert(DecodeResource& res, float* batch_dst,
                                         int batchIndex, cudaStream_t stream) {
    if (!res.egl_cuda_resource) {
        return CudaError(ERROR_SOURCE, "EGL CUDA resource is not registered");
    }

    cudaEglFrame eglFrame;
    // Extremely fast: just retrieves device pointers from the already registered resource
    CUDA_TRY(cudaGraphicsResourceGetMappedEglFrame(&eglFrame, res.egl_cuda_resource, 0, 0));

    // Extract standard Pitch Linear Pointers
    const uint8_t* srcY = static_cast<const uint8_t*>(eglFrame.frame.pPitch[0].ptr);
    const uint8_t* srcUV = static_cast<const uint8_t*>(eglFrame.frame.pPitch[1].ptr);
    int srcPitch = eglFrame.frame.pPitch[0].pitch;

    // Read data out of the surface
    CUDA_TRY(NV12ToRGBPlanar(srcY, srcUV, srcPitch,
                             batch_dst, batchIndex,
                             res.width, res.height, targetW_, targetH_, false, stream));
    return CudaError();
}

CudaError MMAPIJpegSource::GetNextBatch(BatchData& outBatch, size_t batchSize, bool& process) {
    int buf_idx = active_buffer_;

    // Save the frame count before dispatching the next batch
    int current_chunk_frames = frames_in_buffer_[buf_idx];

    // 1. Check for End of Stream
    if (frames_in_buffer_[buf_idx] == 0) {
        process = false;
        return CudaError();
    }

    if (batchSize > batch_size_) {
        return CudaError(ERROR_SOURCE, "Requested batchSize exceeds pre-allocated pool");
    }

    // 2. Gather Async Threads together
    std::vector<MMAPIDecodeStatus> valid_statuses;

    // Calculate EXACTLY how many threads were actually dispatched for this chunk
    int num_threads = std::min(current_chunk_frames, DecodeResource::HW_DECODERS_PER_BUFFER);

    // Unpack only the active threads
    for (int t = 0; t < num_threads; ++t) {
        std::vector<MMAPIDecodeStatus> chunk_statuses = futures_[buf_idx][t].get();
        for (const auto& status : chunk_statuses) {
            if (status.success) {
                valid_statuses.push_back(status);
            } else {
                std::cerr << "\n[MMAPIJpegSource] Warning: Skipping bad file '"
                          << status.filename << "': " << status.err.Text() << std::endl;
                if (status.new_pitch_linear_surf) {
                    NvBufSurfaceDestroy(status.new_pitch_linear_surf);
                }
            }
        }
    }

    int validCount = valid_statuses.size();
    if (validCount == 0) {
        // Free buffer, switch, and skip
        CUDA_TRY(cudaEventRecord(*dma_complete_event_[buf_idx], *cuda_stream_));
        active_buffer_ = (active_buffer_ + 1) & 0x01;
        frameCounter_ += frames_in_buffer_[buf_idx];
        DispatchAsyncBatch(buf_idx); // Start fetching the next chunk
        std::cerr << "\n[MMAPIJpegSource] Warning: Skipping fully bad batch: '"
                  << outBatch.batchId << std::endl;
        return CudaError(); // Let pipeline retry
    }

    // 3. Setup GPU Output Layout
    size_t framePixelsTarget = targetW_ * targetH_;
    CUDA_TRY(outBatch.deviceData.resize(batchSize * framePixelsTarget * 3, *cuda_stream_));
    outBatch.sourceIdentifiers.clear();

    // 4. Main-Thread EGL Mapping & CUDA Kernel Dispatch
    for (int k = 0; k < validCount; ++k) {
        const MMAPIDecodeStatus& status = valid_statuses[k];
        int i = status.batch_index;
        auto& res = resource_pool_[buf_idx][i];

        // Process Reallocation ONLY if the async thread detected a geometry change
        if (status.geometry_changed) {

            // Cleanly teardown old EGL mapping and Pitch Linear surface
            if (res.egl_cuda_resource) {
                CUDA_TRY(cudaGraphicsUnregisterResource(res.egl_cuda_resource));
                res.egl_cuda_resource = nullptr;
            }
            if (res.pitch_linear_surf) {
                CUDA_TRY(NvBufSurfaceUnMapEglImage(res.pitch_linear_surf, 0));
                CUDA_TRY(NvBufSurfaceDestroy(res.pitch_linear_surf));
                res.pitch_linear_surf = nullptr;
            }

            // Update resource trackers
            res.width = status.new_width;
            res.height = status.new_height;
            res.pixfmt = status.new_pixfmt;

            // Take ownership of the NEW surface allocated by the async thread
            res.pitch_linear_surf = status.new_pitch_linear_surf;

            // Re-register to EGL (Must be done in Main Thread Context!)
            CUDA_TRY(NvBufSurfaceMapEglImage(res.pitch_linear_surf, 0));
            EGLImageKHR egl_image = res.pitch_linear_surf->surfaceList[0].mappedAddr.eglImage;
            CUDA_TRY(cudaGraphicsEGLRegisterImage(&res.egl_cuda_resource, egl_image,
                                                  cudaGraphicsRegisterFlagsReadOnly));

        }

        // Dispatch CUDA Texture conversion
        CUDA_TRY(MapAndConvert(res, outBatch.deviceData.data(), k, *cuda_stream_));
        outBatch.sourceIdentifiers.push_back(std::to_string(frameCounter_ + i));

        // // True Filename Passthrough
        // std::string stem = fs::path(status.filename).stem().string();
        // outBatch.sourceIdentifiers.push_back(stem);
    }

    // 5. Zero-fill padding (if batch is incomplete)
    if (validCount < batchSize) {
        size_t offset = validCount * framePixelsTarget * 3;
        CUDA_TRY(outBatch.deviceData.fill_back(offset, 0.0f, *cuda_stream_));
    }

    // 6. Write the conversion finish event into the stream (Releases buffer for next cycle)
    CUDA_TRY(cudaEventRecord(*dma_complete_event_[buf_idx], *cuda_stream_));

    if (!outBatch.readyEvent) {
        CUDA_TRY(CudaEvent::Create(outBatch.readyEvent));
    }
    CUDA_TRY(cudaEventRecord(*outBatch.readyEvent, *cuda_stream_));

    // 7. Fire all async threads for the next cycle for this buffer
    DispatchAsyncBatch(buf_idx);

    // 8. Switch current buffers set
    outBatch.batchId = frameCounter_ / batchSize;
    frameCounter_ += current_chunk_frames;
    outBatch.batchSize = validCount;
    outBatch.width = targetW_;
    outBatch.height = targetH_;
    process = true;
    active_buffer_ = (active_buffer_ + 1) & 0x01;
    // std::cout << "Batch is decoded: " << outBatch.batchId << std::endl;

    return CudaError();
}

void MMAPIJpegSource::DispatchAsyncBatch(int buf_idx) {
    frames_in_buffer_[buf_idx] = 0;

    // Gather filenames for this specific batch
    std::vector<std::string> batch_filenames;
    while (frames_in_buffer_[buf_idx] < batch_size_ && current_file_idx_ < file_list_.size()) {
        batch_filenames.push_back(file_list_[current_file_idx_]);
        current_file_idx_++;
        frames_in_buffer_[buf_idx]++;
    }

    if (frames_in_buffer_[buf_idx] == 0) {
        return; // End of stream
    }

    int num_threads = std::min(frames_in_buffer_[buf_idx], DecodeResource::HW_DECODERS_PER_BUFFER);
    int frames_per_thread = (frames_in_buffer_[buf_idx] + num_threads - 1) / num_threads; // Ceiling division

    // Fire Async Threads
    for (int t = 0; t < num_threads; ++t) {
        futures_[buf_idx][t] = std::async(std::launch::async,
            [this, t, frames_per_thread, frames_in_buffer = frames_in_buffer_[buf_idx],
             batch_filenames, buf_idx]() ->
                                          std::vector<MMAPIDecodeStatus> {
            std::vector<MMAPIDecodeStatus> thread_statuses;
            int start_idx = t * frames_per_thread;
            int end_idx = std::min(start_idx + frames_per_thread, frames_in_buffer);

            // Fetch the shared hardware decoder assigned specifically to this thread
            auto& hw_decoder = hw_decoders_[buf_idx][t];

            // This single thread sequentially reads and decodes its chunk of files
            for (int i = start_idx; i < end_idx; ++i) {
                // Define AND immediately execute the lambda for this specific frame
                MMAPIDecodeStatus frame_status = [&, frame_idx = i]() -> MMAPIDecodeStatus {
                    MMAPIDecodeStatus status;
                    status.filename = batch_filenames[frame_idx];
                    status.batch_index = frame_idx; // Tell main thread where to put it in outBatch

                    // Fetch the dedicated memory surface for this EXACT frame!
                    auto& res = resource_pool_[buf_idx][frame_idx];

                    // A. Read File to RAM
                    std::ifstream file(batch_filenames[frame_idx], std::ios::in | std::ios::binary | std::ios::ate);
                    if (!file) {
                        status.err = CudaError(ERROR_SOURCE, "Cannot read");
                        return status;
                    }

                    size_t file_size = file.tellg();
                    file.seekg(0, std::ios::beg);
                    res.raw_buffer.resize(file_size);
                    file.read(reinterpret_cast<char*>(res.raw_buffer.data()), file_size);

                    // Magic byte check protects the shared V4L2 hardware decoder from entering a global
                    // SMMU fault when fed blatantly invalid non-JPEG files.
                    if (file_size < 2 || res.raw_buffer[0] != 0xFF || res.raw_buffer[1] != 0xD8) {
                        status.err = CudaError(ERROR_SOURCE, "Invalid JPEG magic bytes. Skipping to protect V4L2 engine.");
                        return status;
                    }
                    // Wait for GPU to release this buffer (Condition always met on first run)
                    CUDA_TRY_LAMBDA(cudaEventSynchronize(*dma_complete_event_[buf_idx]), status);

                    // Hardware Decode
                    uint32_t width, height, pixfmt;
                    CUDA_TRY_LAMBDA(hw_decoder->decodeToFd(res.block_linear_fd, res.raw_buffer.data(),
                                                            file_size, pixfmt, width, height), status);

                    // D. Check for Geometry Changes
                    if (!res.pitch_linear_surf || width != res.width || height != res.height || pixfmt != res.pixfmt) {
                        // PROPOSE: Tell main thread to update the page tables and memory
                        status.geometry_changed = true;
                        status.new_width = width;
                        status.new_height = height;
                        status.new_pixfmt = pixfmt;

                        // Allocate the NEW surface safely in the background thread
                        NvBufSurfaceCreateParams create_params{};
                        create_params.gpuId = 0;
                        create_params.width = width;
                        create_params.height = height;
                        create_params.size = 0;
                        create_params.colorFormat = NVBUF_COLOR_FORMAT_NV12;
                        create_params.layout = NVBUF_LAYOUT_PITCH;
                        create_params.memType = NVBUF_MEM_SURFACE_ARRAY;

                        CUDA_TRY_LAMBDA(NvBufSurfaceCreate(&status.new_pitch_linear_surf, 1, &create_params), status);

                        // Perform VIC Transform IMMEDIATELY to prevent FD invalidation by the next loop iteration
                        NvBufSurface *pDecodeSurf = nullptr;
                        CUDA_TRY_LAMBDA(NvBufSurfaceFromFd(res.block_linear_fd, (void**)&pDecodeSurf), status);
                        NvBufSurfTransformParams transform_params{};
                        transform_params.transform_flag = NVBUFSURF_TRANSFORM_FILTER;
                        transform_params.transform_filter = NvBufSurfTransformInter_Nearest;
                        CUDA_TRY_LAMBDA(NvBufSurfTransform(pDecodeSurf, status.new_pitch_linear_surf, &transform_params), status);
                    } else {
                        // NORMAL PATH: Memory already matches! Run the VIC transform.
                        status.geometry_changed = false;
                        NvBufSurface *pDecodeSurf = nullptr;
                        CUDA_TRY_LAMBDA(NvBufSurfaceFromFd(res.block_linear_fd, (void**)&pDecodeSurf), status);
                        NvBufSurfTransformParams transform_params{};
                        transform_params.transform_flag = NVBUFSURF_TRANSFORM_FILTER;
                        transform_params.transform_filter = NvBufSurfTransformInter_Nearest;
                        CUDA_TRY_LAMBDA(NvBufSurfTransform(pDecodeSurf, res.pitch_linear_surf, &transform_params), status);
                    }
                    status.success = true;
                    return status;
                }();
                thread_statuses.push_back(frame_status);
            }
            return thread_statuses;
        });
    }
}

} // namespace cropandweed
