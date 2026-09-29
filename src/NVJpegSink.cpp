#include "NVJpegSink.h"
#include "SinkKernels.h"
#include "helpers.h"
#include <numeric>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <sstream>
#include <iomanip>
#include <future>

namespace fs = std::filesystem;

namespace cropandweed {

CudaError NVJpegSink::Init() {
    if (!fs::exists(output_path_)) {
        fs::create_directories(output_path_);
    }

    CUDA_TRY(CudaStream::Create(cuda_stream_, cudaStreamNonBlocking));
    CUDA_TRY(ObjectTracker::Create(tracker_, modelProps_.numClasses, *cuda_stream_));
    CUDA_TRY(nvjpegCreateSimple(&nvjpeg_handle_));

    CUDA_TRY(nvjpegEncoderParamsCreate(nvjpeg_handle_, &encode_params_, *cuda_stream_));
    CUDA_TRY(nvjpegEncoderParamsSetSamplingFactors(encode_params_, NVJPEG_CSS_444, *cuda_stream_));
    CUDA_TRY(nvjpegEncoderParamsSetQuality(encode_params_, 90, *cuda_stream_));
    CUDA_TRY(CheckNVJpegVersion());
    size_t max_jpeg_reservation = BatchData::MAX_BATCH_SIZE * 10 * 1024 * 1024; // 10MB per image max

    // Initialize logical pools per buffer to guarantee thread/stream isolation
    for (int b = 0; b < 2; ++b) {
        EncodeState& buf = staging_buffers_[b];
        CUDA_TRY(buf.pinned_buffer.reserve(max_jpeg_reservation, *cuda_stream_));
        buf.lengths.resize(BatchData::MAX_BATCH_SIZE);

        buf.encoder_states.resize(EncodeState::ENCODERS_PER_BUFFER);
        buf.encode_streams.resize(EncodeState::ENCODERS_PER_BUFFER);

        for (int t = 0; t < EncodeState::ENCODERS_PER_BUFFER; ++t) {
            CUDA_TRY(CudaStream::Create(buf.encode_streams[t], cudaStreamNonBlocking));
            CUDA_TRY(nvjpegEncoderStateCreate(nvjpeg_handle_, &buf.encoder_states[t], *buf.encode_streams[t]));
        }

        CUDA_TRY(CudaEvent::Create(buf.dma_complete_event, cudaEventDisableTiming));
        // Prime the event so the first cycle doesn't deadlock on sync
        CUDA_TRY(cudaEventRecord(*buf.dma_complete_event, *cuda_stream_));
    }

    return CudaError();
}

CudaError NVJpegSink::CheckNVJpegVersion() const {

    int rtMajor, rtMinor;
    CUDA_TRY(nvjpegGetProperty(MAJOR_VERSION, &rtMajor));
    CUDA_TRY(nvjpegGetProperty(MINOR_VERSION, &rtMinor));
    int cMajor = NVJPEG_VER_MAJOR;
    int cMinor = NVJPEG_VER_MINOR;
    int cPatch = NVJPEG_VER_PATCH;

    std::cout << "[System] NVJpeg Version Check:" << std::endl;
    std::cout << "   - Compile-time (Headers): " << cMajor << "." << cMinor << "." << cPatch << std::endl;
    std::cout << "   - Runtime      (Library): " << rtMajor << "." << rtMinor << std::endl;

    // Allow Runtime to be NEWER than Compile-time (Forward Compatibility)
    if (rtMajor == cMajor && rtMinor >= cMinor) {
        std::cout << "   - Status: MATCH (Safe - Forward Compatible)" << std::endl;
    } else {
        std::cerr << "[WARNING] NVJpeg Version Mismatch! Runtime is older than Headers." << std::endl;
    }

    return CudaError();
}

NVJpegSink::~NVJpegSink() {
    // Safe, non-allocating, non-throwing termination
    if (!is_closed_) {
        try {
            Close();
        } catch (...) {}
    }
    // Cleanly destroy the isolated encoder state pools
    for (int b = 0; b < 2; ++b) {
        for (auto& state : staging_buffers_[b].encoder_states) {
            if (state) {
                CUDA_CALL_NO_THROW(nvjpegEncoderStateDestroy(state));
            }
        }
    }
    if (encode_params_) {
        CUDA_CALL_NO_THROW(nvjpegEncoderParamsDestroy(encode_params_));
    }
    if (nvjpeg_handle_) {
        CUDA_CALL_NO_THROW(nvjpegDestroy(nvjpeg_handle_));
    }

}

CudaError NVJpegSink::Save(BatchData &data, BatchDetections &results) {
    if (data.batchSize == 0) {
        return CudaError();
    }

    int buf_idx = active_buffer_;
    EncodeState& buf = staging_buffers_[buf_idx];

    // Join background threads for THIS specific buffer from the previous cycle
    // Prevents overwriting buf.device_decode_buffer while it's still being read
    // Check chunk statuses. Log failures, but do NOT abort pipeline
    for (auto& task : buf.async_tasks) {
        if (task.valid()) {
            try {
                std::vector<NVJpegEncodeStatus> chunk_statuses = task.get();
                for (const auto& status : chunk_statuses) {
                    if (!status.success) {
                        std::cerr << "\n[NVJpegSink] Warning: Async encode/IO failed for file:\n"
                                  << status.filename << "':\n" << status.err.Text() << std::endl;
                    }
                }
            } catch (const std::exception& e) {
                std::cerr << "\n[NVJpegSink] Warning: Async task exception: " << e.what() << std::endl;
            } catch (...) {
                std::cerr << "\n[NVJpegSink] Warning: Unknown async task exception." << std::endl;
            }
        }
    }
    buf.async_tasks.clear();

    buf.batch_size = data.batchSize;
    buf.filenames.clear();

    // Wait for Detector to finish inference
    if (data.readyEvent) {
        CUDA_TRY(cudaStreamWaitEvent(*cuda_stream_, *data.readyEvent, 0));
    }
    if (results.readyEvent) {
        CUDA_TRY(cudaStreamWaitEvent(*cuda_stream_, *results.readyEvent, 0));
    }

    int stride = BatchDetections::MAX_DETECTIONS_PER_FRAME;

    // Run Tracking (Sequential per frame in batch)
    for (int i = 0; i < data.batchSize; ++i) {
        CUDA_TRY(tracker_->ProcessBatch(
            i,
            results.data,
            results.counts,
            stride,
            (int)data.width,
            (int)data.height,
            *cuda_stream_
            ));
    }
    CUDA_TRY(tracker_->Compact(*cuda_stream_));

    // Run Annotation
    CUDA_TRY(tracker_->Annotate(
        data.deviceData,
        data.batchSize,
        data.width, data.height,
        results.data,
        results.counts,
        *cuda_stream_
        ));

    // Convert directly into the thread-safe ping-pong device buffer
    int channels = 3;
    size_t framePixels = data.width * data.height;
    size_t totalElements = framePixels * channels * data.batchSize;
    CUDA_TRY(buf.device_decode_buffer.resize(totalElements, *cuda_stream_));
    CUDA_TRY(FloatToUint8(data.deviceData.data(), buf.device_decode_buffer.data(),
                          totalElements, *cuda_stream_));

    // Mark format conversion as complete for the async threads to safely read
    CUDA_TRY(cudaEventRecord(*buf.dma_complete_event, *cuda_stream_));

    // Launch Asynchronous Encoding with Fixed Offsets
    size_t maxBytesAllocated = buf.pinned_buffer.capacity() / BatchData::MAX_BATCH_SIZE;

    // Distribute batch among the fixed Logical Encoder Pool
    int num_threads = std::min((int)data.batchSize, EncodeState::ENCODERS_PER_BUFFER);
    int frames_per_thread = (data.batchSize + num_threads - 1) / num_threads;

    for (int t = 0; t < num_threads; ++t) {
        buf.async_tasks.push_back(std::async(std::launch::async,
            [this, &buf, t, frames_per_thread, maxBytesAllocated,
             batch_size = data.batchSize, batch_id = data.batchId,
             identifiers = data.sourceIdentifiers, w = data.width, h = data.height, framePixels]() -> std::vector<NVJpegEncodeStatus> {
            std::vector<NVJpegEncodeStatus> thread_statuses;

            // Barrier: Wait for GPU conversion to finish
            cudaError_t sync_err = cudaEventSynchronize(*buf.dma_complete_event);
            if (sync_err != cudaSuccess) {
                NVJpegEncodeStatus err_status;
                err_status.err = CudaError(ERROR_SOURCE, sync_err);
                thread_statuses.push_back(err_status);
                return thread_statuses;
            }

            cudaStream_t t_stream = *buf.encode_streams[t];
            nvjpegEncoderState_t t_state = buf.encoder_states[t];

            int start_idx = t * frames_per_thread;
            int end_idx = std::min(start_idx + frames_per_thread, (int)batch_size);

            // Sequentially encode the thread's designated chunk
            for (int i = start_idx; i < end_idx; ++i) {
                // [UPDATE] Inner Lambda per-frame matching MMAPI error isolation
                NVJpegEncodeStatus frame_status = [&, frame_idx = i]() -> NVJpegEncodeStatus {
                    NVJpegEncodeStatus status;
                    std::string id = (frame_idx < identifiers.size() && !identifiers[frame_idx].empty())
                                         ? identifiers[frame_idx]
                                         : std::to_string(batch_id * batch_size + frame_idx);
                    std::stringstream ss;
                    ss << "frame_" << std::setw(4) << std::setfill('0') << id << ".jpg";
                    status.filename = fs::path(output_path_) / ss.str();

                    nvjpegImage_t img_desc;
                    uint8_t* frameStart = buf.device_decode_buffer.data() + (frame_idx * framePixels * 3);
                    img_desc.channel[0] = frameStart;
                    img_desc.channel[1] = frameStart + framePixels;
                    img_desc.channel[2] = frameStart + (2 * framePixels);
                    img_desc.pitch[0] = w;
                    img_desc.pitch[1] = w;
                    img_desc.pitch[2] = w;

                    // GPU Encode
                    CUDA_TRY_LAMBDA(nvjpegEncodeImage(nvjpeg_handle_, t_state, encode_params_,
                                                      &img_desc, NVJPEG_INPUT_RGB, w, h, t_stream), status);

                    uint8_t* targetHostPtr = buf.pinned_buffer.data() + (frame_idx * maxBytesAllocated);
                    buf.lengths[frame_idx] = maxBytesAllocated;

                    // D2H Transfer
                    CUDA_TRY_LAMBDA(nvjpegEncodeRetrieveBitstream(nvjpeg_handle_, t_state,
                                                                  targetHostPtr, &buf.lengths[frame_idx], t_stream), status);

                    // Explicit Sync: Await PCIe transfer completion before writing to disk
                    CUDA_TRY_LAMBDA(cudaStreamSynchronize(t_stream), status);

                    // SSD I/O
                    std::ofstream outFile(status.filename, std::ios::out | std::ios::binary);
                    if (outFile) {
                        outFile.write(reinterpret_cast<const char*>(targetHostPtr), buf.lengths[frame_idx]);
                        if (outFile.fail()) {
                            status.err = CudaError(ERROR_SOURCE, "Failed to write encoded JPEG to SSD.");
                            return status;
                        }
                    } else {
                        status.err = CudaError(ERROR_SOURCE, "Failed to open output file: " + status.filename);
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
    active_buffer_ = (active_buffer_ + 1) & 0x01;
    return CudaError();
}

CudaError NVJpegSink::Close() {
    if (is_closed_) return CudaError();
    is_closed_ = true;

    CudaError final_err;

    if (cuda_stream_) {
        cudaError_t err = cudaStreamSynchronize(*cuda_stream_);
        if (err != cudaSuccess) {
            final_err = CudaError(ERROR_SOURCE, std::string("Stream sync failed: ") + cudaGetErrorString(err));
        }
    }

    // Block and drain all background encoding threads before tearing down GPU states
    // Evaluate vectors of statuses and latch the first critical error encountered
    for (int b = 0; b < 2; ++b) {
        for (auto& task : staging_buffers_[b].async_tasks) {
            if (task.valid()) {
                try {
                    std::vector<NVJpegEncodeStatus> statuses = task.get();
                    for (const auto& status : statuses) {
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
                        final_err = CudaError(ERROR_SOURCE, "Unknown async encode exception");
                    }
                }
            }
        }
        staging_buffers_[b].async_tasks.clear();
    }
    return final_err;
}

} // namespace cropandweed
