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

    encoder_states_.resize(BatchData::MAX_BATCH_SIZE);
    for (auto& state : encoder_states_) {
        CUDA_TRY(nvjpegEncoderStateCreate(nvjpeg_handle_, &state, *cuda_stream_));
    }

    CUDA_TRY(nvjpegEncoderParamsCreate(nvjpeg_handle_, &encode_params_, *cuda_stream_));
    CUDA_TRY(nvjpegEncoderParamsSetSamplingFactors(encode_params_, NVJPEG_CSS_444, *cuda_stream_));
    CUDA_TRY(nvjpegEncoderParamsSetQuality(encode_params_, 90, *cuda_stream_));
    CUDA_TRY(CheckNVJpegVersion());
    size_t max_jpeg_reservation = BatchData::MAX_BATCH_SIZE * 10 * 1024 * 1024; // 10MB per image max
    for (int i = 0; i < 2; ++i) {
        CUDA_TRY(staging_buffers_[i].pinned_buffer.reserve(max_jpeg_reservation, *cuda_stream_));
        staging_buffers_[i].lengths.resize(BatchData::MAX_BATCH_SIZE);
        CUDA_TRY(CudaEvent::Create(staging_buffers_[i].dma_complete_event, cudaEventDisableTiming));
        CUDA_TRY(cudaEventRecord(*staging_buffers_[i].dma_complete_event, *cuda_stream_));
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

// The DRY Helper Method
CudaError NVJpegSink::FlushBufferToDisk(EncodeState& buf) {
    if (!buf.has_data) {
        return CudaError();
    }

    // 1. Sync: Wait for GPU to finish DMA writing to this buffer
    CUDA_TRY(cudaEventSynchronize(*buf.dma_complete_event));

    // 2. Dispatch Parallel SSD Writes
    std::vector<std::future<CudaError>> write_futures;
    size_t maxBytesPerImage = buf.pinned_buffer.capacity() / buf.batch_size;

    for (int i = 0; i < buf.batch_size; ++i) {
        write_futures.push_back(std::async(std::launch::async, [this, &buf, i, maxBytesPerImage]() -> CudaError {
            fs::path filePath = fs::path(output_path_) / buf.filenames[i];
            std::ofstream outFile(filePath, std::ios::out | std::ios::binary);

            if (outFile) {
                outFile.write(reinterpret_cast<const char*>(buf.pinned_buffer.data() + (i * maxBytesPerImage)), buf.lengths[i]);
                return CudaError();
            } else {
                return CudaError(ERROR_SOURCE, "Failed to write file: " + filePath.string());
            }
        }));
    }

    // 3. Barrier: Wait for all SSD write threads to complete and capture errors
    CudaError pipeline_err;
    for (auto& f : write_futures) {
        CudaError thread_err = f.get();
        // Capture the first error encountered, if any
        if (CudaError::IsFailure(thread_err) && !CudaError::IsFailure(pipeline_err)) {
            pipeline_err = thread_err;
        }
    }

    // Mark as safely flushed regardless of write success to prevent infinite retry loops
    buf.has_data = false;

    return pipeline_err;
}

NVJpegSink::~NVJpegSink() {
    // 1. Safe, non-allocating, non-throwing termination
    try {
        for (int b = 0; b < 2; ++b) {
            auto &buf = staging_buffers_[b];
            if (buf.has_data) {

                CUDA_CALL_NO_THROW(cudaEventSynchronize(*buf.dma_complete_event));
                size_t maxBytesPerImage = buf.pinned_buffer.capacity() / buf.batch_size;

                // Zero dynamic heap allocations. Strict synchronous I/O.
                for (int i = 0; i < buf.batch_size; ++i) {
                    fs::path filePath = fs::path(output_path_) / buf.filenames[i];
                    std::ofstream outFile(filePath, std::ios::out | std::ios::binary);
                    if (outFile) {
                        outFile.write(reinterpret_cast<const char*>(buf.pinned_buffer.data() +
                                                                     (i * maxBytesPerImage)), buf.lengths[i]);
                    }
                }
                buf.has_data = false;
            }
        }
    } catch (...) {
        // Ultimate safety net: Swallow any STL exceptions to prevent std::terminate
    }
    // Standard Cleanup
    for (auto& state : encoder_states_) {
        if (state) {
            CUDA_CALL_NO_THROW(nvjpegEncoderStateDestroy(state));
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

    int prev_buffer_idx = (active_buffer_ + 1) & 0x01;
    EncodeState& buf = staging_buffers_[active_buffer_];
    EncodeState& prev_buf = staging_buffers_[prev_buffer_idx];

    // Setup CURRENT batch metadata
    buf.batch_size = data.batchSize;
    buf.filenames.clear();
    buf.has_data = true;

    // 3. Wait for Detector to finish inference
    if (data.readyEvent) {
        CUDA_TRY(cudaStreamWaitEvent(*cuda_stream_, *data.readyEvent, 0));
    }
    if (results.readyEvent) {
        CUDA_TRY(cudaStreamWaitEvent(*cuda_stream_, *results.readyEvent, 0));
    }

    int stride = BatchDetections::MAX_DETECTIONS_PER_FRAME;

    // 4. Run Tracking (Sequential per frame in batch)
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

    // 5. Run Annotation
    CUDA_TRY(tracker_->Annotate(
        data.deviceData,
        data.batchSize,
        data.width, data.height,
        results.data,
        results.counts,
        *cuda_stream_
        ));

    // 6. Format Conversion (Float -> Uint8)
    int channels = 3;
    size_t framePixels = data.width * data.height;
    size_t totalElements = framePixels * channels * data.batchSize;
    CUDA_TRY(device_decode_buffer_.resize(totalElements, *cuda_stream_));

    // Convert Float->Uint8 (Kernel)
    CUDA_TRY(FloatToUint8(data.deviceData.data(), device_decode_buffer_.data(),
                          totalElements, *cuda_stream_));

    // 7. Launch Asynchronous Encoding with Fixed Offsets
    size_t maxBytesAllocated = buf.pinned_buffer.capacity() / data.batchSize;

    for (int i = 0; i < data.batchSize; ++i) {
        std::string id = (i < data.sourceIdentifiers.size() && !data.sourceIdentifiers[i].empty())
                             ? data.sourceIdentifiers[i]
                             : std::to_string(data.batchId * data.batchSize + i);

        std::stringstream ss;
        ss << "frame_" << std::setw(4) << std::setfill('0') << id << ".jpg";
        buf.filenames.push_back(ss.str());

        nvjpegImage_t img_desc;
        uint8_t* frameStart = device_decode_buffer_.data() + (i * framePixels * 3);
        img_desc.channel[0] = frameStart;
        img_desc.channel[1] = frameStart + framePixels;
        img_desc.channel[2] = frameStart + (2 * framePixels);
        img_desc.pitch[0] = data.width;
        img_desc.pitch[1] = data.width;
        img_desc.pitch[2] = data.width;

        CUDA_TRY(nvjpegEncodeImage(nvjpeg_handle_, encoder_states_[i], encode_params_,
                                   &img_desc, NVJPEG_INPUT_RGB, data.width, data.height, *cuda_stream_));

        uint8_t* targetHostPtr = buf.pinned_buffer.data() + (i * maxBytesAllocated);

        // Tell nvJPEG the physical capacity of the buffer so it doesn't fail the bounds check
        buf.lengths[i] = maxBytesAllocated;

        CUDA_TRY(nvjpegEncodeRetrieveBitstream(nvjpeg_handle_, encoder_states_[i],
                                               targetHostPtr, &buf.lengths[i], *cuda_stream_));
    }

    // 8. Protect Pinned Buffer, Flush Previous & Swap
    CUDA_TRY(cudaEventRecord(*buf.dma_complete_event, *cuda_stream_));
    CUDA_TRY(FlushBufferToDisk(prev_buf));
    active_buffer_ = prev_buffer_idx;

    return CudaError();
}

CudaError NVJpegSink::Close() {
    // Flush any remaining data in the buffers using the fast parallel async path
    for (int b = 0; b < 2; ++b) {
        if (staging_buffers_[b].has_data) {
            CUDA_TRY(FlushBufferToDisk(staging_buffers_[b]));
        }
    }
    return CudaError();
}

} // namespace cropandweed
