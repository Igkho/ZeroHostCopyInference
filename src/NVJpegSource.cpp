#include "NVJpegSource.h"
#include "SourceKernels.h"
#include <filesystem>
#include <algorithm>
#include <fstream>
#include <iostream>

namespace fs = std::filesystem;

// Explicitly define the hardware padding requirement
// 64 bytes safely covers maximum PCIe/GPU DMA over-read transactions during bitstream parsing
constexpr size_t GPU_DMA_PADDING_BYTES = 64;

namespace cropandweed {

// Ensures Forward Compatibility with underlying GPU Driver API
static CudaError CheckNVJpegVersion() {
    int rtMajor, rtMinor;
    CUDA_TRY(nvjpegGetProperty(MAJOR_VERSION, &rtMajor));
    CUDA_TRY(nvjpegGetProperty(MINOR_VERSION, &rtMinor));
    int cMajor = NVJPEG_VER_MAJOR;
    int cMinor = NVJPEG_VER_MINOR;

    if (rtMajor != cMajor || rtMinor < cMinor) {
        std::cerr << "[WARNING] NVJpeg Version Mismatch (Source Module)! "
                  << "Headers: " << cMajor << "." << cMinor 
                  << ", Runtime: " << rtMajor << "." << rtMinor << std::endl;
    }
    return CudaError();
}

NVJpegSource::~NVJpegSource() {
    // Join outstanding futures safely to avoid zombie threads on termination
    if (cuda_stream_) {
        CUDA_CALL_NO_THROW(cudaStreamSynchronize(*cuda_stream_));
    }
    for (int b = 0; b < 2; ++b) {
        for (auto& task : futures_[b]) {
            if (task.valid()) {
                try {
                    task.get();
                } catch (...) {}
            }
        }
    }
    // Destroy Double-Buffered Arrays
    for (int b = 0; b < 2; ++b) {
        for (auto& state : decoupled_states_[b]) {
            if (state) CUDA_CALL_NO_THROW(nvjpegJpegStateDestroy(state));
        }
        for (auto& stream : jpeg_streams_[b]) {
            if (stream) CUDA_CALL_NO_THROW(nvjpegJpegStreamDestroy(stream));
        }
        for (auto& p_buf : pinned_buffers_[b]) {
            if (p_buf) CUDA_CALL_NO_THROW(nvjpegBufferPinnedDestroy(p_buf));
        }
        for (auto& d_buf : device_buffers_[b]) {
            if (d_buf) CUDA_CALL_NO_THROW(nvjpegBufferDeviceDestroy(d_buf));
        }
    }

    // Destroy Shared Decoupled Components
    if (decode_params_) {
        CUDA_CALL_NO_THROW(nvjpegDecodeParamsDestroy(decode_params_));
    }
    if (jpeg_decoder_) {
        CUDA_CALL_NO_THROW(nvjpegDecoderDestroy(jpeg_decoder_));
    }

    // Destroy Base Handle Last
    if (nvjpeg_handle_) {
        CUDA_CALL_NO_THROW(nvjpegDestroy(nvjpeg_handle_));
    }
}

CudaError NVJpegSource::Create(std::unique_ptr<ISource>& out, std::string folderPath,
                               int width, int height, size_t batch_size) {
    auto ptr = std::make_unique<NVJpegSource>(Token{}, std::move(folderPath),
                                              width, height, batch_size);
    CUDA_TRY(ptr->Init());
    out = std::move(ptr);
    return CudaError();
}

CudaError NVJpegSource::Init() {
    if (!fs::exists(folder_path_) || !fs::is_directory(folder_path_)) {
        return CudaError(ERROR_SOURCE, "Invalid folder path: " + folder_path_);
    }

    // 1. Gather all target files
    for (const auto& entry : fs::directory_iterator(folder_path_)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".jpg" || ext == ".jpeg") {
                file_list_.push_back(entry.path().string());
            }
        }
    }
    
    // Sort to maintain deterministic processing 
    std::sort(file_list_.begin(), file_list_.end());
    
    if (file_list_.empty()) {
        std::cerr << "[NVJpegSource] Warning: No JPEG files found in " << folder_path_ << std::endl;
    }

    // 2. Initialize Hardware / Software Resources
    CUDA_TRY(CudaStream::Create(cuda_stream_, cudaStreamNonBlocking));
    CUDA_TRY(CheckNVJpegVersion());
    CUDA_TRY(nvjpegCreateSimple(&nvjpeg_handle_));
    CUDA_TRY(nvjpegDecoderCreate(nvjpeg_handle_, NVJPEG_BACKEND_DEFAULT, &jpeg_decoder_));
    CUDA_TRY(nvjpegDecodeParamsCreate(nvjpeg_handle_, &decode_params_));
    CUDA_TRY(nvjpegDecodeParamsSetOutputFormat(decode_params_, NVJPEG_OUTPUT_RGB));

    size_t max_batch = BatchData::MAX_BATCH_SIZE;

    // Initialize logical decoders based on limit macro
    for (int b = 0; b < 2; ++b) {
        resource_pool_[b].resize(batch_size_);
        futures_[b].resize(DECODERS_PER_BUFFER);

        decoupled_states_[b].resize(DECODERS_PER_BUFFER);
        jpeg_streams_[b].resize(DECODERS_PER_BUFFER);
        pinned_buffers_[b].resize(DECODERS_PER_BUFFER);
        device_buffers_[b].resize(DECODERS_PER_BUFFER);
        decode_streams_[b].resize(DECODERS_PER_BUFFER);

        for (int t = 0; t < DECODERS_PER_BUFFER; ++t) {
            CUDA_TRY(CudaStream::Create(decode_streams_[b][t], cudaStreamNonBlocking));
            CUDA_TRY(nvjpegDecoderStateCreate(nvjpeg_handle_, jpeg_decoder_, &decoupled_states_[b][t]));
            CUDA_TRY(nvjpegJpegStreamCreate(nvjpeg_handle_, &jpeg_streams_[b][t]));
            CUDA_TRY(nvjpegBufferPinnedCreate(nvjpeg_handle_, nullptr, &pinned_buffers_[b][t]));
            CUDA_TRY(nvjpegBufferDeviceCreate(nvjpeg_handle_, nullptr, &device_buffers_[b][t]));
        }

        CUDA_TRY(CudaEvent::Create(dma_complete_event_[b], cudaEventDisableTiming));
        CUDA_TRY(cudaEventRecord(*dma_complete_event_[b], *cuda_stream_));

        // Fire the background pre-fetching automatically
        DispatchAsyncBatch(b);
    }

    return CudaError();
}

CudaError NVJpegSource::GetNextBatch(BatchData& outBatch, size_t batchSize, bool& process) {
    int buf_idx = active_buffer_;

    // Save the frame count before dispatching the next batch
    int current_chunk_frames = frames_in_buffer_[buf_idx];

    // 1. Check for End of Stream
    if (current_chunk_frames == 0) {
        process = false;
        return CudaError();
    }

    if (batchSize > batch_size_) {
        return CudaError(ERROR_SOURCE, "Requested batchSize exceeds pre-allocated pool");
    }

    // 2. Gather Async Threads together
    std::vector<NVJpegDecodeStatus> valid_statuses;
    int num_threads = std::min(current_chunk_frames, DECODERS_PER_BUFFER);

    for (int t = 0; t < num_threads; ++t) {
        std::vector<NVJpegDecodeStatus> chunk_statuses = futures_[buf_idx][t].get();
        for (const auto& status : chunk_statuses) {
            if (status.success) {
                valid_statuses.push_back(status);
            } else {
                std::cerr << "\n[NVJpegSource] Warning: Skipping bad file '"
                          << status.filename << "': " << status.err.Text() << std::endl;
            }
        }
    }

    int validCount = valid_statuses.size();
    if (validCount == 0) {
        // Free buffer, switch, and recursively fetch the next chunk
        CUDA_TRY(cudaEventRecord(*dma_complete_event_[buf_idx], *cuda_stream_));
        active_buffer_ = (active_buffer_ + 1) & 0x01;
        frameCounter_ += current_chunk_frames;
        DispatchAsyncBatch(buf_idx);
        std::cerr << "\n[NVJpegSource] Warning: Skipping fully bad batch." << std::endl;
        return GetNextBatch(outBatch, batchSize, process);
    }

    // 3. Setup GPU Output Layout
    size_t framePixelsTarget = targetW_ * targetH_;
    CUDA_TRY(outBatch.deviceData.resize(batchSize * framePixelsTarget * 3, *cuda_stream_));
    outBatch.sourceIdentifiers.clear();

    // 4. Main-Thread Operations (Color Conversion)
    for (int k = 0; k < validCount; ++k) {
        const NVJpegDecodeStatus& status = valid_statuses[k];
        int i = status.batch_index;
        auto& res = resource_pool_[buf_idx][i];

        float* batch_dst = outBatch.deviceData.data() + (k * framePixelsTarget * 3);

        // Convert the decoded RGB planar bytes to normalized floats
        CUDA_TRY(ResizeAndCastRGBPlanar(
            res.decoded_pixels.data(),
            res.decoded_pixels.data() + status.width * status.height,
            res.decoded_pixels.data() + 2 * status.width * status.height,
            status.width, status.height, status.width,
            batch_dst, targetW_, targetH_, *cuda_stream_));

        outBatch.sourceIdentifiers.push_back(std::to_string(frameCounter_ + i));
    }

    // 5. Zero-fill padding (if batch is incomplete)
    if (validCount < batchSize) {
        size_t offset = validCount * framePixelsTarget * 3;
        CUDA_TRY(outBatch.deviceData.fill_back(offset, 0.0f, *cuda_stream_));
    }

    // 6. Write the conversion finish event (Releases memory for async threads on next cycle)
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

    return CudaError();
}

// True multi-threaded pre-fetching and hardware scheduling
void NVJpegSource::DispatchAsyncBatch(int buf_idx) {
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

    int num_threads = std::min(frames_in_buffer_[buf_idx], DECODERS_PER_BUFFER);
    int frames_per_thread = (frames_in_buffer_[buf_idx] + num_threads - 1) / num_threads; // Ceiling division

    // Fire Async Threads
    for (int t = 0; t < num_threads; ++t) {
        futures_[buf_idx][t] = std::async(std::launch::async,
            [this, t, frames_per_thread, frames_in_buffer = frames_in_buffer_[buf_idx],
             batch_filenames, buf_idx]() -> std::vector<NVJpegDecodeStatus> {

            std::vector<NVJpegDecodeStatus> thread_statuses;
            int start_idx = t * frames_per_thread;
            int end_idx = std::min(start_idx + frames_per_thread, frames_in_buffer);

            cudaStream_t local_stream = *decode_streams_[buf_idx][t];

            // This single thread sequentially reads and decodes its chunk of files
            for (int i = start_idx; i < end_idx; ++i) {

                NVJpegDecodeStatus frame_status = [&, frame_idx = i]() -> NVJpegDecodeStatus {
                    NVJpegDecodeStatus status;
                    status.filename = batch_filenames[frame_idx];
                    status.batch_index = frame_idx;

                    // Fetch resource targeted for this exact frame (Data Safety)
                    auto& res = resource_pool_[buf_idx][frame_idx];

                    // Fetch decoder structures constrained by thread limit 't' (Hardware Limits)
                    auto& state = decoupled_states_[buf_idx][t];
                    auto& jpeg_stream = jpeg_streams_[buf_idx][t];
                    auto& pinned_buf = pinned_buffers_[buf_idx][t];
                    auto& device_buf = device_buffers_[buf_idx][t];

                    // Read File to RAM
                    std::ifstream file(status.filename, std::ios::in | std::ios::binary | std::ios::ate);
                    if (!file) {
                        status.err = CudaError(ERROR_SOURCE, "Cannot read file.");
                        return status;
                    }

                    size_t file_size = file.tellg();
                    file.seekg(0, std::ios::beg);

                    std::vector<uint8_t> local_raw_data(file_size + GPU_DMA_PADDING_BYTES, 0);
                    if (!file.read(reinterpret_cast<char*>(local_raw_data.data()), file_size)) {
                        status.err = CudaError(ERROR_SOURCE, "File read failure.");
                        return status;
                    }

                    // Magic byte check (Protects NVJPEG engine from faulting on standard text/invalid files)
                    if (file_size < 2 || local_raw_data[0] != 0xFF || local_raw_data[1] != 0xD8) {
                        status.err = CudaError(ERROR_SOURCE, "Invalid JPEG magic bytes.");
                        return status;
                    }

                    // CPU Phase
                    CUDA_TRY_LAMBDA(nvjpegStateAttachDeviceBuffer(state, device_buf), status);
                    CUDA_TRY_LAMBDA(nvjpegStateAttachPinnedBuffer(state, pinned_buf), status);
                    CUDA_TRY_LAMBDA(nvjpegJpegStreamParse(nvjpeg_handle_, local_raw_data.data(), file_size, 0, 0, jpeg_stream), status);

                    int channels, widths[NVJPEG_MAX_COMPONENT], heights[NVJPEG_MAX_COMPONENT];
                    nvjpegChromaSubsampling_t subsampling;
                    CUDA_TRY_LAMBDA(nvjpegGetImageInfo(nvjpeg_handle_, local_raw_data.data(), file_size, &channels, &subsampling, widths, heights), status);

                    status.width = widths[0];
                    status.height = heights[0];
                    status.channels = channels;

                    CUDA_TRY_LAMBDA(nvjpegDecodeJpegHost(nvjpeg_handle_, jpeg_decoder_, state, decode_params_, jpeg_stream), status);

                    // GPU Phase
                    // Wait for main thread to finish processing this memory block from the previous cycle
                    CUDA_TRY_LAMBDA(cudaEventSynchronize(*dma_complete_event_[buf_idx]), status);

                    size_t plane_size = status.width * status.height;
                    CUDA_TRY_LAMBDA(res.decoded_pixels.resize(plane_size * status.channels, local_stream), status);

                    nvjpegImage_t destImage;
                    destImage.channel[0] = res.decoded_pixels.data();
                    destImage.channel[1] = res.decoded_pixels.data() + plane_size;
                    destImage.channel[2] = res.decoded_pixels.data() + 2 * plane_size;
                    destImage.pitch[0] = status.width;
                    destImage.pitch[1] = status.width;
                    destImage.pitch[2] = status.width;

                    CUDA_TRY_LAMBDA(nvjpegDecodeJpegTransferToDevice(nvjpeg_handle_, jpeg_decoder_, state, jpeg_stream, local_stream), status);
                    CUDA_TRY_LAMBDA(nvjpegDecodeJpegDevice(nvjpeg_handle_, jpeg_decoder_, state, &destImage, local_stream), status);

                    // Thread Synchronization
                    // Ensure the transfer is finished before releasing 'local_raw_data' and yielding to the main thread
                    CUDA_TRY_LAMBDA(cudaStreamSynchronize(local_stream), status);

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
