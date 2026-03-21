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
    // 1. Destroy Double-Buffered Arrays
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

    // 2. Destroy Shared Decoupled Components
    if (decode_params_) {
        CUDA_CALL_NO_THROW(nvjpegDecodeParamsDestroy(decode_params_));
    }
    if (jpeg_decoder_) {
        CUDA_CALL_NO_THROW(nvjpegDecoderDestroy(jpeg_decoder_));
    }

    // 3. Destroy Base Handle Last
    if (nvjpeg_handle_) {
        CUDA_CALL_NO_THROW(nvjpegDestroy(nvjpeg_handle_));
    }
}

CudaError NVJpegSource::Create(std::unique_ptr<ISource>& out, std::string folderPath, int width, int height) {
    auto ptr = std::make_unique<NVJpegSource>(Token{}, std::move(folderPath), width, height);
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

    // Initialize exactly two sets of GPU-boundary buffers
    for (int b = 0; b < 2; ++b) {
        decoupled_states_[b].resize(max_batch);
        jpeg_streams_[b].resize(max_batch);
        pinned_buffers_[b].resize(max_batch);
        device_buffers_[b].resize(max_batch);

        for (size_t i = 0; i < max_batch; ++i) {
            CUDA_TRY(nvjpegDecoderStateCreate(nvjpeg_handle_, jpeg_decoder_, &decoupled_states_[b][i]));
            CUDA_TRY(nvjpegJpegStreamCreate(nvjpeg_handle_, &jpeg_streams_[b][i]));
            CUDA_TRY(nvjpegBufferPinnedCreate(nvjpeg_handle_, nullptr, &pinned_buffers_[b][i]));
            CUDA_TRY(nvjpegBufferDeviceCreate(nvjpeg_handle_, nullptr, &device_buffers_[b][i]));
        }

        CUDA_TRY(CudaEvent::Create(dma_complete_event_[b], cudaEventDisableTiming));
        CUDA_TRY(cudaEventRecord(*dma_complete_event_[b], *cuda_stream_));
    }

    return CudaError();
}

CudaError NVJpegSource::GetNextBatch(BatchData& outBatch, size_t batchSize, bool& process) {
    if (current_file_idx_ >= file_list_.size()) {
        process = false;
        return CudaError();
    }

    int buf_idx = active_buffer_;

    // 1. Sync: Wait for GPU to finish pulling from this specific buffer
    CUDA_TRY(cudaEventSynchronize(*dma_complete_event_[buf_idx]));

    std::vector<std::string> batch_filenames;
    int framesCollected = 0;
        while (framesCollected < batchSize && current_file_idx_ < file_list_.size()) {
        batch_filenames.push_back(file_list_[current_file_idx_]);
        current_file_idx_++;
        framesCollected++;
    }

    if (framesCollected == 0) {
        process = false;
        return CudaError();
    }

    // 2. Dispatch Parallel Tasks with Local, Ephemeral Memory
    std::vector<std::future<DecodeResult>> futures;
    for (int i = 0; i < framesCollected; ++i) {
        futures.push_back(std::async(std::launch::async,
                                     [this, i, buf_idx, filepath = batch_filenames[i]]() -> DecodeResult {
            DecodeResult res;

            std::ifstream file(filepath, std::ios::in | std::ios::binary | std::ios::ate);
            if (!file) {
                res.err = CudaError(ERROR_SOURCE, "Can't open the file: " + filepath);
                return res;
            }

            size_t file_size = file.tellg();
            file.seekg(0, std::ios::beg);

            std::vector<uint8_t> local_raw_data(file_size);
            if (!file.read(reinterpret_cast<char*>(local_raw_data.data()), file_size)) {
                res.err = CudaError(ERROR_SOURCE, "Can't read the file: " + filepath);
                return res;
            }

            // Bind decoupled buffers to state
            CUDA_TRY_LAMBDA(nvjpegStateAttachDeviceBuffer(decoupled_states_[buf_idx][i],
                                                          device_buffers_[buf_idx][i]),
                                                          res);
            CUDA_TRY_LAMBDA(nvjpegStateAttachPinnedBuffer(decoupled_states_[buf_idx][i],
                                                          pinned_buffers_[buf_idx][i]), res);

            // Parse Stream from our local vector
            CUDA_TRY_LAMBDA(nvjpegJpegStreamParse(nvjpeg_handle_,
                                                  local_raw_data.data(),
                                                  file_size, 0, 0,
                                                  jpeg_streams_[buf_idx][i]), res);

            int channels, widths[NVJPEG_MAX_COMPONENT], heights[NVJPEG_MAX_COMPONENT];
            nvjpegChromaSubsampling_t subsampling;

            CUDA_TRY_LAMBDA(nvjpegGetImageInfo(nvjpeg_handle_, local_raw_data.data(), file_size,
                                               &channels, &subsampling, widths, heights), res);
            if (subsampling == NVJPEG_CSS_UNKNOWN) {
                res.err = CudaError(ERROR_SOURCE, "File with unknown chroma subsampling: " + filepath);
                return res;
            }

            res.width = widths[0];
            res.height = heights[0];
            res.channels = channels;

            // Execute heavy CPU Phase.
            // This consumes `local_raw_data` and writes safe DMA-ready data to `pinned_buffers_[buf_idx][i]`
            CUDA_TRY_LAMBDA(nvjpegDecodeJpegHost(nvjpeg_handle_, jpeg_decoder_,
                                                 decoupled_states_[buf_idx][i], decode_params_,
                                                 jpeg_streams_[buf_idx][i]), res);

            res.success = true;
            return res;
        }));
    }

    // 3. Barrier: Wait for all CPU threads to finish and compute offsets
    std::vector<int> original_widths;
    std::vector<int> original_heights;
    std::vector<size_t> raw_offsets;
    std::vector<int> valid_indices;
    size_t total_raw_bytes_needed = 0;

    for (int i = 0; i < framesCollected; ++i) {
        DecodeResult res = futures[i].get();
        if (!res.success) {
            // Log the error locally and skip the file
            std::cerr << "\n[NVJpegSource] Warning: Skipping bad file '" << batch_filenames[i]
                      << "' due to decoding error:\n" << res.err.Text() << "\n" << std::endl;
            continue;
        }
        original_widths.push_back(res.width);
        original_heights.push_back(res.height);
        raw_offsets.push_back(total_raw_bytes_needed);
        valid_indices.push_back(i);
        total_raw_bytes_needed += res.width * res.height * res.channels;
    }

    int validCount = valid_indices.size();

    // If all frames in this chunk failed, safely release the buffer and fetch the next chunk recursively
    if (validCount == 0) {
        CUDA_TRY(cudaEventRecord(*dma_complete_event_[buf_idx], *cuda_stream_));
        active_buffer_ = (active_buffer_ + 1) & 0x01;
        frameCounter_ += framesCollected;
        return GetNextBatch(outBatch, batchSize, process);
    }

    // 4. Setup GPU Output Layout
    size_t framePixelsTarget = targetW_ * targetH_;

    // Always allocate for the FULL requested batchSize to satisfy strict-batch engines
    CUDA_TRY(outBatch.deviceData.resize(batchSize * framePixelsTarget * 3, *cuda_stream_));
    CUDA_TRY(device_decode_buffer_.resize(total_raw_bytes_needed, *cuda_stream_));
    outBatch.sourceIdentifiers.clear();

    // 5. Queue Asynchronous GPU Transfers (DMA reads from pinned_buffers_)
    for (int k = 0; k < validCount; ++k) {
        int i = valid_indices[k]; // Lookup the original index 'i' for the hardware bindings
        int srcW = original_widths[k];
        int srcH = original_heights[k];
        size_t srcPlaneSize = srcW * srcH;

        uint8_t* frameStart = device_decode_buffer_.data() + raw_offsets[k];
        nvjpegImage_t destImage;
        destImage.channel[0] = frameStart;
        destImage.channel[1] = frameStart + srcPlaneSize;
        destImage.channel[2] = frameStart + (2 * srcPlaneSize);
        destImage.pitch[0] = srcW;
        destImage.pitch[1] = srcW;
        destImage.pitch[2] = srcW;

        CUDA_TRY(nvjpegDecodeJpegTransferToDevice(nvjpeg_handle_, jpeg_decoder_,
                                                  decoupled_states_[buf_idx][i],
                                                  jpeg_streams_[buf_idx][i], *cuda_stream_));

        CUDA_TRY(nvjpegDecodeJpegDevice(nvjpeg_handle_, jpeg_decoder_,
                                        decoupled_states_[buf_idx][i], &destImage, *cuda_stream_));

        float* batch_dst = outBatch.deviceData.data() + (k * framePixelsTarget * 3);

        CUDA_TRY(ResizeAndCastRGBPlanar(destImage.channel[0], destImage.channel[1], destImage.channel[2],
            srcW, srcH, srcW, batch_dst, targetW_, targetH_, *cuda_stream_));

        outBatch.sourceIdentifiers.push_back(std::to_string(frameCounter_ + i));
    }

    // Zero-fill the padding frames so the CNN doesn't process garbage/NaNs
    if (validCount < batchSize) {
        size_t offset = validCount * framePixelsTarget * 3;
        CUDA_TRY(outBatch.deviceData.fill_back(offset, 0.0f, *cuda_stream_));
    }

    // 6. Protect Pinned Buffers
    // CPU cannot overwrite this set of pinned_buffers_ until this event triggers next cycle
    CUDA_TRY(cudaEventRecord(*dma_complete_event_[buf_idx], *cuda_stream_));

    // 7. Signal Pipeline Readiness
    if (!outBatch.readyEvent) {
        CUDA_TRY(CudaEvent::Create(outBatch.readyEvent));
    }
    CUDA_TRY(cudaEventRecord(*outBatch.readyEvent, *cuda_stream_));

    outBatch.batchId = frameCounter_ / batchSize;
    frameCounter_ += framesCollected;
    outBatch.batchSize = validCount; // Assign the compressed, valid size to the downstream detector
    outBatch.width = targetW_;
    outBatch.height = targetH_;
    process = true;

    // 8. Swap active buffer index for the next iteration
    active_buffer_ = (active_buffer_ + 1) & 0x01;

    return CudaError();
}

} // namespace cropandweed
