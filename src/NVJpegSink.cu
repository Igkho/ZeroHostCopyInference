#include "NVJpegSink.h"
#include "NVJpegSinkKernels.h"
#include "helpers.h"
#include <numeric>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <sstream>
#include <iomanip>
#include "DetectionRaw.h"

namespace fs = std::filesystem;

namespace cropandweed {

CudaError NVJpegSink::Init() {
    if (!fs::exists(output_path_)) {
        fs::create_directories(output_path_);
    }

    // Init Stream using your Factory Pattern
    CUDA_TRY(CudaStream::Create(cuda_stream_, cudaStreamNonBlocking));

    // Initialize Tracker
    CUDA_TRY(ObjectTracker::Create(tracker_, modelProps_.numClasses, *cuda_stream_));

    // Init nvJPEG
    CUDA_TRY(nvjpegCreateSimple(&nvjpeg_handle_));

    encoder_states_.resize(BatchData::MAX_BATCH_SIZE);
    for (auto& state : encoder_states_) {
        CUDA_TRY(nvjpegEncoderStateCreate(nvjpeg_handle_, &state, *cuda_stream_));
    }

    CUDA_TRY(nvjpegEncoderParamsCreate(nvjpeg_handle_, &encode_params_, *cuda_stream_));
    CUDA_TRY(nvjpegEncoderParamsSetSamplingFactors(encode_params_, NVJPEG_CSS_444, *cuda_stream_));
    CUDA_TRY(nvjpegEncoderParamsSetQuality(encode_params_, 90, *cuda_stream_));
    CUDA_TRY(CheckNVJpegVersion());
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
    if (encode_params_) {
        CUDA_CALL_NO_THROW(nvjpegEncoderParamsDestroy(encode_params_));
    }
    for (auto& state : encoder_states_) {
        if (state) {
            CUDA_CALL_NO_THROW(nvjpegEncoderStateDestroy(state));
        }
    }
    if (nvjpeg_handle_) {
        CUDA_CALL_NO_THROW(nvjpegDestroy(nvjpeg_handle_));
    }
}

CudaError NVJpegSink::Save(BatchData &data, BatchDetections &results) {
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

    int channels = 3;
    size_t framePixels = data.width * data.height;
    size_t totalElements = framePixels * channels * data.batchSize;

    // Resize buffer (Automatic growth, no-op if large enough)
    CUDA_TRY(buffer_block_.resize(totalElements, *cuda_stream_));

    // Convert Float->Uint8 (Kernel)
    CUDA_TRY(FloatToUint8(data.deviceData.data(), buffer_block_.data(),
                          totalElements, *cuda_stream_));

    // Encode
    for (int i = 0; i < data.batchSize; ++i) {
        size_t frameSizeBytes = framePixels * channels;
        uint8_t* frameStart = buffer_block_.data() + (i * frameSizeBytes);

        nvjpegImage_t imgDesc = {0};
        imgDesc.channel[0] = frameStart;
        imgDesc.channel[1] = frameStart + framePixels;
        imgDesc.channel[2] = frameStart + (2 * framePixels);
        imgDesc.pitch[0] = (unsigned int)data.width;
        imgDesc.pitch[1] = (unsigned int)data.width;
        imgDesc.pitch[2] = (unsigned int)data.width;

        CUDA_TRY(nvjpegEncodeImage(nvjpeg_handle_, encoder_states_[i], encode_params_,
                                   &imgDesc, NVJPEG_INPUT_RGB,
                                   data.width, data.height, *cuda_stream_));
    }

    // Retrieve Bitstream Sizes
    std::vector<size_t> lengths(data.batchSize, 0);
    for (int i = 0; i < data.batchSize; ++i) {
        CUDA_TRY(nvjpegEncodeRetrieveBitstream(nvjpeg_handle_, encoder_states_[i],
                                               nullptr, &lengths[i], *cuda_stream_));
    }

    CUDA_TRY(cudaStreamSynchronize(*cuda_stream_));
    size_t totalBytes = std::accumulate(lengths.begin(), lengths.end(), 0);

    if (totalBytes > 0) {
        CUDA_TRY(pinned_buffer_.resize(totalBytes, *cuda_stream_));
        uint8_t* currentHostPtr = pinned_buffer_.data();
        std::vector<uint8_t*> framePointers;

        for (int i = 0; i < data.batchSize; ++i) {
            framePointers.push_back(currentHostPtr);
            CUDA_TRY(nvjpegEncodeRetrieveBitstream(nvjpeg_handle_, encoder_states_[i],
                                                   currentHostPtr, &lengths[i], *cuda_stream_));
            currentHostPtr += lengths[i];
        }

        CUDA_TRY(cudaStreamSynchronize(*cuda_stream_));

        // File writing (Standard C++ I/O)
        for (int i = 0; i < data.batchSize; ++i) {
            std::string id;
            if (i < data.sourceIdentifiers.size() && !data.sourceIdentifiers[i].empty()) {
                id = data.sourceIdentifiers[i];
            } else {
                // Global Frame ID calculation: BatchID * BatchSize + Offset
                id = std::to_string(data.batchId * data.batchSize + i);
            }

            std::stringstream ss;
            ss << "frame_" << std::setw(4) << std::setfill('0') << id << ".jpg";
            std::string filename = ss.str();

            fs::path filePath = fs::path(output_path_) / filename;
            std::ofstream outFile(filePath, std::ios::out | std::ios::binary);
            if (outFile) {
                outFile.write(reinterpret_cast<const char*>(framePointers[i]), lengths[i]);
            }
        }
    }
    return CudaError();
}

} // namespace cropandweed
