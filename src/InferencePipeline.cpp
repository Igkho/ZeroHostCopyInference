#include "InferencePipeline.h"

namespace cropandweed {

InferencePipeline::InferencePipeline(std::unique_ptr<ISource> src,
                                     std::unique_ptr<IDetector> det,
                                     std::unique_ptr<ISink> sink,
                                     int batchSize)
    : source_(std::move(src)),
      detector_(std::move(det)),
      sink_(std::move(sink)),
      batchSize_(batchSize) {
}

InferencePipeline::~InferencePipeline() {
    // Ensure threads are joined if the object is destroyed before Run() finishes normally
    running_ = false;
    for (auto& t : workers_) {
        if (t.joinable()) {
            t.join();
        }
    }
}

void InferencePipeline::PrintStats(long long initTimeMs) {
    stats_.PrintReport(initTimeMs);
}

CudaError InferencePipeline::Run() {
    // Track Wall Clock Time
    PerformanceTimer wallTimer("Pipeline Run");

    running_ = true;

    // Start Inference Worker Thread
    workers_.emplace_back(&InferencePipeline::inferenceWorker, this);

    // Start Output Worker Thread
    workers_.emplace_back(&InferencePipeline::outputWorker, this);

    // Main Loop: Producer (Read Source)
    // Reuse 'batch' memory by popping from the recycle queue
    bool process = true;

    Stopwatch sw; // Reusable timer

    while (process && running_) {
        BatchData batch;

        // Try to recycle an old batch buffer to save allocation
        // If recycle queue is empty, 'batch' is default constructed (empty Block)
        // and GetNextBatch will perform the initial allocation.
        if (recycleQueue_.TryPop(batch, std::chrono::milliseconds(0))) {
            // DEBUG: Confirm recycling is working
//            std::cout << "[Pipeline] Recycling BatchData buffer (Capacity: "
//                      << batch.deviceData.capacity() << " bytes)" << std::endl;
        }

        sw.Start();
        CUDA_TRY(source_->GetNextBatch(batch, batchSize_, process));
        double elapsed = sw.Stop();

        if (process) {
            // Accumulate Stats
            stats_.totalDecodingMs = stats_.totalDecodingMs + elapsed;
            stats_.totalBatches++;
            stats_.totalFrames = stats_.totalFrames + (int)batch.batchSize;

            // Push valid batch to processing
            preProcessQueue_.Push(std::move(batch));
        }
    }

    // Shutdown Sequence
    // Wait for queues to drain
    while (running_ && (!preProcessQueue_.Empty() || !postProcessQueue_.Empty())) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    running_ = false;
    for (auto& t : workers_) {
        if (t.joinable()) {
            t.join();
        }
    }

    // Capture total wall time
    stats_.totalWallTimeMs = wallTimer.Stop();

    return CudaError();
}

void InferencePipeline::inferenceWorker() {
    Stopwatch sw;
    while (running_) {
        BatchData batch;
        BatchDetections results;
        // Wait up to 100ms for data
        if (preProcessQueue_.TryPop(batch, std::chrono::milliseconds(100))) {
            try {
                sw.Start();
                CUDA_CALL(detector_->Detect(batch, results));
                double elapsed = sw.Stop();

                stats_.totalDetectionMs = stats_.totalDetectionMs + elapsed;

                // Push successful result
                postProcessQueue_.Push({std::move(batch), std::move(results)});
            } catch (const std::exception& e) {
                std::cerr << "Pipeline Detector Error: " << e.what() << std::endl;
                running_ = false; // Stop pipeline on detector failure
                return;
            }
        }
    }
}

void InferencePipeline::outputWorker() {
    Stopwatch sw;
    while (running_) {
        std::pair<BatchData, BatchDetections> item;
        
        // Wait up to 100ms for data
        if (postProcessQueue_.TryPop(item, std::chrono::milliseconds(100))) {
            try {
                sw.Start();
                CUDA_CALL(sink_->Save(item.first, item.second));
                double elapsed = sw.Stop();

                stats_.totalSinkMs = stats_.totalSinkMs + elapsed;
            } catch (const std::exception& e) {
                // Log but don't necessarily kill the pipeline for one bad frame save
                std::cerr << "Pipeline Sink Error: " << e.what() << std::endl;
            }

            // CRITICAL: We must recycle the input buffer even if Save() failed,
            // otherwise that memory is lost and we will eventually OOM or stop recycling.
            recycleQueue_.Push(std::move(item.first));
        }
    }
}

} // namespace cropandweed
