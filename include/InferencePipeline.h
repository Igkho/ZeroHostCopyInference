#pragma once

#include <memory>
#include <vector>
#include <thread>
#include <atomic>
#include <utility>
#include "Interfaces.h"
#include "SafeQueue.h"
#include "PerformanceTimer.h"

namespace cropandweed {

class InferencePipeline {
public:
    InferencePipeline(std::unique_ptr<ISource> src,
                      std::unique_ptr<IDetector> det,
                      std::unique_ptr<ISink> sink,
                      int batchSize);

    ~InferencePipeline();

    /**
     * Starts the processing loop.
     * This function blocks the calling thread until the Source runs out of data.
     */
    CudaError Run();

    // Prints the final statistics using the gathered data
    void PrintStats(long long initTimeMs);

private:
    // Dependencies
    std::unique_ptr<ISource> source_;
    std::unique_ptr<IDetector> detector_;
    std::unique_ptr<ISink> sink_;

    // Data Queues
    SafeQueue<BatchData> preProcessQueue_;
    // Stores pair of {Input Data, Output Detections}
    SafeQueue<std::pair<BatchData, BatchDetections>> postProcessQueue_;
    // Recycling queue to prevent GPU malloc/free every frame
    SafeQueue<BatchData> recycleQueue_;
    // Recycling queue for Detections
    SafeQueue<BatchDetections> resultRecycleQueue_;

    // Threading
    std::vector<std::thread> workers_;
    std::atomic<bool> running_{true};
    int batchSize_;

    // Performance Tracking
    PipelineStats stats_;

    // Atomic counter to track items currently being processed across all threads
    std::atomic<int> batchesInFlight_{0};

    // Internal worker functions
    void inferenceWorker();
    void outputWorker();
};

} // namespace cropandweed
