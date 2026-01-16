#pragma once

#include <chrono>
#include <iostream>
#include <string>
#include <atomic>
#include <iomanip>

namespace cropandweed {

// Original RAII Timer (kept for backward compatibility if used elsewhere)
class PerformanceTimer {
public:
    explicit PerformanceTimer(const std::string& name = "Operation")
        : name_(name), start_time_(std::chrono::high_resolution_clock::now()) {}

    ~PerformanceTimer() {
        if (!stopped_) {
            Stop();
            std::cout << "[PERFORMANCE] " << name_ << " took " << duration_ms_ << " ms" << std::endl;
        }
    }

    // Allow manual stop to get value
    long long Stop() {
        if (!stopped_) {
            auto end_time = std::chrono::high_resolution_clock::now();
            duration_ms_ = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time_).count();
            stopped_ = true;
        }
        return duration_ms_;
    }

    long long ElapsedMilliseconds() const {
        if (stopped_) return duration_ms_;
        auto end_time = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time_).count();
    }

private:
    std::string name_;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time_;
    long long duration_ms_ = 0;
    bool stopped_ = false;
};

// --- NEW: lightweight manual timer for loops ---
class Stopwatch {
public:
    void Start() {
        start_ = std::chrono::high_resolution_clock::now();
    }

    // Returns elapsed milliseconds since Start()
    double Stop() {
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> ms = end - start_;
        return ms.count();
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

// --- NEW: Thread-safe Stats Accumulator ---
struct PipelineStats {
    std::atomic<double> totalDecodingMs{0};
    std::atomic<double> totalDetectionMs{0};
    std::atomic<double> totalSinkMs{0};

    std::atomic<long long> totalWallTimeMs{0};

    std::atomic<int> totalBatches{0};
    std::atomic<int> totalFrames{0};

    // Helper to print the final report
    void PrintReport(long long initTimeMs) {
        double decoding = totalDecodingMs.load();
        double detection = totalDetectionMs.load();
        double sink = totalSinkMs.load();
        long long wall = totalWallTimeMs.load();
        int batches = totalBatches.load();
        int frames = totalFrames.load();

        if (batches == 0) batches = 1; // Prevent div by zero
        if (frames == 0) frames = 1;

        double totalComponentTime = decoding + detection + sink;
        if (totalComponentTime <= 0.0001) totalComponentTime = 1.0;

        std::cout << "\n=======================================================\n";
        std::cout << "               INFERENCE PIPELINE REPORT               \n";
        std::cout << "=======================================================\n";

        std::cout << "Total Frames Processed: " << frames << "\n";
        std::cout << "Total Batches Processed: " << batches << "\n\n";

        std::cout << std::fixed << std::setprecision(2);

        // 1. Initialization
        std::cout << "1. Initialization: " << initTimeMs << " ms\n";

        // 2. Wall Clock
        std::cout << "2. Pipeline Wall Time: " << wall << " ms ("
                  << (1000.0 * frames / wall) << " FPS)\n\n";

        // 3. Detailed Breakdown
        auto print_stage = [&](const char* name, double totalMs) {
            double avgBatch = totalMs / batches;
            double avgFrame = totalMs / frames;
            double percent = (totalMs / totalComponentTime) * 100.0;

            std::cout << name << ":\n";
            std::cout << "   - Total:     " << totalMs << " ms\n";
            std::cout << "   - Avg/Batch: " << avgBatch << " ms\n";
            std::cout << "   - Avg/Frame: " << avgFrame << " ms\n";
            std::cout << "   - Share:     " << percent << "% (of active work)\n\n";
        };

        print_stage("3. Decoding (Source)", decoding);
        print_stage("4. Inference (Detector)", detection);
        print_stage("5. Storage (Sink)", sink);

        std::cout << "=======================================================\n";
    }
};

} // namespace cropandweed
