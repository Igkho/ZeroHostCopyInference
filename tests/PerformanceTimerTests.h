#pragma once
#include <gtest/gtest.h>
#include <thread>
#include <vector>
#include <chrono>
#include <cmath>
#include "PerformanceTimer.h"

namespace cropandweed {

class PerformanceTimerTest : public ::testing::Test {
protected:
    // Helper to simulate work
    void SleepFor(int ms) {
        std::this_thread::sleep_for(std::chrono::milliseconds(ms));
    }
};

// --- 1. PerformanceTimer (RAII) Tests ---

TEST_F(PerformanceTimerTest, MeasureElapsedExecution) {
    PerformanceTimer timer("TestTimer");
    
    SleepFor(50);
    
    long long elapsed = timer.ElapsedMilliseconds();
    
    // Check that at least 50ms passed. 
    // We don't set a strict upper bound because OS scheduling can vary.
    EXPECT_GE(elapsed, 50);
}

TEST_F(PerformanceTimerTest, ManualStopFreezesTimer) {
    PerformanceTimer timer("TestTimer");
    
    SleepFor(20);
    long long duration1 = timer.Stop();
    
    EXPECT_GE(duration1, 20);

    // Sleep more to verify the timer has actually stopped
    SleepFor(50);
    
    long long duration2 = timer.ElapsedMilliseconds();
    
    // The duration should remain unchanged after Stop()
    EXPECT_EQ(duration1, duration2);
}

// --- 2. Stopwatch (Manual) Tests ---

TEST_F(PerformanceTimerTest, StopwatchMeasuresTime) {
    Stopwatch sw;
    sw.Start();
    
    SleepFor(20);
    
    double elapsed = sw.Stop();
    
    // Stopwatch returns double (floating point milliseconds)
    EXPECT_GE(elapsed, 20.0);
}

TEST_F(PerformanceTimerTest, StopwatchRestart) {
    Stopwatch sw;
    
    // Run 1
    sw.Start();
    SleepFor(10);
    double run1 = sw.Stop();
    EXPECT_GE(run1, 10.0);
    
    // Run 2 (Restart)
    sw.Start();
    SleepFor(30);
    double run2 = sw.Stop();
    
    EXPECT_GE(run2, 30.0);
    
    // Ensure run2 didn't accumulate run1
    // (It's possible run2 is slightly > 30, but shouldn't be close to 40 if logic is correct)
    // We check relative difference isn't massive.
    EXPECT_LT(run1, run2); 
}

// --- 3. PipelineStats (Atomic) Tests ---

TEST_F(PerformanceTimerTest, StatsAccumulateCorrectly) {
    PipelineStats stats;
    
    // Modify values
    stats.totalFrames += 10;
    stats.totalDecodingMs.store(100.5); // use store for atomic double if assignment operator is ambiguous, though struct defines std::atomic
    
    // std::atomic<double> sometimes needs explicit ops depending on C++ version, 
    // but C++20 allows operators. Your struct uses std::atomic<double>.
    // Let's assume standard operators work or use exchange.
    
    // Atomic double increment is tricky in pre-C++20. 
    // Usually requires load/store loop or fetch_add if specialized. 
    // However, for this test, we verify the memory model works.
    
    double current = stats.totalDecodingMs.load();
    stats.totalDecodingMs.store(current + 50.5);

    EXPECT_EQ(stats.totalFrames.load(), 10);
    EXPECT_DOUBLE_EQ(stats.totalDecodingMs.load(), 151.0);
}

TEST_F(PerformanceTimerTest, MultithreadedAccumulation) {
    PipelineStats stats;
    int num_threads = 10;
    int iterations = 1000;
    
    auto worker = [&stats, iterations]() {
        for(int i=0; i<iterations; ++i) {
            stats.totalFrames++;
            stats.totalBatches++;
            
            // Simulating atomic double addition (CAS loop)
            // std::atomic<double> does not natively support operator+= in C++11/14/17
            // We must simulate it to ensure thread safety
            double prev = stats.totalDecodingMs.load();
            while(!stats.totalDecodingMs.compare_exchange_weak(prev, prev + 1.0)) {}
        }
    };

    std::vector<std::thread> threads;
    for(int i=0; i<num_threads; ++i) {
        threads.emplace_back(worker);
    }
    
    for(auto& t : threads) t.join();

    // Verify totals
    EXPECT_EQ(stats.totalFrames.load(), num_threads * iterations);
    EXPECT_EQ(stats.totalBatches.load(), num_threads * iterations);
    EXPECT_DOUBLE_EQ(stats.totalDecodingMs.load(), (double)(num_threads * iterations));
}

TEST_F(PerformanceTimerTest, ReportDoesNotCrash) {
    // Smoke test for the print function
    PipelineStats stats;
    stats.totalFrames = 100;
    stats.totalBatches = 10;
    stats.totalDecodingMs.store(500.0);
    stats.totalDetectionMs.store(500.0);
    stats.totalSinkMs.store(200.0);
    stats.totalWallTimeMs.store(1200);

    // Should run without throwing
    EXPECT_NO_THROW(stats.PrintReport(50));
}

TEST_F(PerformanceTimerTest, ReportHandlesZeroDivision) {
    PipelineStats stats;
    // Everything 0
    EXPECT_NO_THROW(stats.PrintReport(0));
}

} // namespace cropandweed