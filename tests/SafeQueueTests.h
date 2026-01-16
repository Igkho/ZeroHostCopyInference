#pragma once
#include <gtest/gtest.h>
#include <thread>
#include <vector>
#include <atomic>
#include <algorithm>
#include <numeric>
#include <chrono>
#include "SafeQueue.h"

namespace cropandweed {

class SafeQueueTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// --- 1. Basic Functionality ---

TEST_F(SafeQueueTest, BasicPushPop) {
    SafeQueue<int> q;
    
    // Initial state
    EXPECT_TRUE(q.Empty());
    EXPECT_EQ(q.Size(), 0);

    // Push items
    q.Push(10);
    q.Push(20);
    q.Push(30);

    EXPECT_FALSE(q.Empty());
    EXPECT_EQ(q.Size(), 3);

    // Pop items
    int val = 0;
    bool success = q.TryPop(val, std::chrono::milliseconds(1)); // Instant pop
    EXPECT_TRUE(success);
    EXPECT_EQ(val, 10);
    EXPECT_EQ(q.Size(), 2);

    success = q.TryPop(val, std::chrono::milliseconds(1));
    EXPECT_TRUE(success);
    EXPECT_EQ(val, 20);

    success = q.TryPop(val, std::chrono::milliseconds(1));
    EXPECT_TRUE(success);
    EXPECT_EQ(val, 30);

    // Queue empty now
    EXPECT_TRUE(q.Empty());
    EXPECT_EQ(q.Size(), 0);
}

TEST_F(SafeQueueTest, TimeoutBehavior) {
    SafeQueue<int> q;
    int val = 0;

    auto start = std::chrono::steady_clock::now();
    bool success = q.TryPop(val, std::chrono::milliseconds(50));
    auto end = std::chrono::steady_clock::now();

    EXPECT_FALSE(success); // Should fail (timeout)
    
    // Verify it waited at least the requested duration
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    EXPECT_GE(elapsed, 50);
}

// --- 2. Multithreaded Stress Test ---

TEST_F(SafeQueueTest, MultiThreadedPushPop) {
    SafeQueue<int> q;
    const int num_threads = 10;
    const int items_per_thread = 1000;
    const int total_items = num_threads * items_per_thread;

    // Output storage
    std::mutex results_mutex;
    std::vector<int> popped_values;
    popped_values.reserve(total_items);

    // Atomic counter to coordinate shutdown of consumers
    std::atomic<int> consumed_count{0};

    // Producer Lambda
    auto producer = [&](int thread_id) {
        for (int i = 0; i < items_per_thread; ++i) {
            // Generate unique numbers: thread_id * 1000 + i
            // e.g., Thread 0: 0-999, Thread 1: 1000-1999, etc.
            int val = thread_id * items_per_thread + i;
            q.Push(val);
        }
    };

    // Consumer Lambda
    auto consumer = [&]() {
        int val;
        // Continue trying to pop until we know global target is reached
        while (consumed_count < total_items) {
            // Use a short timeout to keep checking the condition
            if (q.TryPop(val, std::chrono::milliseconds(10))) {
                std::lock_guard<std::mutex> lock(results_mutex);
                popped_values.push_back(val);
                consumed_count++;
            }
        }
    };

    // Launch threads
    std::vector<std::thread> producers;
    std::vector<std::thread> consumers;

    for (int i = 0; i < num_threads; ++i) {
        producers.emplace_back(producer, i);
        consumers.emplace_back(consumer);
    }

    // Join threads
    for (auto& t : producers) t.join();
    for (auto& t : consumers) t.join();

    // Verification
    EXPECT_EQ(popped_values.size(), total_items);
    EXPECT_EQ(consumed_count.load(), total_items);
    EXPECT_TRUE(q.Empty());

    // Verify Data Integrity
    // 1. Sort the output
    std::sort(popped_values.begin(), popped_values.end());

    // 2. Generate expected linear sequence 0 to total_items-1
    //    (Because we generated thread_id * items_per_thread + i)
    std::vector<int> expected_values(total_items);
    std::iota(expected_values.begin(), expected_values.end(), 0);

    // 3. Compare
    EXPECT_EQ(popped_values, expected_values) << "Data corruption: Popped values do not match pushed values.";
}

// --- 3. Advanced Type Safety & Memory Management ---

TEST_F(SafeQueueTest, HandlesMoveOnlyTypes) {
    // std::unique_ptr cannot be copied, only moved. 
    // This confirms SafeQueue uses std::move correctly internally.
    SafeQueue<std::unique_ptr<int>> q;

    // Push a move-only object
    q.Push(std::make_unique<int>(42));
    
    EXPECT_EQ(q.Size(), 1);

    // Pop into a move-only object
    std::unique_ptr<int> result;
    bool success = q.TryPop(result, std::chrono::milliseconds(10));
    
    EXPECT_TRUE(success);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(*result, 42);
    EXPECT_EQ(q.Size(), 0);
}

// Helper struct to track object creation/destruction
struct LifetimeTracker {
    static std::atomic<int> active_count;
    
    LifetimeTracker() { active_count++; }
    ~LifetimeTracker() { active_count--; }
    
    // Copying creates a new object, so increment
    LifetimeTracker(const LifetimeTracker&) { active_count++; }
    
    // Moving creates a new object (target), source remains valid but empty.
    // Both destructors will eventually run, so we strictly increment here.
    LifetimeTracker(LifetimeTracker&&) noexcept { active_count++; }

    // Assignments
    LifetimeTracker& operator=(const LifetimeTracker&) = default;
    LifetimeTracker& operator=(LifetimeTracker&&) = default;
};

// Initialize static member
std::atomic<int> LifetimeTracker::active_count{0};

TEST_F(SafeQueueTest, DestructorCleansUpRemainingItems) {
    // Ensure clean state
    LifetimeTracker::active_count = 0;

    {
        SafeQueue<LifetimeTracker> q;
        
        // Push 3 items
        q.Push(LifetimeTracker{});
        q.Push(LifetimeTracker{});
        q.Push(LifetimeTracker{});

        // Verify items are alive inside the queue
        // (Note: count might be transiently higher during moves in Push, 
        // but should settle to queue size after return)
        EXPECT_EQ(q.Size(), 3);
        EXPECT_EQ(LifetimeTracker::active_count.load(), 3);
        
    } // <--- Queue goes out of scope here. Destructor runs.

    // Verify all items contained in the queue were destroyed
    EXPECT_EQ(LifetimeTracker::active_count.load(), 0);
}

} // namespace cropandweed