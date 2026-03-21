#pragma once
#include <queue>
#include <mutex>
#include <condition_variable>
#include <chrono>

namespace cropandweed {

template <typename T>
class SafeQueue {
public:

    // Added capacity parameter to enable backpressure (0 = Unbounded)
    explicit SafeQueue(size_t capacity = 0) : capacity_(capacity) {}
    ~SafeQueue() = default;

    // Disable copying
    SafeQueue(const SafeQueue&) = delete;
    SafeQueue& operator=(const SafeQueue&) = delete;

    /**
     * Pushes a value into the queue.
     */
    void Push(T value) {
//        std::lock_guard<std::mutex> lock(mutex_);
        std::unique_lock<std::mutex> lock(mutex_);

        // Suspend Producer thread if queue is full (Backpressure)
        if (capacity_ > 0) {
            cond_full_.wait(lock, [this] { return queue_.size() < capacity_; });
        }
        queue_.push(std::move(value));
        cond_empty_.notify_one();
    }

    /**
     * Tries to pop an item from the queue with a timeout.
     * Returns true if successful, false if the queue remained empty for the duration.
     */
    template <typename Rep, typename Period>
    bool TryPop(T& outValue, const std::chrono::duration<Rep, Period>& timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!cond_empty_.wait_for(lock, timeout, [this] { return !queue_.empty(); })) {
            return false; // Timeout
        }
        
        outValue = std::move(queue_.front());
        queue_.pop();

        // Wake up the Producer thread if it was blocked waiting for space
        if (capacity_ > 0) {
            cond_full_.notify_one();
        }
        return true;
    }

    /**
     * Non-blocking check for emptiness.
     */
    bool Empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

    /**
     * Returns current size (approximate in concurrent context).
     */
    size_t Size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

private:
    std::queue<T> queue_;
    mutable std::mutex mutex_;
    std::condition_variable cond_empty_; // Waits for items to process
    std::condition_variable cond_full_;  // Waits for space to become available
    size_t capacity_;                    // Max items allowed before blocking};
};

} // namespace cropandweed
