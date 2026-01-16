#pragma once
#include <queue>
#include <mutex>
#include <condition_variable>
#include <chrono>

namespace cropandweed {

template <typename T>
class SafeQueue {
public:
    SafeQueue() = default;
    ~SafeQueue() = default;

    // Disable copying
    SafeQueue(const SafeQueue&) = delete;
    SafeQueue& operator=(const SafeQueue&) = delete;

    /**
     * Pushes a value into the queue.
     */
    void Push(T value) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(std::move(value));
        cond_.notify_one();
    }

    /**
     * Tries to pop an item from the queue with a timeout.
     * Returns true if successful, false if the queue remained empty for the duration.
     */
    template <typename Rep, typename Period>
    bool TryPop(T& outValue, const std::chrono::duration<Rep, Period>& timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!cond_.wait_for(lock, timeout, [this] { return !queue_.empty(); })) {
            return false; // Timeout
        }
        
        outValue = std::move(queue_.front());
        queue_.pop();
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
    std::condition_variable cond_;
};

} // namespace cropandweed