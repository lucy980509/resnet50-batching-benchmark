#pragma once

#include <condition_variable>
#include <deque>
#include <mutex>

#include "request.hpp"

class RequestQueue {
public:
    bool push(Request request) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (closed_) {
                return false;
            }
            request.enqueued_at = Clock::now();
            queue_.push_back(request);
        }
        ready_.notify_one();
        return true;
    }

    bool pop(Request& request) {
        std::unique_lock<std::mutex> lock(mutex_);
        ready_.wait(lock, [this] { return closed_ || !queue_.empty(); });
        return take_front(request);
    }

    bool pop_until(Request& request, TimePoint deadline) {
        std::unique_lock<std::mutex> lock(mutex_);
        ready_.wait_until(lock, deadline, [this] {
            return closed_ || !queue_.empty();
        });
        if (queue_.empty() || queue_.front().enqueued_at > deadline) {
            return false;
        }
        return take_front(request);
    }

    bool try_pop(Request& request){
        std::lock_guard<std::mutex> lock(mutex_);
        return take_front(request);
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

    void close() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            closed_ = true;
        }
        ready_.notify_all();
    }

private:
    bool take_front(Request& request) {
        if (queue_.empty()) {
            return false;
        }
        request = queue_.front();
        queue_.pop_front();
        request.dequeued_at = Clock::now();
        return true;
    }

    std::deque<Request> queue_;
    mutable std::mutex mutex_;
    std::condition_variable ready_;
    bool closed_ = false;
};
