#pragma once

#include <mutex>
#include <condition_variable>
#include <optional>
#include <atomic>

#include "BoundsUtils.hpp"

template<typename Q, typename Model>
class BlockingQueue {
public:
    BlockingQueue(std::atomic<gfl::i32> & inFlight,
                std::mutex & mutex,
                std::condition_variable & cv)
      : inFlight_(inFlight), mutex_(mutex), cv_(cv) {}



    bool push(auto const * const node) {
        {
            std::unique_lock lock(mutex_);
            if (stopped_) return false;
            queue_.push(node);
            inFlight_++;
        }
        cv_.notify_all();
        return true;
    }

    bool push(auto const * const node, gfl::f64 const primal) {
        {
            std::unique_lock lock(mutex_);
            if (stopped_) return false;
            queue_.push(node, primal);
            inFlight_++;
        }
        cv_.notify_all();
        return true;
    }

    void drainInto(BlockingQueue & other) {
        {
            std::unique_lock lock(mutex_);
            while (not queue_.empty()) {
                other.queue_.push(queue_.pullBest());
            }
        } // lock released here
        cv_.notify_all(); // then notify
    }

    template<typename Node>
    void push(std::vector<gfl::ArrayView<Node>> const & cutsetFragments, gfl::f64 const f, gfl::f64 const primal) {
        // Pre-process nodes BEFORE acquiring the lock
        for (auto const & fragment : cutsetFragments)
            for (auto & node : fragment)
                node.h(worse<Model>(f - node.g(), node.h()));

        // Lock only for the actual queue insertion
        {
            std::unique_lock lock(mutex_);
            if (stopped_) return;
            for (auto const & fragment : cutsetFragments) {
                for (auto & node : fragment) {
                    queue_.push(&node, primal);
                    inFlight_++;
                }
            }
        }
        cv_.notify_all();
    }

    // Blocking pop - waits until something is available or stopped
    auto pop() {
        using ReturnType = decltype(queue_.pullBest());
        std::unique_lock lock(mutex_);
        cv_.wait(lock, [&] { return !queue_.empty() || stopped_; });
        if (queue_.empty()) return std::optional<ReturnType>{std::nullopt};
        return std::optional<ReturnType>{queue_.pullBest()};
    }

    // Blocking pull of up to n nodes - waits for at least one, then greedily takes more
    template<typename N>
    auto pullUpTo(gfl::i32 const n, std::vector<N> & bufferIn) {
        std::unique_lock lock(mutex_);
        // Block until at least one node is available or stopped
        cv_.wait(lock, [&] { return !queue_.empty() || stopped_; });
        if (queue_.empty()) return false;

        while (bufferIn.size() < static_cast<size_t>(n) && !queue_.empty()) {
            bufferIn.push_back(queue_.pullBest());
        }
        return true;
    }


    // Push a batch without incrementing inFlight (nodes already counted)
    template<typename N>
    bool pushBatchNoCount(std::vector<N> const & batch) {
        {
            std::unique_lock lock(mutex_);
            if (stopped_) return false;
            for (auto const & node : batch) {
                queue_.push(node);
            }
        }
        cv_.notify_all();
        return true;
    }

    void done() {
        inFlight_--;
        cv_.notify_all();  // wake popOrDone in case inFlight hit 0
    }

    auto popOrDone() {
        using ReturnType = decltype(queue_.pullBest());
        std::unique_lock lock(mutex_);
        cv_.wait(lock, [&] {
            return !queue_.empty() || stopped_ || inFlight_ == 0;
        });
        if (queue_.empty()) return std::optional<ReturnType>{std::nullopt};
        auto node = queue_.pullBest();
        inFlight_--;          // node leaves system, going to GPU
        cv_.notify_all();
        return std::optional<ReturnType>{node};
    }

    // Push without incrementing inFlight (node already counted)
    bool pushNoCount(auto const * const node) {
        {
            std::unique_lock lock(mutex_);
            if (stopped_) return false;
            queue_.push(node);
        }
        cv_.notify_all();
        return true;
    }

    auto popIfBlocking(gfl::f64 const pBound) {
        using ReturnType = decltype(queue_.pullBest());
        std::unique_lock lock(mutex_);
        cv_.wait(lock, [&] { return !queue_.empty() || stopped_; });
        if (queue_.empty()) return std::optional<ReturnType>{std::nullopt};
        if (isWorseEq<Model>(queue_.peekBest()->f(),pBound)) return std::optional<ReturnType>{std::nullopt};
        return std::optional<ReturnType>{queue_.pullBest()};
    }

    void notifyAll() { cv_.notify_all(); }

    // Non-blocking conditional pop - returns nullopt if empty or best f() != expectedF
    auto popIf(gfl::f64 const pBound) {
        using ReturnType = decltype(queue_.pullBest());
        std::unique_lock lock(mutex_);
        if (queue_.empty()) return std::optional<ReturnType>{std::nullopt};
        if (isBetter<Model>(queue_.peekBest()->f(), pBound))
        {
            return std::optional<ReturnType>{queue_.pullBest()};
        }
        return std::optional<ReturnType>{std::nullopt};
    }

    // Non-blocking conditional pop - returns nullopt if empty or best f() != expectedF
    auto popIf(gfl::f64 const fValue, gfl::i64 const depth) {
        using ReturnType = decltype(queue_.pullBest());
        std::unique_lock lock(mutex_);
        if (queue_.empty()) return std::optional<ReturnType>{std::nullopt};
        if (queue_.peekBest()->f() == fValue and queue_.peekBest()->depth() == depth)
        {
            auto node = queue_.pullBest();
            inFlight_--;          // leaving system
            cv_.notify_all();
            return std::optional<ReturnType>{node};
        }
        return std::optional<ReturnType>{std::nullopt};
    }

    auto peekBest() {
        std::unique_lock lock(mutex_);
        if (queue_.empty()) return decltype(queue_.peekBest()){nullptr};
        return queue_.peekBest();
    }

    bool     empty() const { std::unique_lock lock(mutex_); return queue_.empty(); }
    gfl::i64 size()  const { std::unique_lock lock(mutex_); return queue_.size(); }

    void stop() {
        { std::unique_lock lock(mutex_); stopped_ = true; }
        cv_.notify_all();
    }

    bool isStopped() const { return stopped_; }

    gfl::i64 pulled() const noexcept {return queue_.pulled();}

private:
    Q                       queue_;
    std::mutex &                mutex_;      // shared reference
    std::condition_variable &   cv_;         // shared reference
    std::atomic<bool>       stopped_ = false;
    std::atomic<gfl::i32> &       inFlight_;
};
