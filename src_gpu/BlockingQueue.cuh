#pragma once

#include <mutex>
#include <condition_variable>
#include <optional>
#include <atomic>

#include "BoundsUtils.hpp"

template <typename Q, typename Model>
class BlockingQueue
{
public:
    // inFlight is shared across all queues: incremented on every push,
    // decremented by the caller when a node is retired (pruned or solved).
    // Termination invariant: inFlight == 0  ⟹  all work is done.
    BlockingQueue(std::mutex& mutex, std::condition_variable& cv, std::atomic<gfl::i32>& inFlight)
        : mutex_(mutex), cv_(cv), inFlight_(inFlight)
    {
    }

    // ── Single-node push (e.g. root) ─────────────────────────────────
    bool push(auto const* const node)
    {
        {
            std::unique_lock lock(mutex_);
            if (stopped_) return false;
            queue_.push(node);
            ++inFlight_;
        }
        cv_.notify_all();
        return true;
    }

    // ── Validator pull: up to n nodes, filters against primal ────────
    template <typename N>
    bool pullUpTo(gfl::i32 const n, std::vector<N>& bufferIn, gfl::f64& bestF, gfl::f64 const primal)
    {
        std::unique_lock lock(mutex_);
        cv_.wait(lock, [&] {
            return !queue_.empty() || stopped_ || inFlight_.load() == 0;
        });
        if (queue_.empty() || stopped_) return false;

        bestF = worst<Model>();
        while (bufferIn.size() < static_cast<size_t>(n) && !queue_.empty())
        {
            auto node = queue_.pullBest();
            if (not isBetter<Model>(node->f(), primal))
            {
                --inFlight_;   // pruned here, caller won't see it
                continue;
            }
            bestF = better<Model>(bestF, node->f());
            bufferIn.push_back(node);
        }
        return true;
    }

    // ── GPU pull: up to n nodes, same f/depth, filters against primal ─
    template <typename N>
    bool pullUpTo(gfl::i32 const n, std::vector<N>& bufferIn, gfl::f64 const primal)
    {
        using namespace gfl;
        std::unique_lock lock(mutex_);
        cv_.wait(lock, [&] {
            return !queue_.empty() || stopped_ || inFlight_.load() == 0;
        });
        if (queue_.empty() || stopped_) return false;

        // Drain any stale nodes at the front before starting the batch
        while (!queue_.empty() && not isBetter<Model>(queue_.peekBest()->f(), primal))
        {
            queue_.pullBest();
            --inFlight_;
        }
        if (queue_.empty()) return false;

        bufferIn.push_back(*queue_.pullBest());
        f64 const fValue = bufferIn.back().f();
        i32 const depth  = bufferIn.back().depth();

        while (bufferIn.size() < static_cast<size_t>(n) && !queue_.empty() &&
               queue_.peekBest()->f()     == fValue &&
               queue_.peekBest()->depth() == depth)
        {
            bufferIn.push_back(*queue_.pullBest());
        }
        return true;
    }

    // ── Batch push (validator → readyQueue) ──────────────────────────
    // Sets bestF to the current best in the queue (used for dual tracking).
    template <typename N>
    bool push(std::vector<N> const& batch, gfl::f64& bestF)
    {
        {
            std::unique_lock lock(mutex_);
            if (stopped_) return false;
            for (auto const& node : batch)
                queue_.push(node);
            inFlight_ += static_cast<gfl::i32>(batch.size());
            bestF = worst<Model>();
        }
        cv_.notify_all();
        return true;
    }

    // ── GPU cutset push ───────────────────────────────────────────────
    template <typename Node>
    void push(std::vector<gfl::ArrayView<Node>> const& cutsetFragments,
              gfl::f64 const f, gfl::f64 const primal)
    {
        using namespace gfl;
        i32 count = 0;
        {
            std::unique_lock lock(mutex_);
            if (stopped_) return;
            for (auto const& fragment : cutsetFragments)
            {
                queue_.pushBatch(fragment, f, primal);
                count += static_cast<i32>(fragment.size());
            }
            inFlight_ += count;
        }
        cv_.notify_all();
    }

    // ── Primal callback: drain readyQueue back into pendingQueue ─────
    // Nodes just move between queues — inFlight is unchanged.
    void drainInto(BlockingQueue& other)
    {
        {
            std::unique_lock lock(mutex_);
            while (not queue_.empty())
                other.queue_.push(queue_.pullBest());
        }
        cv_.notify_all();
    }

    // ── Misc ─────────────────────────────────────────────────────────
    auto peekBest()
    {
        std::unique_lock lock(mutex_);
        if (queue_.empty()) return decltype(queue_.peekBest()){nullptr};
        return queue_.peekBest();
    }

    auto peekBestBlocking(std::atomic<gfl::i32> & inFlight)
    {
        std::unique_lock lock(mutex_);
        cv_.wait(lock, [&] {
            return !queue_.empty() || stopped_ || inFlight.load() == 0;
        });
        if (queue_.empty() || stopped_) return decltype(queue_.peekBest()){nullptr};
        return queue_.peekBest();
    }

    bool empty() const
    {
        std::unique_lock lock(mutex_);
        return queue_.empty();
    }

    gfl::i64 size() const
    {
        std::unique_lock lock(mutex_);
        return queue_.size();
    }

    void stop()
    {
        {
            std::unique_lock lock(mutex_);
            stopped_ = true;
        }
        cv_.notify_all();
    }

    // Add to BlockingQueue:
    void done(gfl::i32 const n)
    {
        inFlight_ -= n;
        if (inFlight_.load() == 0)
            cv_.notify_all();  // wake all waiters so they can observe inFlight == 0
    }

    bool emptyUnlocked() const  { return queue_.empty(); }
    auto peekBestUnlocked()     { return queue_.peekBest(); }
    bool isStopped() const      { return stopped_; }
    gfl::i64 pulled() const noexcept { return queue_.pulled(); }

private:
    Q                           queue_;
    std::mutex&                 mutex_;
    std::condition_variable&    cv_;
    std::atomic<gfl::i32>&      inFlight_;   // shared ref — incremented here, decremented by callers
    std::atomic<bool>           stopped_ = false;
};