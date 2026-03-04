#pragma once

#include <GFL.hpp>
#include <shared_mutex>
#include "BoundsUtils.hpp"

template<typename Model, typename Node>
class BnBManager
{
    using PrimalListener = std::function<void()>;
    using DualListener = std::function<void()>;
    std::vector<PrimalListener> primalListeners;
    std::vector<DualListener> dualListeners;

    gfl::f64 primal_{worst<Model>()};
    gfl::f64 dual_{best<Model>()};
    bool queueExhausted_{false};
    Node solution_;

    mutable std::shared_mutex mutex_;

    // ── private unlocked ────────────────────────────────────────────
    bool hasPrimal_() const noexcept { return primal_ != worst<Model>(); }
    bool hasDual_()   const noexcept { return dual_   != best<Model>(); }
    bool hasGap_()    const noexcept { return hasPrimal_() and hasDual_(); }
    bool solved_()    const noexcept { return hasGap_() and isBetterEq<Model>(primal_, dual_); }

public:
    BnBManager() = default;
    ~BnBManager() = default;

    BnBManager(BnBManager const &) = delete;
    BnBManager(BnBManager &&) = delete;

    BnBManager & operator=(BnBManager const &) = delete;
    BnBManager & operator=(BnBManager &&) = delete;

    void onPrimal(PrimalListener l) { primalListeners.emplace_back(std::move(l)); }
    void onDual(DualListener l)     { dualListeners.emplace_back(std::move(l)); }

    void notifyPrimal() { for (auto const & l : primalListeners) l(); }
    void notifyDual()   { for (auto const & l : dualListeners)   l(); }

    bool     hasPrimal() const noexcept { std::shared_lock lock(mutex_); return hasPrimal_(); }
    bool     hasDual()   const noexcept { std::shared_lock lock(mutex_); return hasDual_(); }
    bool     hasGap()    const noexcept { std::shared_lock lock(mutex_); return hasGap_(); }
    bool     solved()    const noexcept { std::shared_lock lock(mutex_); return solved_(); }
    gfl::f64 primal()    const noexcept { std::shared_lock lock(mutex_); return primal_; }
    gfl::f64 dual()      const noexcept { std::shared_lock lock(mutex_); return dual_; }

    void dual(gfl::f64 const dual) noexcept
    {
        {
            std::unique_lock lock(mutex_);
            if (not hasDual_() or isWorse<Model>(dual, dual_))
                dual_ = dual;
            else
                return;
        } // lock released here
        notifyDual(); // listeners can now safely read bnb
    }

    void primal(Node const & node) noexcept
    {
        using namespace gfl;
        {
            std::unique_lock lock(mutex_);
            if (not node.approximated() and isBetter<Model>(node.g(), primal_))
            {
                primal_ = node.g();
                solution_ = node;
            }
            else
                return;
        } // lock released here
        notifyPrimal();
    }
    gfl::f64 gap() const noexcept
    {
        std::shared_lock lock(mutex_);
        return 100.0 * std::abs(primal_ - dual_) / std::abs(primal_);
    }

    void printSolution() const noexcept
    {
        std::shared_lock lock(mutex_);
        solution_.printSolution();
    }
};