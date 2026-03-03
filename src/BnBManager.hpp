#pragma once

#include <GFL.hpp>
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

public:
    BnBManager() = default;
    ~BnBManager() = default;

    BnBManager(BnBManager const &) = delete;
    BnBManager(BnBManager &&) = delete;

    BnBManager & operator=(BnBManager const &) = delete;
    BnBManager & operator=(BnBManager &&) = delete;

    void onPrimal(PrimalListener l) { primalListeners.emplace_back(std::move(l));}
    void onDual(DualListener l) { dualListeners.emplace_back(std::move(l));}

    void
    notifyPrimal()
    {
        for(auto const & l : primalListeners) { l(); }
    }

    void
    notifyDual()
    {
        for(auto const & l : dualListeners) { l(); }
    }

    bool hasPrimal() const noexcept { return primal_ != worst<Model>(); }

    gfl::f64 primal() const noexcept { return primal_; }

    void primal(Node const & node) noexcept
    {
        using namespace gfl;
        if (not node.approximated() and isBetter<Model>(node.g(), primal_))
        {
            primal_ = node.g();
            solution_ = node;
            notifyPrimal();
        }
    }

    bool hasDual() const noexcept { return dual_ != best<Model>();}

    gfl::f64 dual() const noexcept {return dual_;}

    void dual(gfl::f64 const dual) noexcept
    {
        //printf("[DBG] BNB Dual: %.2f vs IN Dual: %.2f\n", dual_, dual);
        if (not hasDual() or isWorse<Model>(dual, dual_))
        {
         //   printf("BNB Dual: %.2f vs IN Dual: %.2f\n", dual_, dual);
         //   printf("%.2f > %.2f ? %d\n", dual , dual_, dual > dual_);
            dual_ = dual;
            notifyDual();
        }
    }

    bool hasGap() const noexcept { return hasPrimal() and hasDual(); }

    gfl::f64 gap() const noexcept
    {
        return 100.0 * std::abs(primal_ - dual_) / std::abs(primal_);
    }

    bool solved() const noexcept
    {
        return hasGap() and isWorse<Model>(dual_,primal_);
    }

    void printSolution() const noexcept { return solution_.printSolution(); }
};