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
    gfl::Vector<gfl::i16> solution_{128};

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

    bool solved() const
    {
        return isBetterEq<Model>(primal_,dual_);
    }

    gfl::f64 gap() const noexcept
    {
        return 100.0 * std::abs(primal_ - dual_) / std::abs(primal_);
    }

    bool pruneAncestor(Node const & node) noexcept
    {
        return isWorseEq<Model>(node.g(), primal_);
    }

    gfl::f64 primal() const noexcept { return primal_; }

    void primal(Node const & node) noexcept
    {
        using namespace gfl;

        if (not node.approximated() and isBetter<Model>(node.g(), primal_))
        {
            primal_ = node.g();
            auto const path = node.path();
            solution_.resizeTo(path.size());
            memcpy(solution_.data(), path.data(), path.dataMemSize());
            notifyPrimal();
        }
    }

    gfl::f64 dual() const noexcept { return dual_; }

    void dual(gfl::f64 const dual) noexcept
    {
        if (isValid<Model>(dual) and isTighter<Model>(dual, dual_))
        {
            dual_ = dual;
            notifyDual();
        }
    }

    void dual(Node const & node) noexcept
    {
        using namespace gfl;
        // Node should be a target from relaxed
        if (not isValid<Model>(dual_) and isValid<Model>(node.g()))
        {
            dual_ = node.g();
            notifyDual();
        }
    }

    gfl::ArrayView<gfl::i16> solution() const noexcept { return solution_; }
};