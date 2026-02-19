#pragma once

#include <GFL.hpp>
#include "BoundsUtils.hpp"

template<typename Model, typename Node>
class BnBManager
{
    bool primal_updated_{false};
    bool dual_updated_{false};
    gfl::f64 primal_{worst<Model>()};
    gfl::f64 dual_{best<Model>()};
    gfl::Vector<gfl::i16> solution_{128};

public:
    BnBManager() = default;
    ~BnBManager() = default;

    BnBManager(BnBManager const &) = delete;
    BnBManager(BnBManager &&) = delete;

    BnBManager & operator=(BnBManager const &) = delete;
    BnBManager & operator=(BnBManager &&) = delete;

    bool gapClosed() const
    {
        assert(isBetterEq<Model>(dual_, primal_));
        return dual_ == primal_;
    }

    gfl::f64 gap() const noexcept
    {
        return 100.0 * std::abs(primal_ - dual_) / std::abs(primal_);
    }

    bool pruneAncestor(Model const * const model, Node const & node) noexcept
    {
        assert(node.isTarget(model));
        return isWorseEq<Model>(node.primal(), primal_);
    }

    bool primalUpdated() noexcept
    {
        bool const val = primal_updated_;
        primal_updated_ = false;
        return val;
    }

    gfl::f64 primal() const noexcept { return primal_; }

    bool updatePrimal(Model const * const model, Node const & node) noexcept
    {
        using namespace gfl;

        assert(node.isTarget(model));
        if (isBetter<Model>(node.primal(), primal_))
        {
            primal_ = node.primal();
            primal_updated_ = true;
            auto const path = node.path();
            solution_.resizeTo(path.size());
            memcpy(solution_.data(), path.data(), sizeof(i16) * path.size());
            return true;
        }
        return false;
    }

    bool dualUpdated() noexcept
    {
        bool const val = dual_updated_;
        dual_updated_ = false;
        return val;
    }

    gfl::f64 dual() const noexcept { return dual_; }

    bool updateDual(gfl::f64 const bound) noexcept
    {
        if (isWorse<Model>(bound, dual_) and isValid<Model>(bound))
        {
            dual_         = bound;
            dual_updated_ = true;
            return true;
        }
        return false;
    }

    gfl::ArrayView<gfl::i16> solution() const noexcept
    { return solution_.slice(0,solution_.size()); }
};