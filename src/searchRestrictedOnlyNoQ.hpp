#pragma once

#include <functional>
#include "dd.hpp"
#include "store.hpp"
#include "search.hpp"

class BAndBRestrictedOnlyNoQ: public BAndB
{
    using BAndB::BAndB;

public:
    void search(Bounds &bnds);
};
