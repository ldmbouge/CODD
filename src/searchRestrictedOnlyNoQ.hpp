#ifndef __SEARCH_RESTRICTED_ONLY_NO_Q_HPP__
#define __SEARCH_RESTRICTED_ONLY_NO_Q_HPP__

#include <functional>
#include "dd.hpp"
#include "store.hpp"
#include "search.hpp"

class BAndBRestrictedOnlyNoQ: BAndB {
    using BAndB::BAndB;
public:
    void search(Bounds &bnds);
};

#endif
