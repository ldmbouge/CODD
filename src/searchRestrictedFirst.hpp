#ifndef __SEARCH_RESTRICTED_FIRST_HPP__
#define __SEARCH_RESTRICTED_FIRST_HPP__

#include <functional>
#include "dd.hpp"
#include "store.hpp"
#include "search.hpp"

class BAndBRestrictedFirst: BAndB {
    using BAndB::BAndB;
public:
    void search(Bounds &bnds);
};

#endif
