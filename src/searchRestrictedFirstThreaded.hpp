#ifndef __SEARCH_RESTRICTED_FIRST_THREADED_HPP__
#define __SEARCH_RESTRICTED_FIRST_THREADED_HPP__

#include <functional>
#include "dd.hpp"
#include "store.hpp"
#include "search.hpp"

class BAndBRestrictedFirstThreaded: public BAndB {
    using BAndB::BAndB;
public:
    void search(Bounds &bnds);
};

#endif
