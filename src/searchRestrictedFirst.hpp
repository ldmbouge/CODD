#ifndef __SEARCH_RESTRICTED_FIRST_HPP__
#define __SEARCH_RESTRICTED_FIRST_HPP__

#include "dd.hpp"
#include "search.hpp"

class BAndBRestrictedFirst: public BAndB {
    using BAndB::BAndB;
public:
    void search(Bounds &bnds);
};

#endif
