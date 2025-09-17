#ifndef __SEARCH_RELAXED_FIRST_HPP__
#define __SEARCH_RELAXED_FIRST_HPP__

#include "search.hpp"

class BAndBRelaxedFirst : public BAndB {
    using BAndB::BAndB;
public:
    void search(Bounds& bnds) override;
};  

#endif
