#include "util.hpp"

// DDGen::Ptr makeDDGen(GNSet&& gns) {
//    return std::make_unique<DDGenGNset>(std::move(gns));
// }
DDGen::Ptr makeDDGen(const GNSet& gns) {
   return std::make_unique<DDGenGNset>(gns);
}

DDGen::Ptr makeDDGen(const Range& r) {
   return std::make_unique<DDGenRange>(r);
}