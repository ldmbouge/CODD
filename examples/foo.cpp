#include "dd.hpp"
#include "util.hpp"
#include <concepts>
#include <iostream>
#include <set>
#include <optional>
#include <ranges>
#include <algorithm>
#include <map>
#include "search.hpp"

struct MISP {
   GNSet sel;
   friend std::ostream& operator<<(std::ostream& os,const MISP& m) {
      return os << "<" << m.sel << ">";
   }
};

template<> struct std::equal_to<MISP> {
   bool operator()(const MISP& s1,const MISP& s2) const {
      return s1.sel == s2.sel;
   }
};

template<> struct std::hash<MISP> {
   std::size_t operator()(const MISP& v) const noexcept {
      return std::hash<GNSet>{}(v.sel);
   }
};

struct GE {
   int a,b;
   friend bool operator==(const GE& e1,const GE& e2) {
      return e1.a == e2.a && e1.b == e2.b;
   }
};

int main()
{
   // using STL containers for the graph
   const GNSet ns = {1,2,3,4,5};
   const int top = ns.size() + 1;
   const std::vector<GE> es = { GE {1,2}, GE {1,3},
                                GE {2,1}, GE {2,3}, GE {2,4},
                                GE {3,1}, GE {3,2}, GE {3,4},
                                GE {4,2}, GE {4,3}, GE {4,5},
                                GE {5,4}
   };
   Bounds bnds;
   const auto labels = ns | GNSet { top };     // using a plain set for the labels
   constexpr const double weight[] = {0,3,4,2,2,7};    // plain array for the weights
   std::map<int,GNSet> neighbors {};  // computing the neigbhors using STL (not pretty)
   for(int i : ns) {
      neighbors[i] = filter(ns,[i,&es](auto j) {
         return j==i || member(es,[e1=GE {i,j},e2=GE {j,i}](auto e) { return e==e1 || e==e2;});
      });
      std::cout << i << " -> " << neighbors[i] << std::endl;
   }
   const auto myInit = [top]() {   // The root state
      GNSet U = {}; // std::views::iota(1,top) | std::ranges::to<std::set>();
      for(auto i : std::views::iota(1,top))
         U.insert(i);
      return MISP { U };
   };
   const auto myTarget = []() {    // The sink state
      return MISP { GNSet {} };
   };
   const auto lgf = [top](const MISP& s) -> Range {
      return Range::close(1,top);
   };
   auto myStf = [top,&neighbors](const MISP& s,const int label) -> std::optional<MISP> {
      if (label == top)
         return MISP { GNSet {}}; // head to sink
      else if (s.sel.contains(label)) {
         return MISP {
            filter(s.sel,[label,nl = neighbors[label]](int i) {
               return i >= label && !nl.contains(i);
            })
         };
      } else return std::nullopt;  // return the empty optional 
   };
   const auto scf = [top,weight](const MISP& s,int label) { // cost function 
      return (label == top) ? 0 : weight[label];
   };
   const auto smf = [](const MISP& s1,const MISP& s2) -> std::optional<MISP> { // merge function
      using namespace std;
      if (max(s1.sel) == max(s2.sel))
         return MISP {s1.sel & s2.sel};
      else return std::nullopt; // return  the empty optional
   };
   const auto eqs = [](const MISP& s) -> bool {
      return s.sel.size() == 0;
   };

   std::cout << "LABELS:" << labels << "\n";

   std::cout << "exact\n"; 
   auto myxDD = DD<MISP,
                   std::greater<double>, // to maximize
                   //decltype(myInit), 
                   decltype(myTarget), // MISP(*)(),
                   decltype(lgf),
                   decltype(myStf),
                   decltype(scf),
                   decltype(smf),
                   decltype(eqs)
                   >::makeDD(myInit,myTarget,lgf,myStf,scf,smf,eqs,labels);
   myxDD->setStrategy(new Exact);
   myxDD->compute();
   //myxDD->display();
   std::cout << myxDD->incumbent() << std::endl;
   
   
   BAndB engine(DD<MISP,std::greater<double>, // to maximize
                //decltype(myInit), 
                decltype(myTarget), // MISP(*)(),
                decltype(lgf),
                decltype(myStf),
                decltype(scf),
                decltype(smf),
                decltype(eqs)                   
                >::makeDD(myInit,myTarget,lgf,myStf,scf,smf,eqs,labels),1);
   engine.search(bnds);
   return 0;
}

