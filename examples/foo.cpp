#include "dd.hpp"
#include "util.hpp"
#include <concepts>
#include <iostream>
#include <set>
#include <optional>
#include <ranges>
#include <algorithm>
#include <map>

struct MISP {
   std::set<int> sel;
   friend std::ostream& operator<<(std::ostream& os,const MISP& m) {
      return os << "<" << m.sel << ">";
   }
};

template<> struct std::equal_to<MISP> {
   constexpr bool operator()(const MISP& s1,const MISP& s2) const {
      return s1.sel == s2.sel;
   }
};

template<> struct std::not_equal_to<MISP> {
   constexpr bool operator()(const MISP& s1,const MISP& s2) const {
      return s1.sel != s2.sel;
   }
};

template<> struct std::hash<MISP> {
   std::size_t operator()(const MISP& v) const noexcept {
      return std::hash<std::set<int>>{}(v.sel);
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
   const std::set<int> ns = {1,2,3,4,5};
   const int top = ns.size() + 1;
   const std::vector<GE> es = { GE {1,2}, GE {1,3},
                                GE {2,1}, GE {2,3}, GE {2,4},
                                GE {3,1}, GE {3,2}, GE {3,4},
                                GE {4,2}, GE {4,3}, GE {4,5},
                                GE {5,4}
   };
   const auto labels = ns | std::set<int> { top };     // using a plain set for the labels
   constexpr const double weight[] = {0,3,4,2,2,7};    // plain array for the weights
   std::map<int,std::set<int>> neighbors {};  // computing the neigbhors using STL (not pretty)
   for(int i : ns) {
      neighbors[i] = filter(ns,[i,&es](auto j) {
         return j==i || member(es,[e1=GE {i,j},e2=GE {j,i}](auto e) { return e==e1 || e==e2;});
      });
      std::cout << i << " -> " << neighbors[i] << std::endl;
   }
   const auto myInit = [top]() {   // The root state
      std::set<int> U = {}; // std::views::iota(1,top) | std::ranges::to<std::set>();
      for(auto i : std::views::iota(1,top))
         U.insert(i);
      return MISP { U };
   };
   const auto myTarget = []() {    // The sink state
      return MISP { std::set<int> {} };
   };
   auto myStf = [top,&neighbors](const MISP& s,const int label) -> std::optional<MISP> {
      if (label == top)
         return MISP { std::set<int> {}}; // head to sink
      else if (s.sel.contains(label)) {
         return MISP {
            filter(s.sel,[label,nl = neighbors[label]](int i) {
               return i >= label && !nl.contains(i);
            })
         };
      } else return std::nullopt;  // return the empty optional 
   };
   const auto scf = [top](const MISP& s,int label) { // cost function 
      return (label == top) ? 0 : weight[label];
   };
   const auto smf = [](const MISP& s1,const MISP& s2) -> std::optional<MISP> { // merge function
      if (std::max(s1.sel) == std::max(s2.sel))
         return MISP {s1.sel & s2.sel};
      else return std::nullopt; // return  the empty optional
   };

   std::cout << "LABELS:" << labels << "\n";

   std::cout << "exact\n"; 
   auto myxDD = DD<MISP,
                   std::greater<double>, // to maximize
                   decltype(myInit), 
                   decltype(myTarget), // MISP(*)(),
                   decltype(myStf),
                   decltype(scf),
                   decltype(smf)                   
                   >::makeDD(myInit,myTarget,myStf,scf,smf,labels);
   myxDD->setStrategy(new Exact);
   myxDD->compute();

   std::cout << "restricted\n"; 
   auto myrDD = DD<MISP,
                   std::greater<double>, // to maximize
                   decltype(myInit), 
                   decltype(myTarget), // MISP(*)(),
                   decltype(myStf),
                   decltype(scf),
                   decltype(smf)
                   >::makeDD(myInit,myTarget,myStf,scf,smf,labels);
   myrDD->setStrategy(new Restricted(1));
   myrDD->compute();

   std::cout << "relaxed\n"; 
   auto mylDD = DD<MISP,
                   std::greater<double>, // to maximize
                   decltype(myInit), 
                   decltype(myTarget), // MISP(*)(),
                   decltype(myStf),
                   decltype(scf),
                   decltype(smf)
                   >::makeDD(myInit,myTarget,myStf,scf,smf,labels);
   mylDD->setStrategy(new Relaxed(1));
   mylDD->compute();

   return 0;
}

