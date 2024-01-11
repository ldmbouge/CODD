#include "dd.hpp"
#include "util.hpp"
#include <iostream>
#include <set>
#include <unordered_set>
#include <optional>

struct MISP {
   std::set<int> sel;
   int           e;
   friend std::ostream& operator<<(std::ostream& os,const MISP& m) {
      return os << "<" << m.sel << "," << m.e << ">";
   }
};

template<> struct std::equal_to<MISP> {
   constexpr bool operator()(const MISP& s1,const MISP& s2) const {
      return s1.sel == s2.sel && s1.e == s2.e;
   }
};

template<> struct std::not_equal_to<MISP> {
   constexpr bool operator()(const MISP& s1,const MISP& s2) const {
      return s1.sel != s2.sel || s1.e != s2.e;
   }
};

template<> struct std::hash<MISP> {
   std::size_t operator()(const MISP& v) const noexcept {
      return (std::hash<std::set<int>>{}(v.sel) << 32) | std::hash<int>{}(v.e);
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
   const std::set<int> ns = {1,2,3,4,5};
   const int top = ns.size() + 1;
   const std::vector<GE> es = { GE {1,2}, GE {1,3},
                                GE {2,1}, GE {2,3}, GE {2,4},
                                GE {3,1}, GE {3,2}, GE {3,4},
                                GE {4,2}, GE {4,3}, GE {4,5},
                                GE {5,4}
   };
   const auto labels = ns | std::set<int> { top }; 
   constexpr const double weight[] = {0,3,4,2,2,7};
   std::vector<std::set<int>> neighbors(ns.size());
   for(int i : ns) {
      neighbors.push_back(std::set<int> {});
      std::copy_if(begin(ns),end(ns),std::inserter(neighbors[i],neighbors[i].begin()),[i,&es](auto j) {
         const GE e1 {i,j},e2 {j,i};
         return std::find_if(es.begin(),es.end(),
                             [e1,e2](auto e) { return e == e1 || e == e2;}) != es.end();
      });
      neighbors[i].insert(i);
      std::cout << i << " -> " << neighbors[i] << std::endl;
   }
   const auto myInit = [top]() {
      std::set<int> U {};
      for(int i=1;i < top;i++)
         U.insert(i);
      return MISP { U, -1 };
   };
   const auto myTarget = []() {
      return MISP { std::set<int> {}, -1 };
   };
   auto myStf = [top,&neighbors](const MISP& s,int label) -> std::optional<MISP> {
      if (label == top)
         return MISP { std::set<int> {}, -1};
      else if (s.sel.contains(label)) {
         std::set<int> ns = s.sel;         
         for(int i=1;i< label;i++)
            ns.erase(i);
         for(auto ngh : neighbors[label])
            ns.erase(ngh);
         return MISP {ns,(ns.size()==0) ? -1 : label};
      } else return std::nullopt;
   };
   const auto scf = [top](const MISP& s,int label) {
      return (label == top) ? 0 : weight[label];
   };
   const auto smf = [](const MISP& s1,const MISP& s2) -> std::optional<MISP> {
      if (s1.e == s2.e)
         return MISP {s1.sel & s2.sel,s1.e};
      else return std::nullopt;
   };

   std::cout << "LABELS:" << labels << "\n";

   Pool mine;
   auto myxDD = DD<MISP,
                   std::greater<double>, // to maximize
                   decltype(myInit), 
                   decltype(myTarget), // MISP(*)(),
                   decltype(myStf),
                   decltype(scf),
                   decltype(smf)                   
                   >::makeDD(&mine,myInit,myTarget,myStf,scf,smf,labels);
   myxDD->setStrategy(new Exact(myxDD));
   myxDD->compute();

   std::cout << "Stage 2\n"; 
   auto myrDD = DD<MISP,
                   std::greater<double>, // to maximize
                   decltype(myInit), 
                   decltype(myTarget), // MISP(*)(),
                   decltype(myStf),
                   decltype(scf),
                   decltype(smf)
                   >::makeDD(&mine,myInit,myTarget,myStf,scf,smf,labels);
   myrDD->setStrategy(new Restricted(myrDD,2));
   myrDD->compute();

   return 0;
}

