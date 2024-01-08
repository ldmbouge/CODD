#include "dd.hpp"
#include "util.hpp"
#include <iostream>
#include <set>
#include <unordered_set>
#include <optional>

struct MISP {
   std::set<int> sel;
   int           e;
   friend std::ostream& operator<<(std::ostream& os,const MISP& m);
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
      std::size_t ttl = 0;
      for(auto e : v.sel)
         ttl = (ttl << 2) | std::hash<int>{}(e);
      return (ttl << 32) | std::hash<int>{}(v.e);
   }
};

MISP stInit()
{
   std::set<int> U;
   for(int i=0;i < 10;i++)
      U.insert(i);
   return MISP { U, -1 };
}
MISP stTarget()
{
   std::set<int> U;
   return MISP { U, -1 };
}

std::optional<MISP> smf(const MISP& s1,const MISP& s2)
{
   if (s1.e == s2.e)
      return MISP {s1.sel & s2.sel,s1.e};
   else return std::nullopt;
}

std::ostream& operator<<(std::ostream& os,const MISP& m)
{
   os << "<{ ";
   for(int v : m.sel)
      os << v << ',';
   return os << "\b}," << m.e << ">";
}

struct GE {
   int a,b;
   friend bool operator==(const GE& e1,const GE& e2) {
      return e1.a == e2.a && e1.b == e2.b;
   }
};

template<> struct std::hash<GE> {
   std::size_t operator()(const GE& v) const noexcept {
      return (std::hash<unsigned>{}(v.a) << 8) + std::hash<unsigned>{}(v.b);
   }
};

int main()
{
   const std::set<int> ns = {1,2,3,4,5};
   constexpr const int top = 6; // ns.size() + 1;
   const std::unordered_set<GE> es = { GE {1,2}, GE {1,3},
                                       GE {2,1}, GE {2,3}, GE {2,4},
                                       GE {3,1}, GE {3,2}, GE {3,4},
                                       GE {4,2}, GE {4,3}, GE {4,5},
                                       GE {5,4}
   };
   const auto labels = ns | std::set<int> { top }; 
   constexpr const int weight[] = {0,3,4,2,2,7};
   std::vector<std::set<int>> neighbors(ns.size());
   for(int i : ns) {
      neighbors.push_back(std::set<int> {});
      std::copy_if(begin(ns),end(ns),std::inserter(neighbors[i],neighbors[i].begin()),[i,&es](auto j) {
         const GE e1 {i,j},e2 {j,i};
         return std::find_if(es.begin(),es.end(),[e1,e2](auto e) { return e == e1 || e == e2;}) != es.end();
      });
      neighbors[i].insert(i);
      std::cout << i << " -> " << neighbors[i] << std::endl;
   }
   constexpr auto myInit = []() {
      std::set<int> U;
      for(int i=0;i < top;i++)
         U.insert(i);
      return MISP { U, -1 };      
   };
   auto myStf = [&neighbors](const MISP& s,int label) -> std::optional<MISP> {
      if (label == top)
         return MISP { std::set<int> {}, label};
      else if (s.sel.contains(label)) {
         std::set<int> ns = s.sel;         
         for(int i=1;i< label;i++)
            ns.erase(i);
         ns.erase(neighbors[label].begin(),neighbors[label].end());
         //auto ns = remove(s.sel,label);         
         return MISP {ns,label};
      } else return std::nullopt;
   };
   constexpr auto scf = [](const MISP& s,int label) {
      return (label == top) ? 0 : weight[label];
   };
   std::cout << "LABELS:" << labels << "\n";

   Pool mine;
   auto myDD = DD<MISP,
                  MISP(*)(),
                  MISP(*)(),
                  decltype(myStf),
                  decltype(scf)>::makeDD(&mine,myInit,stTarget,myStf,scf,smf);
   myDD->doIt();
   return 0;
}

