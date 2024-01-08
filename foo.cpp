#include "dd.hpp"
#include "util.hpp"
#include <iostream>
#include <set>


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
MISP stf(const MISP& s,int label)
{
   auto ns = remove(s.sel,label);
   return MISP {ns,(ns.size()==0) ? -1 : label};
}
MISP smf(const MISP& s1,const MISP& s2)
{
   return MISP {s1.sel | s2.sel,s1.e};
}

std::ostream& operator<<(std::ostream& os,const MISP& m)
{
   os << "<{ ";
   for(int v : m.sel)
      os << v << ',';
   return os << "\b}," << m.e << ">";
}

int main()
{
   const int top = 10;
   constexpr const int weight[] = {0,3,4,2,2,7};
   constexpr auto myInit = []() {
      std::set<int> U;
      for(int i=0;i < top;i++)
         U.insert(i);
      return MISP { U, -1 };      
   };
   constexpr auto scf = [](const MISP& s,int label) {
      if (label == top)
         return 0;
      else
         return weight[label];
   };

   Pool mine;
   auto myDD = DD<MISP,
                  MISP(*)(),
                  MISP(*)(),
                  MISP(*)(const MISP&,int),
                  decltype(scf)>::makeDD(&mine,myInit,stTarget,stf,scf,smf);
   myDD->doIt();
   return 0;
}

