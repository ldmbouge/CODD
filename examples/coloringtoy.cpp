#include "dd.hpp"
#include "util.hpp"
#include "search.hpp"
#include <concepts>
#include <iostream>
#include <set>
#include <optional>
#include <ranges>
#include <algorithm>
#include <map>

typedef std::vector<std::set<int>> Legal;

struct COLOR {
   Legal s;
   int last;
   int vtx;
   friend std::ostream& operator<<(std::ostream& os,const COLOR& m) {
      return os << "<" << m.s << ',' << m.last << ',' << m.vtx << ">";
   }
};

template<> struct std::equal_to<COLOR> {
   constexpr bool operator()(const COLOR& s1,const COLOR& s2) const {
      return s1.last == s2.last && s1.vtx==s2.vtx && s1.s == s2.s;
   }
};

template<> struct std::not_equal_to<COLOR> {
   constexpr bool operator()(const COLOR& s1,const COLOR& s2) const {
      return s1.last != s2.last || s1.vtx != s2.vtx || s1.s != s2.s;
   }
};

template<> struct std::hash<COLOR> {
   std::size_t operator()(const COLOR& v) const noexcept {
      return (std::hash<Legal>{}(v.s) << 24) |
         (std::hash<int>{}(v.last) << 16) |
         std::hash<int>{}(v.vtx);
   }
};

struct GE {
   int a,b;
   friend bool operator==(const GE& e1,const GE& e2) {
      return e1.a == e2.a && e1.b == e2.b;
   }
   friend bool operator<(const GE& e1,const GE& e2) {
      return e1.a < e2.a || (e1.a == e2.a && e1.b < e2.b);
   }
};

template<class T,class B>
std::set<T> setFrom(const std::ranges::iota_view<T,B>& from) {
   std::set<T> res {};
   for(auto v : from)
      res.insert(v);
   return res;
}

int main()
{
   // using STL containers for the graph
   const std::set<int> ns = {0,1,2,3,4};//,5,6,7,8,9,10,11,12,13,14,15};
   const std::set<GE> es = {GE {0,3},
                            GE {0,4},
                            GE {1,2},
                            GE {1,3},
                            GE {1,4},                            
                            GE {2,4}                             
   };
   const int K = ns.size();
   const auto labels = setFrom(std::views::iota(1,K+1));     // using a plain set for the labels
   const auto init = [ns,labels]() {   // The root state
      Legal A(ns.size(), std::set<int>{});      
      for(auto v : ns)
         A[v] = labels;
      //A[0] = std::set<int>{1};
      return COLOR {A,0,0 };
   };
   const auto target = [K]() {    // The sink state
      return COLOR { Legal{},0,K};
   };
   const auto stf = [K,es](const COLOR& s,const int label) -> std::optional<COLOR> {
      if (s.vtx < K && s.s[s.vtx].contains(label) && label <= s.last+1) {
         Legal B = s.s;
         for(auto vIdx = s.vtx+1;vIdx < (int)B.size();vIdx++)
            if ((es.contains(GE {s.vtx,vIdx}) || es.contains(GE {vIdx,s.vtx})) &&
                s.s[vIdx].contains(label))
               B[vIdx].erase(label);
         B[s.vtx] = std::set<int>{label};
         if (s.vtx+1 == K)
            return COLOR { Legal {},0, K };
         else
            return COLOR {B, std::max(label,s.last), s.vtx + 1};
      } 
      else return std::nullopt;  // return the empty optional 
   };
   const auto scf = [](const COLOR& s,int label) { // partial cost function 
      return std::max(0,label -  s.last);
   };
   const auto smf = [K](const COLOR& s1,const COLOR& s2) -> std::optional<COLOR> {
      if (s1.last == s2.last && s1.vtx == s2.vtx) {
         Legal B(s1.s.size(),std::set<int>{});
         for(auto i=0;i < K;i++) 
            B[i] = s1.s[i] | s2.s[i];         
         return COLOR {B , s1.last, s1.vtx};
      }
      else return std::nullopt; // return  the empty optional
   };

   std::cout << "LABELS:" << labels << "\n";

   BAndB engine(DD<COLOR,std::less<double>, // to minimize
                ///decltype(init), 
                decltype(target), 
                decltype(stf),
                decltype(scf),
                decltype(smf)
                >::makeDD(init,target,stf,scf,smf,labels),1);
   engine.search();
   return 0;
}

