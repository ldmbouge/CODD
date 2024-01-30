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

struct TSP {
   GNSet  s;
   int last;
   int hops;
   friend std::ostream& operator<<(std::ostream& os,const TSP& m) {
      return os << "<" << m.s << ',' << m.last << ',' << m.hops << ">";
   }
};

template<> struct std::equal_to<TSP> {
   constexpr bool operator()(const TSP& s1,const TSP& s2) const {
      return s1.last == s2.last && s1.hops==s2.hops && s1.s == s2.s;
   }
};

template<> struct std::not_equal_to<TSP> {
   constexpr bool operator()(const TSP& s1,const TSP& s2) const {
      return s1.last != s2.last || s1.hops != s2.hops || s1.s != s2.s;
   }
};

template<> struct std::hash<TSP> {
   std::size_t operator()(const TSP& v) const noexcept {
      return (std::hash<GNSet>{}(v.s) << 24) |
         (std::hash<int>{}(v.last) << 16) |
         std::hash<int>{}(v.hops);
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

int main()
{
   // using STL containers for the graph
   const GNSet ns = {1,2,3,4};//,5,6,7,8,9,10,11,12,13,14,15};
   const std::map<GE,double> es = { {GE {1,2}, 10},
                             {GE {1,3}, 15},
                             {GE {1,4}, 20},
                             {GE {2,1}, 10},
                             {GE {2,3}, 35},                            
                             {GE {2,4}, 25},
                             {GE {3,1}, 15},
                             {GE {3,2}, 35},
                             {GE {3,4}, 30},
                             {GE {4,1}, 20},
                             {GE {4,2}, 25},
                             {GE {4,3}, 30}
   };
   Bounds bnds;
   const auto labels = ns;     // using a plain set for the labels
   const int sz = (int)ns.size();
   const auto init = []() {   // The root state
      return TSP { GNSet{},1,0 };
   };
   const auto target = [sz]() {    // The sink state
      return TSP { GNSet{},1,sz };
   };
   const auto stf = [sz](const TSP& s,const int label) -> std::optional<TSP> {
      if ((label==1 && s.hops < sz-1) || (s.hops == sz-1 && label!=1))
         return std::nullopt;
      if (s.hops < sz && !s.s.contains(label) && (s.last != label)) {
         if (s.hops + 1 == sz)
            return TSP { GNSet{}, label,s.hops + 1};
         else
            return TSP { s.s | GNSet{label},label,s.hops + 1}; // head to sink
      } else return std::nullopt;  // return the empty optional 
   };
   const auto scf = [&es](const TSP& s,int label) { // partial cost function 
      return es.at(GE {s.last,label});
   };
   const auto smf = [](const TSP& s1,const TSP& s2) -> std::optional<TSP> {
      if (s1.last == s2.last && s1.hops == s2.hops) 
         return TSP {s1.s & s2.s , s1.last, s1.hops};
      else return std::nullopt; // return  the empty optional
   };

   std::cout << "LABELS:" << labels << "\n";

   BAndB engine(DD<TSP,std::less<double>, // to minimize
                ///decltype(init), 
                decltype(target), 
                decltype(stf),
                decltype(scf),
                decltype(smf)
                >::makeDD(init,target,stf,scf,smf,labels),1);
   engine.search(bnds);
   return 0;
}

