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

//typedef std::set<int> GRSet;
typedef NatSet<2> GRSet;


struct SGRuler {
   GRSet m; // set of marks
   GRSet d; // set of distances
   int k;           // number of marks made
   int e;           // last mark
   friend std::ostream& operator<<(std::ostream& os,const SGRuler& m) {
      return os << "<" << m.m << ',' << m.d << ',' << m.k << ',' << m.e << ">";
   }
};

template<> struct std::equal_to<SGRuler> {
   constexpr bool operator()(const SGRuler& s1,const SGRuler& s2) const {
      return s1.k == s2.k && s1.e==s2.e && s1.m == s2.m;
   }
};

template<> struct std::not_equal_to<SGRuler> {
   constexpr bool operator()(const SGRuler& s1,const SGRuler& s2) const {
      return s1.k != s2.k || s1.e != s2.e || s1.m != s2.m;
   }
};

template<> struct std::hash<SGRuler> {
   std::size_t operator()(const SGRuler& v) const noexcept {
      return (std::hash<GRSet>{}(v.m) << 24) |
         (std::hash<GRSet>{}(v.d) << 16) |
         (std::hash<int>{}(v.k) << 8) |
         std::hash<int>{}(v.e);
   }
};

int main(int argc,char* argv[])
{
   if (argc < 3) {
      std::cout << "Usage gruler <#marks> <#ubLen>\n";
      exit(1);
   }
   const int n = atoi(argv[1]);
   const int L = atoi(argv[2]);
   
   const auto labels = setFrom(std::views::iota(1,L+1));     // using a plain set for the labels
   const auto init = []() {   // The root state      
      return SGRuler {GRSet {0},GRSet {},1,0};
   };
   const auto target = [n]() {    // The sink state
      return SGRuler {GRSet {},GRSet {},n,0};
   };
   const auto stf = [n](const SGRuler& s,const int label) -> std::optional<SGRuler> {
      int illegal = 0;
      GRSet ad {};
      for(auto i : s.m) {
         ad.insert(label - i);
         illegal += s.d.contains(label-i);
      }
      if (s.k < n && label > s.e && illegal == 0) {
         if (s.k == n-1) // illegal is necessarily == 0 (false)
            return SGRuler { GRSet {},GRSet {},s.k+1,0};
         else return SGRuler { s.m | GRSet {label},s.d | ad, s.k + 1,label };
      } else return std::nullopt;  // return the empty optional 
   };
   const auto scf = [](const SGRuler& s,int label) { // partial cost function 
      return label - s.e;
   };
   const auto smf = [](const SGRuler& s1,const SGRuler& s2) -> std::optional<SGRuler> {
      if (s1.k == s2.k && s1.e == s2.e) {
         return SGRuler { s1.m & s2.m,s1.d & s2.d, s1.k , std::min(s1.e,s2.e) }; 
      }
      else return std::nullopt; // return  the empty optional
   };

   std::cout << "LABELS:" << labels << "\n";

   BAndB engine(DD<SGRuler,std::less<double>, // to minimize
                ///decltype(init), 
                decltype(target), 
                decltype(stf),
                decltype(scf),
                decltype(smf)
                >::makeDD(init,target,stf,scf,smf,labels),4);
   engine.search();
   return 0;
}

