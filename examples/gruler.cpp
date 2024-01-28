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

struct SGRuler {
   std::set<int> m; // set of marks
   std::set<int> d; // set of distances
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
      return (std::hash<std::set<int>>{}(v.m) << 24) |
         (std::hash<std::set<int>>{}(v.d) << 16) |
         (std::hash<int>{}(v.k) << 8) |
         std::hash<int>{}(v.e);
   }
};

template<class T,class B>
std::set<T> setFrom(const std::ranges::iota_view<T,B>& from) {
   std::set<T> res {};
   for(auto v : from)
      res.insert(v);
   return res;
}

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
      return SGRuler {std::set<int>{0},std::set<int>{},1,0};
   };
   const auto target = [n]() {    // The sink state
      return SGRuler { std::set<int>{},std::set<int>{},n,0};
   };
   const auto stf = [n](const SGRuler& s,const int label) -> std::optional<SGRuler> {
      int illegal = 0;
      std::set<int> ad {};
      for(auto i : s.m) {
         ad.insert(label - i);
         illegal += s.d.contains(label-i);
      }
      if (s.k < n && label > s.e && illegal == 0) {
         if (s.k == n-1) // illegal is necessarily == 0 (false)
            return SGRuler { std::set<int>{},std::set<int>{},s.k+1,0};
         else return SGRuler { s.m | std::set<int>{label},s.d | ad, s.k + 1,label };
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
