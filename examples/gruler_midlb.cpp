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
#include <math.h>

//typedef std::set<int> GRSet;
typedef NatSet<4> GRSet;  // 4 double-word (256 labels)


struct SGRuler {
   GRSet m; // set of marks
   GRSet d; // set of distances
   int k;   // number of marks made
   int e;   // last mark
   int sm;  // smallest unused distance
   friend std::ostream& operator<<(std::ostream& os,const SGRuler& m) {
      return os << "<" << m.m << ',' << m.d << ',' << m.k << ',' << m.e << ',' << m.sm << ">";
   }
};

template<> struct std::equal_to<SGRuler> {
   constexpr bool operator()(const SGRuler& s1,const SGRuler& s2) const {
      return s1.k == s2.k && s1.e==s2.e && s1.m == s2.m;
      // no need to check s1.sm == s2.sm because this is implied by s1.m == s2.m
   }
};

template<> struct std::hash<SGRuler> {
   std::size_t operator()(const SGRuler& v) const noexcept {
      return std::rotl(std::hash<GRSet>{}(v.m),20) |
         std::rotl(std::hash<GRSet>{}(v.d),10)  | 
         std::rotl(std::hash<int>{}(v.k),8) |
         std::hash<int>{}(v.e) |
         std::hash<int>{}(v.sm);
   }
};

int main(int argc,char* argv[])
{
   if (argc < 3) {
      std::cout << "Usage gruler <#marks> <#ubLen> <maxWidth>\n";
      exit(1);
   }
   const int n = atoi(argv[1]);
   const int L = atoi(argv[2]);
   const int w = argc==4 ? atoi(argv[3]) : 64;

   std::map<int, int> OPT = {
	{ 0, 0 },
	{ 1, 0 },
	{ 2, 1 },
	{ 3, 3 },
	{ 4, 6 },
	{ 5, 11 },
	{ 6, 17 },
	{ 7, 25 },
	{ 8, 34 },
	{ 9, 44 },
	{ 10, 55 },
	{ 11, 72 },
	{ 12, 85 },
	{ 13, 106 },
	{ 14, 127 },
	{ 15, 151 },
	{ 16, 177 },
	{ 17, 199 },
	{ 18, 216 },
	{ 19, 246 }
   };
   
   //std::cout << "OPT[3] = "<< OPT[3] << std::endl;

   Bounds bnds;
   const auto labels = setFrom(std::views::iota(1,L+1));     // using a plain set for the labels
   const auto init = []() {   // The root state      
      return SGRuler {GRSet {0},GRSet {},1,0,1};
   };
   const auto target = [n,L]() {    // The sink state
      return SGRuler {GRSet {},GRSet {},n,0,L+1};  // smallest unused distance should be set explicitly to L+1
   };
   const auto lgf = [n,&bnds,&OPT,L](const SGRuler& s) -> GNSet {
      auto ub = L+1;
      if (s.k <= n/2)
        ub = std::min({(int)std::floor(((int)bnds.getPrimal()-1)/2) - OPT[std::floor(n/2)-s.k], (int)std::floor((L+1)/2) - OPT[std::floor(n/2)-s.k]});
      else 
        ub = std::min({(int)bnds.getPrimal() - OPT[n-s.k] - 1,L+1 - OPT[n-s.k]});
      auto lb = std::max({s.e+s.sm,(int)std::ceil(s.k * (s.k -1)/2)});  // added s.sm (replacing 1) to improve lb
      //if (s.sm != 1) std::cout << "smallest unused distance is " << s.sm << std::endl;
      return GNSet(lb,ub);
   };
   const auto stf = [n,L](const SGRuler& s,const int label) -> std::optional<SGRuler> {
      bool legal = !std::foldl(s.m,[label,&s](bool acc,int i) { return acc || s.d.contains(label - i);},false);
      if (legal) { // this must be a legal move (illegal==0)
         if (s.k == n-1)  // this moves goes to the sink
            return SGRuler { GRSet {},GRSet {},n,0,L+1};
         else { 
	    GRSet d_new = (label - s.m) | s.d;
	    int smallest_dist = s.sm;
	    while (d_new.contains(smallest_dist)) smallest_dist += 1;
            return SGRuler { s.m | GRSet {label}, d_new, s.k + 1, label, smallest_dist };         
	 }
      } else return std::nullopt;  // return the empty optional 
   };
   const auto scf = [](const SGRuler& s,int label) { // partial cost function 
      return label - s.e;
   };
   const auto smf = [](const SGRuler& s1,const SGRuler& s2) -> std::optional<SGRuler> {
      if (s1.k == s2.k && s1.e == s2.e) {
         return SGRuler { s1.m & s2.m,s1.d & s2.d, s1.k , std::min(s1.e,s2.e) , std::min(s1.sm,s2.sm)}; 
      }
      else return std::nullopt; // return  the empty optional
   };
   const auto sEq = [n](const SGRuler& s) -> bool {
      return s.k == n;
   };

   std::cout << "LABELS:" << labels << "\n";

   
   /*
     auto myxDD = DD<SGRuler,std::less<double>, // to minimize
                ///decltype(init), 
                decltype(target), 
                decltype(stf),
                decltype(scf),
                decltype(smf)
                >::makeDD(init,target,stf,scf,smf,labels);
   myxDD->setStrategy(new Exact);
   bnds.setPrimal(1000);
   myxDD->compute();

   std::cout << myxDD->incumbent() << std::endl;
   return 0;
   */
   
   BAndB engine(DD<SGRuler,std::less<double>, // to minimize
                ///decltype(init), 
                decltype(target),
                decltype(lgf),
                decltype(stf),
                decltype(scf),
                decltype(smf),
                decltype(sEq)                
                >::makeDD(init,target,lgf,stf,scf,smf,sEq,labels),w);
   engine.search(bnds);
   return 0;
}

