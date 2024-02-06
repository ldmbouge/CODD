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
typedef NatSet<2> GRSet;  // 2 double-word (128 labels)


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
	{ 12, 85 }
   };
   
   //std::cout << "OPT[3] = "<< OPT[3] << std::endl;

   Bounds bnds;
   const auto labels = setFrom(std::views::iota(1,L+1));     // using a plain set for the labels
   const auto init = []() {   // The root state      
      return SGRuler {GRSet {0},GRSet {},1,0};
   };
   const auto target = [n]() {    // The sink state
      return SGRuler {GRSet {},GRSet {},n,0};
   };
   const auto stf = [n,&bnds,&OPT,L](const SGRuler& s,const int label) -> std::optional<SGRuler> {
      if (s.k >= n || label <= s.e) return std::nullopt; // test earlier to avoid expensive tests
      //if (label < s.k * (s.k-1)/2) return  std::nullopt; // this should help, but it changes branching (more nodes)a
      if (label >= bnds.getPrimal()) return std::nullopt;
      if (label + OPT[n-s.k] >= bnds.getPrimal()) return std::nullopt;
      if (label + OPT[n-s.k] >= L+1) {	 
         /*
           //std::cout << "pruned suboptimal transition due to sub-ruler length" << std::endl;
           std::cout << "label = " << label
                     << ", n = " << n
                     << ", k = " << s.k
                     << ", OPT[" << n-s.k << "] = " << OPT[n-s.k]
                     << " >= L+1 = " << L+1 << std::endl;
         */
         return std::nullopt; // cannot improve
      }
      int illegal = 0;
      GRSet ad {};
      for(auto i : s.m) {
         ad.insert(label - i);
         illegal += s.d.contains(label- i);
         if (illegal) break;
      }
      ad.unionWith(s.d);
      if (illegal == 0) {
         if (s.k == n-1) // this must be a legal move (illegal==0)
            return SGRuler { GRSet {},GRSet {},n,0};
         else return SGRuler { s.m | GRSet {label},ad, s.k + 1,label };
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
                decltype(stf),
                decltype(scf),
                decltype(smf),
                decltype(sEq)                
                >::makeDD(init,target,stf,scf,smf,sEq,labels),w);
   engine.search(bnds);
   return 0;
}

