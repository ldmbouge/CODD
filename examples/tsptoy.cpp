#include "codd.hpp"

struct TSP {
   GNSet  s;
   int e;
   int hops;
   friend std::ostream& operator<<(std::ostream& os,const TSP& m) {
      return os << "<" << m.s << ',' << m.e << ',' << m.hops << ">";
   }
};

template<> struct std::equal_to<TSP> {
   constexpr bool operator()(const TSP& s1,const TSP& s2) const {
      return s1.e == s2.e && s1.hops==s2.hops && s1.s == s2.s;
   }
};

template<> struct std::hash<TSP> {
   std::size_t operator()(const TSP& v) const noexcept {
      return (std::hash<GNSet>{}(v.s) << 24) |
         (std::hash<int>{}(v.e) << 16) |
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

int main(int argc,char* argv[]) {
   if (argc < 1) {
      std::cout << "usage: tsptoy2 <width>\n";
      exit(1);
   }
   const int w = argc==2 ? atoi(argv[1]) : 64;
   const GNSet C = {1,2,3,4,5,6}; // ,6,7,8,9,10,11,12,13,14,15};
   const std::map<GE,double> es = {
                             {GE {1,2}, 10},{GE {1,3}, 15},{GE {1,4}, 20},{GE {1,5}, 6},{GE {1,6}, 20},
                             {GE {2,1}, 10},{GE {2,3}, 35},{GE {2,4}, 25},{GE {2,5}, 100},{GE {2,6}, 5},
                             {GE {3,1}, 15},{GE {3,2}, 35},{GE {3,4}, 30},{GE {3,5},100},{GE {3,6}, 100},
                             {GE {4,1}, 20},{GE {4,2}, 25},{GE {4,3}, 30},{GE {4,5}, 3},{GE {4,6}, 30},
                             {GE {5,1},  6},{GE {5,2},100},{GE {5,3},100},{GE {5,4}, 3},{GE {5,6}, 100},
                             {GE {6,1}, 20},{GE {6,2},5}  ,{GE {6,3},100},{GE {6,4},30},{GE {6,5}, 100}
   };
   Bounds bnds;
   const int sz = (int)C.size();
   const auto init = []()               { return TSP { GNSet{},1,0 };}; // initial "end" is 1
   const auto target = [sz,&C]()        { return TSP { C,1,sz };};
   const auto lgf = [sz,&C](const TSP& s,DDContext)  {
      if (s.hops >= sz-1)
         return GNSet {1};
      else 
         return (C - s.s).remove(1).remove(s.e);
   };
   const auto stf = [sz,&C](const TSP& s,const int label) -> std::optional<TSP> {
      if (label==1)
         return TSP { C,1,sz};
      else 
         return TSP { s.s | GNSet{label},label,s.hops + 1};
   };
   const auto scf = [&es](const TSP& s,int label) { // partial cost function 
      return es.at(GE {s.e,label});
   };
   const auto smf = [](const TSP& s1,const TSP& s2) -> std::optional<TSP> {
      auto valid = s1.s & s2.s;
      //std::cout << "MERGE: " << s1.hops  << " " << valid.size() << " " << ((float)valid.size()/s1.hops) << "\n";
      if (s1.e == s2.e && s1.hops == s2.hops && ((float)valid.size()/s1.hops) >= .7) 
         return TSP { std::move(valid) , s1.e, s1.hops};
      else return std::nullopt; // return  the empty optional
   };
   const auto eqs = [sz](const TSP& s) -> bool { return s.e == 1 && s.hops == sz;};

   BAndB engine(DD<TSP,Minimize<double>,
                decltype(target),
                decltype(lgf),
                decltype(stf),
                decltype(scf),
                decltype(smf),
                decltype(eqs)
                >::makeDD(init,target,lgf,stf,scf,smf,eqs,C),w);
   engine.search(bnds);
   return 0;
}

