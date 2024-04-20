#include "codd.hpp"

struct TSP {
   GNSet  s;
   int    e;
   friend std::ostream& operator<<(std::ostream& os,const TSP& m) {
      return os << "<" << m.s << ',' << m.e << ">";
   }
};

template<> struct std::equal_to<TSP> {
   constexpr bool operator()(const TSP& s1,const TSP& s2) const {
      return s1.e == s2.e  && s1.s == s2.s;
   }
};

template<> struct std::hash<TSP> {
   std::size_t operator()(const TSP& v) const noexcept {
      return (std::hash<GNSet>{}(v.s) << 24) | (std::hash<int>{}(v.e) << 16);
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
   const GNSet C = {1,2,3,4};//,5,6,7,8,9,10,11,12,13,14,15};
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
   const auto sz = C.size();
   const auto init = []()               { return TSP { GNSet{},1 };};
   const auto target = [&C]()           { return TSP { C,1 };};
   const auto lgf = [&C](const TSP& s)  { return (C - s.s) | GNSet {1};};
   
   const auto stf = [sz,&C](const TSP& s,const int label) -> std::optional<TSP> {
      bool bad = (label == 1 && s.s.size() < sz-1) || (label != 1 && s.s.size() >= sz-1);
      if (bad)
         return std::nullopt;
      else {
         bool close = label==1 && s.s.size() >= sz-1;
         if (close) return TSP { C,1 };
         else if (!s.s.contains(label) && s.e != label) {
            return TSP { s.s | GNSet{label},label};
         } else return std::nullopt;
      }      
   };
   const auto scf = [&es](const TSP& s,int label) { return es.at(GE {s.e,label});};
   const auto smf = [](const TSP& s1,const TSP& s2) -> std::optional<TSP> {
      if (s1.e == s2.e) 
         return TSP {s1.s & s2.s , s1.e };
      else return std::nullopt; 
   };
   const auto eqs = [](const TSP& s) -> bool { return s.e == 1;};

   BAndB engine(DD<TSP,Minimize<double>, 
                decltype(target),
                decltype(lgf),
                decltype(stf),
                decltype(scf),
                decltype(smf),
                decltype(eqs)
                >::makeDD(init,target,lgf,stf,scf,smf,eqs,C),1);
   engine.search(bnds);
   return 0;
}

