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

struct Loc {
   double x,y;
   friend std::ostream& operator<<(std::ostream& os,const Loc& l) {
      return os << "(" << l.x << ',' << l.y << ")";
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

struct Instance {
   size_t nv;
   std::vector<Loc> loc;
   Matrix<double,2>  d;
   Instance() {}
   GNSet vertices() { return GNSet(0,nv-1);} // 0-based indexing into distance matrix (vertices are 0..nv-1)
   void makeDist() {
      std::cout << d << "\n";
   }
};

Instance readFile(const char* fName)
{
   Instance rv;
   using namespace std;
   ifstream f(fName);
   string label,colon,name,comment,type,edgeWeight;
   f >> label;
   if (label[label.length()-1]==':') {
      std::getline(f,name);
      f >> label;
      std::getline(f,type);
      f >> label;      
      std::getline(f,comment);
      f >> label >> rv.nv;
      do {
         std::string back;
         f >> label;
         std::getline(f,back);
      } while (label != "EDGE_WEIGHT_SECTION");
   } else {
      f >> colon >> name;
      f >> label >> colon;
      std::getline(f,comment);
      f >> label >> colon;
      std::getline(f,type);
      f >> label >> colon >> rv.nv;
      f >> label >> colon;
      std::getline(f,edgeWeight);      
      f >> label;
   }
   assert(label == "EDGE_WEIGHT_SECTION");
   rv.d = Matrix<double,2>(rv.nv,rv.nv);
   for(auto i = 0u;i < rv.nv;++i)
      for(auto j=0u;j < rv.nv;++j) {
         int val;
         f >> val;
         rv.d[i][j] = val;
      }   
   f.close();
   std::cout << rv.d << "\n";
   return rv;
}

int main(int argc,char* argv[]) {
   if (argc < 3) {
      std::cout << "usage: tsp <file> <width>\n";
      exit(1);
   }
   const char* fName = argv[1];
   std::cout << "FILE:" << fName << "\n";
   const int w = argc==3 ? atoi(argv[2]) : 64;
   Instance instance = readFile(fName);
   auto C = instance.vertices();
   auto& d = instance.d; 
   std::cout << "Cities:" << C << "\n";
   Bounds bnds;
   const int sz = (int)C.size();
   const auto init = []()               { return TSP { GNSet{},1,0 };};
   const auto target = [sz,&C]()        { return TSP { C,1,sz };};
   const auto lgf = [&C](const TSP& s)  { return (C - s.s) | GNSet {1};};
   const auto stf = [sz,&C](const TSP& s,const int label) -> std::optional<TSP> {
      bool bad = (label == 1 && s.hops < sz-1) || (label != 1 && s.hops >= sz-1);
      if (bad)
         return std::nullopt;
      else {
         bool close = label==1 && s.hops >= sz-1;
         if (close) return TSP { C,1,sz};
         else if (!s.s.contains(label) && s.e != label) {
            return TSP { s.s | GNSet{label},label,s.hops + 1};
         } else return std::nullopt;
      }      
   };
   const auto scf = [&d](const TSP& s,int label) { // partial cost function 
      return d[s.e][label];
   };
   const auto smf = [](const TSP& s1,const TSP& s2) -> std::optional<TSP> {
      if (s1.e == s2.e && s1.hops == s2.hops) 
         return TSP {s1.s & s2.s , s1.e, s1.hops};
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

