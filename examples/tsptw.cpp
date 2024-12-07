#include "codd.hpp"
#include "heap.hpp"

struct TSPTW {
   GNSet  A; // locations on _all_ paths into state
   int e;
   int hops;
   friend std::ostream& operator<<(std::ostream& os,const TSPTW& m) {
      return os << "<" << m.A << ',' << m.e << ',' << m.hops << ">";
   }
};

template<> struct std::equal_to<TSPTW> {
   constexpr bool operator()(const TSPTW& s1,const TSPTW& s2) const {
      return s1.e == s2.e && s1.hops==s2.hops && s1.A == s2.A;
   }
};

template<> struct std::hash<TSPTW> {
   std::size_t operator()(const TSPTW& v) const noexcept {
      return (std::hash<GNSet>{}(v.A) << 32) |  // check if this is OK
         (std::hash<int>{}(v.e) << 16) |
         std::hash<int>{}(v.hops);
   }
};

struct TimeWindow {
   int a,b;
   friend std::ostream& operator<<(std::ostream& os,const TimeWindow& t) {
      return os << '<' << t.a << ',' << t.b << "> ";
   }
};

struct Instance {
   size_t nv;
   Matrix<double,2>   d;
   std::vector<TimeWindow> twin;
   Instance() {}
   GNSet vertices() { return GNSet(0,nv-1);} 
   void makeDist() {
      std::cout << d << "\n";
   }
};

void skip(std::ifstream& f) {
   char ch = 0;
   std::string comment;
   do {
      ch = f.get();
      if (ch == '#') {
         std::getline(f,comment);
         ch = ' ';
      }
   } while (isspace(ch) && !f.eof());
   if (!f.eof())
      f.unget();
}

Instance readFile(const char* fName)
{
   Instance rv;
   using namespace std;
   ifstream f(fName);
   skip(f);
   f >> rv.nv;
   rv.d = Matrix<double,2>(rv.nv,rv.nv);
   rv.twin = vector<TimeWindow>(rv.nv);
   skip(f);
   for(auto i=0u;i< rv.nv;i++) {
      for(auto j=0u;j< rv.nv;j++) {
         double d;
         f >> d;
         rv.d[i][j] = d;
      }
   }
   skip(f);
   for(auto i=0u;i< rv.nv;i++) {
      int a,b;
      f >> a >> b;
      rv.twin[i] = TimeWindow { a , b};
   }
   f.close();
   std::cout << rv.d << "\n";
   std::cout << rv.twin << "\n";
   return rv;
}


double greedy(Matrix<double,2>& d,GNSet C,GNSet V,int src,int sink,int hops)
{
   // compute lower bound as the sum of the cheapest arc out of each 
   // node except the src (which already has an outgoing arc per the
   // partial solution so far. 
   GNSet t = (C - V).insert(src).insert(sink);
   const auto ts = C.size() - hops; // Beware: set V is an lower bound, it could be too small, its true
   // size is hops. So only pick the ts shortest edges at the end.
   int ne = 0;
   double edge[t.size()];
   for (auto i : t) 
      if (i != src)
         edge[ne++] = min(t,[i](int j) {return i!=j;},[i,&d](int j) { return d[i][j];});
   
   mergeSort(edge,ne,[](double a,double b) { return a < b;});
   return sum(Range(0,ts),[&edge](int e) { return edge[e];});
}

int main(int argc,char* argv[]) {
   if (argc < 3) {
      std::cout << "usage: tsptw <file> <width>\n";
      exit(1);
   }
   const char* fName = argv[1];
   std::cout << "FILE:" << fName << "\n";
   const int w = argc==3 ? atoi(argv[2]) : 64;
   Instance instance = readFile(fName);
   auto C = instance.vertices();
   auto& d = instance.d;
   auto& tw = instance.twin;
   std::cout << "Cities:" << C << "\n";
   Bounds bnds([](const std::vector<int>& inc)  {
   });
   const int depot = 0;
   const int sz = (int)C.size();
   const auto init = []()               { return TSPTW { GNSet{depot},depot,0};};
   const auto target = [sz,&C]()        { return TSPTW { C,depot,sz };}; // the optimal bound is unknown (put 0 here)
   const auto lgf = [sz,&C](const TSPTW& s)  {
      if (s.hops >= sz-1)
         return GNSet {depot};
      else 
         return (C - s.A);      
   };
   const auto stf = [sz,&C](const TSPTW& s,const int label) -> std::optional<TSPTW> {
      if (label==depot)
         return TSPTW { C,depot,sz}; //0 is the dummy value for the lb 
      else 
         return TSPTW { s.A | GNSet{label},label,s.hops + 1};
   };
   const auto scf = [&d](const TSPTW& s,int label) { // partial cost function 
      return d[s.e][label];
   };
   const auto smf = [](const TSPTW& s1,const TSPTW& s2) -> std::optional<TSPTW> {
      if (s1.e == s2.e && s1.hops == s2.hops) 
         return TSPTW {s1.A & s2.A, s1.e, s1.hops};
      else return std::nullopt; // return  the empty optional
   };
   const auto eqs = [sz](const TSPTW& s) -> bool { return s.e == depot && s.hops == sz;};
   const auto local = [&d,&C](const TSPTW& s,LocalContext) -> double {
      const auto greedyVal = greedy(d,C,s.A,depot,s.e,s.hops);
      return greedyVal;
   };   

   BAndB engine(DD<TSPTW,Minimize<double>,
                decltype(target),
                decltype(lgf),
                decltype(stf),
                decltype(scf),
                decltype(smf),
                decltype(eqs),
                decltype(local)
                >::makeDD(init,target,lgf,stf,scf,smf,eqs,C,local),w);
   engine.search(bnds);
   std::cout << "INSTANCE READING ONLY. About to write model ;-)" << "\n";
   return 0;
}

