#include "codd.hpp"
#include "heap.hpp"
#include "searchRestrictedFirst.hpp"

int d2i(double d) { return (int)(d * 10000); }

struct TimeWindow {
   int a,b;
   friend std::ostream& operator<<(std::ostream& os,const TimeWindow& t) {
      return os << '<' << t.a << ',' << t.b << "> ";
   }
};

struct TSPTW {
   GNSet  U; // unvisited cities
   int    e; // current city
   int    t; // time
   int hops;
   friend std::ostream& operator<<(std::ostream& os,const TSPTW& m) {
      return os << "<" << m.U << ',' << m.e << ',' << m.t << ',' << m.hops << ">";
   }
};

template<> struct std::equal_to<TSPTW> {
   constexpr bool operator()(const TSPTW& s1,const TSPTW& s2) const {
      return s1.e == s2.e && s1.t==s2.t && s1.hops==s2.hops && s1.U == s2.U;
      //return s1.e == s2.e && s1.hops==s2.hops;
   }
};

template<> struct std::hash<TSPTW> {
   std::size_t operator()(const TSPTW& v) const noexcept {
      return (std::hash<GNSet>{}(v.U) << 32) |  // check if this is OK
         (std::hash<int>{}(v.e) << 16) |
         std::hash<int>{}(v.hops);
   }
};

struct Instance {
   size_t nv;
   Matrix<int,2>   d;
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
   rv.d = Matrix<int,2>(rv.nv,rv.nv);
   rv.twin = vector<TimeWindow>(rv.nv);
   skip(f);
   for(auto i=0u;i< rv.nv;i++) {
      for(auto j=0u;j< rv.nv;j++) {
         double d;
         f >> d;
         rv.d[i][j] = d2i(d);
      }
   }
   skip(f);
   for(auto i=0u;i< rv.nv;i++) {
      double a,b;
      f >> a >> b;
      rv.twin[i] = TimeWindow { d2i(a) , d2i(b) };
   }
   f.close();
   std::cout << rv.d << "\n";
   std::cout << "tw: " << rv.twin << "\n";
   return rv;
}


double greedy(Matrix<int,2>& d,GNSet C,GNSet U,int src,int sink,int hops)
{
   // compute lower bound as the sum of the cheapest arc out of each 
   // node except the src (which already has an outgoing arc per the
   // partial solution so far. 
   GNSet t = U.insert(src).insert(sink);
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

   const auto init = [&C,&tw]()      { return TSPTW { C - depot, depot, 0,  0 }; };
   const auto target = [sz,&C,&tw]() { return TSPTW { GNSet{},   depot, 0, sz }; };
   // const auto lgf = [sz,&C,&d,&tw](const TSPTW& s,DDContext)  {
   //    if (s.hops >= sz-1) {
   //       return GNSet {depot};
   //    } else {
   //       GNSet valid; 
   //       for(auto u: s.U) {
   //          if( u == depot || u == s.e || s.t + d[s.e][u] > tw[u].b ) continue;
   //          float tu = std::max(s.t + d[s.e][u], tw[u].a);
   //          bool onTime = true;
   //          for (auto v: s.U) {
   //             if(v == depot || v == s.e || v == u) continue;
   //             if( tu + d[u][v] > tw[v].b )
   //                onTime = false;
   //                break;
   //          }
   //          if( onTime ) valid.insert(u);
   //       }
   //       //std::cout << "valid: " << valid << std::endl<< std::endl;
   //       return valid;
   //    }     
   // };
   const auto lgf = [sz,&C,&d,&tw](const TSPTW& s,DDContext)  {
      if (s.hops >= sz-1) {
         return s.t + d[s.e][depot] <= tw[depot].b ? GNSet {depot} : GNSet{};
      } else {
         GNSet valid;
         GNSet nextU = s.U - GNSet{s.e, depot}; 
         for(auto u: nextU) {
            if(s.t + d[s.e][u] <= tw[u].b) valid.insert(u);
         }
         //std::cout << "valid: " << valid << std::endl<< std::endl;
         return valid;
      }     
   };
   const auto stf = [sz,&C,&d,&tw](const TSPTW& s,const int label) -> std::optional<TSPTW> {
      if (label==depot) {
         return TSPTW { GNSet{},depot,0,sz}; 
      } else {
         int nextT = std::max(s.t+d[s.e][label], tw[label].a);
         GNSet nextU = s.U - GNSet{label, depot};
         for(auto u: nextU) {
            if(nextT + d[label][u] > tw[u].b) {
               return std::nullopt;
            }
         }
         return TSPTW { nextU, label, nextT, s.hops+1};
      }
      //}
   };
   const auto scf = [&d,&tw](const TSPTW& s,int label) { // partial cost function 
      return d[s.e][label];
      // int delta = d[s.e][label];
      // if(s.t+delta < tw[label].a) {
      //    return tw[label].a - s.t;
      // } else
      //    return delta;
   };
   const auto smf = [](const TSPTW& s1,const TSPTW& s2) -> std::optional<TSPTW> {
      if (s1.e == s2.e && s1.hops == s2.hops)  {
         int e = s1.e; int hops = s1.hops;
         return TSPTW {s1.U | s2.U, e, hops, std::min(s1.t, s2.t)};
      } else {
         return std::nullopt; // return  the empty optional
      }
      //return std::nullopt;
   };
   const auto eqs = [sz](const TSPTW& s) -> bool { 
      //std::cout << s.e << " == " << depot << " && " << s.hops << " == " << sz << std::endl;
      return s.e == depot && s.hops == sz;// && s.U.empty(); 
   };
   
   // struct LocalKey {
   //    GNSet U;
   //    int e;

   //    std::size_t operator()(const LocalKey& lk) const { // I don't love having the key type be the hash type as well...
   //       return (std::hash<GNSet>{}(lk.U) << 16) |  // check if this is OK
   //              (std::hash<int>{}(lk.e));
   //    }
   //    bool operator==(const LocalKey& other) const { 
   //       return (U==other.U && e==other.e);
   //    }
   // };
   // std::unordered_map<LocalKey, int, LocalKey> localCache; 
   int dIn[sz];
   int dOut[sz];
   int perm1arr[sz];
   int perm2arr[sz];
   for(auto j : C) {
      GNSet allButj = C - GNSet{j};
      auto [e1, minIn]  = argmin(allButj,[&d,j](int k) { return d[k][j];});
      auto [e2, minOut] = argmin(allButj,[&d,j](int k) { return d[j][k];});
      dIn[j] = minIn;
      dOut[j] = minOut;
      perm1arr[j] = j;
      perm2arr[j] = j;
   }
   mergeSortPerm(dIn,  perm1arr, sz, [](double a, double b) { return a < b; });
   mergeSortPerm(dOut, perm2arr, sz, [](double a, double b) { return a < b; });
   std::vector<int> perm1(perm1arr, perm1arr+sz);
   std::vector<int> perm2(perm2arr, perm2arr+sz);
   std::cout << "perm2: " << perm2 << "\n" << (*perm2arr) << " " << *(perm2arr+sz-1) << "\n";
   // std::unordered_map<LocalKey, int, LocalKey> localCache; 
   const auto local = [&d,&C,&dIn,&dOut,&sz,&perm1,&perm2](const TSPTW& s,LocalContext) -> double {
      GNSet curr1 = s.U | GNSet{depot};
      GNSet curr2 = s.U | GNSet{s.e};
      int sumIn = 0, sumOut = 0;
      for(int i = 0, n = 0; (i < sz) && (n < sz-s.hops); i++) { 
         if( curr1.contains(perm1[i]) ) {
            sumIn += dIn[i];
            n++; 
         }
      }
      for(int i = 0, n = 0; (i < sz) && (n < sz-s.hops); i++) { 
         if( curr2.contains(perm2[i]) ) {
            sumOut += dOut[i];
            n++; 
         }
      }
      //auto sumIn = sum(curr1,[&dIn](int j)   { return dIn[j];});
      //auto sumOut = sum(curr2,[&dOut](int j) { return dOut[j];});
      return std::max(sumIn,sumOut);   
   };

   // auto curr = init();
   // std::cout << "init()        = " << init() << "\n";
   // std::cout << "local(init()) = " << local(init(), LocalContext::DDInit)/10000.0 << "\n";
   // return 0;

   BAndBRestrictedFirst engine(DD<TSPTW,Minimize<double>,
                decltype(target),
                decltype(lgf),
                decltype(stf),
                decltype(scf),
                decltype(smf),
                decltype(eqs),
                decltype(local)
                >::makeDD(init,target,lgf,stf,scf,smf,eqs,C,local),w);
   engine.search(bnds);
   return 0;
}

