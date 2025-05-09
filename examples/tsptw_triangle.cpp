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
   NatSet<4>  U; // unvisited cities
   int        e; // current city
   int        t; // time
   int     hops;
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
      return (std::hash<NatSet<4>>{}(v.U) << 32) |  // check if this is OK
         (std::hash<int>{}(v.t) << 16) |
         (std::hash<int>{}(v.e) << 8) |
         std::hash<int>{}(v.hops);
   }
};

struct Instance {
   size_t nv;
   Matrix<int,2>   d;
   std::vector<TimeWindow> twin;
   Instance() {}
   NatSet<4> vertices() {
      if (nv >= 256) abort();
      return NatSet<4>(0,nv-1);      
   } 
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
   const int sz = (const int)C.size();

   const auto init = [&C,&tw]()      { return TSPTW { C - depot, depot, 0,  0 }; };
   const auto target = [sz,&C,&tw]() { return TSPTW { NatSet<4>(),   depot, 0, sz }; };
   const auto lgf = [sz,&C,&d,&tw](const TSPTW& s,DDContext)  {
      if (s.hops >= sz-1) {
         return s.t + d[s.e][depot] <= tw[depot].b ? increasing(NatSet<4> {depot}) : increasing(NatSet<4>{});
         //return s.t + d[s.e][depot] <= tw[depot].b ? NatSet<4> {depot} : NatSet<4>{};
      } else {
         // NatSet<4> valid;
         // for(auto u: s.U) {
         //    if(u == s.e || u == depot) continue;
         //    if(s.t + d[s.e][u] <= tw[u].b) valid.insert(u);
         // }
         // //std::cout << "valid: " << valid << std::endl<< std::endl;
         // return valid;//increasing(valid);
         return increasing(filter(s.U, [&s,&d,&tw](auto& u){ return (u != s.e && u != depot && s.t + d[s.e][u] <= tw[u].b); }));
      }     
   };
   const auto stf = [sz,&C,&d,&tw](const TSPTW& s,const int label) -> std::optional<TSPTW> {
      if (label==depot) {
         return TSPTW { NatSet<4>(),depot,0,sz}; 
      } else {
         int nextT = std::max(s.t+d[s.e][label], tw[label].a);
         for(auto u: s.U) {
            if (u== label || u == depot) continue;
            if(nextT + d[label][u] > tw[u].b) {
               return std::nullopt;
            }
         }
         NatSet<4> nextU = s.U;
         nextU.remove(label).remove(depot);
         return TSPTW { nextU, label, nextT, s.hops+1};
      }
   };
   const auto scf = [&d,&tw](const TSPTW& s,int label) { // partial cost function 
      return d[s.e][label];
   };
   const auto smf = [](const TSPTW& s1,const TSPTW& s2) -> std::optional<TSPTW> {
      if (s1.e == s2.e && s1.hops == s2.hops)  {
         return TSPTW {s1.U | s2.U, s1.e, s1.hops, std::min(s1.t, s2.t)};
      } else {
         return std::nullopt; // return  the empty optional
      }
   };
   const auto eqs = [sz](const TSPTW& s) -> bool { 
      //std::cout << s.e << " == " << depot << " && " << s.hops << " == " << sz << std::endl;
      return s.e == depot && s.hops == sz;
   };
   
   int* dIn = new int[sz];
   int* dOut= new int[sz];
   int* perm1 = new int[sz];
   int* perm2 = new int[sz];
   for(auto j : C) {
      auto allButj = C;allButj.remove(j);
      auto [e1, minIn]  = argmin(allButj,[&d,j](int k) { return d[k][j];});
      auto [e2, minOut] = argmin(allButj,[&d,j](int k) { return d[j][k];});
      dIn[j] = minIn;
      dOut[j] = minOut;
      perm1[j] = j;
      perm2[j] = j;
   }
   mergeSortPerm(dIn,  perm1, sz, [](double a, double b) { return a < b; });
   mergeSortPerm(dOut, perm2, sz, [](double a, double b) { return a < b; });
   const auto local = [&dIn,&dOut,&sz,&perm1,&perm2](const TSPTW& s,LocalContext) -> double {
      int sumIn = 0,sumOut = 0,n1=0,n2=0;
      for(int i = 0; (i < sz) && (n1 < sz-s.hops); i++) { 
         if( s.U.contains(perm1[i]) || perm1[i] == depot) {
            sumIn += dIn[i];
            n1++; 
         }
      }
      for(int i = 0; (i < sz) && (n2 < sz-s.hops); i++) { 
         if( s.U.contains(perm2[i]) || perm2[i] == s.e) {
            sumOut += dOut[i];
            n2++; 
         }
      }
      return std::max(sumIn,sumOut);   
   };
   const auto sDom = [](const TSPTW& a,const TSPTW& b) -> bool { 
      // if (a.U <= b.U) && (a.e == b.e) then a doms b iff a.t < b.t
      return  (a.e == b.e) && a.t < b.t && ((a.U & b.U) == a.U);
   };
   BAndB engine(DD<TSPTW,Minimize<double>, // to minimize
                //   BAndBRestrictedFirst engine(DD<TSPTW,Minimize<double>,
                decltype(target),
                decltype(lgf),
                decltype(stf),
                decltype(scf),
                decltype(smf),
                decltype(eqs),
                decltype(local)
                >::makeDD(init,target,lgf,stf,scf,smf,eqs,C,local,sDom),w);
   engine.search(bnds);
   return 0;
}

