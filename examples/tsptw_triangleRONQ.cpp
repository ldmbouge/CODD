#include "codd.hpp"
#include "heap.hpp"
#include "searchRestrictedFirst.hpp"
#include "searchRestrictedOnly.hpp"
#include "searchRestrictedOnlyNoQ.hpp"

int d2i(double d) { return (int)(d * 10000); }

struct TimeWindow {
   int a,b;
   friend std::ostream& operator<<(std::ostream& os,const TimeWindow& t) {
      return os << '<' << t.a << ',' << t.b << "> ";
   }
};

using Set = NatSet<4>;
using TSPTW = std::tuple<Set,int,int,int>;
//using DC = Projection<TSPTW,0,1>::Tuple;

/*
std::ostream& operator<<(std::ostream& os,const TSPTW& m) {
   const auto& [U,e,t,h] = m;
   return os << "<" << U << ',' << e << ',' << t << ',' << h << ">";
   }*/

/*struct TSPTW {
   Set    U; // unvisited cities
   int    e; // current city
   int    t; // time
   int hops;
   friend std::ostream& operator<<(std::ostream& os,const TSPTW& m) {
      return os << "<" << m.U << ',' << m.e << ',' << m.t << ',' << m.hops << ">";
   }
};
*/
/*
template<> struct std::equal_to<TSPTW> {
   constexpr bool operator()(const TSPTW& s1,const TSPTW& s2) const {
      const auto& [s1U,s1e,s1t,s1h] = s1;
      const auto& [s2U,s2e,s2t,s2h] = s2;
      return s1e==s2e && s1t==s2t && s1h==s2h && s1U==s2U;
      //return s1.e == s2.e && s1.t==s2.t && s1.hops==s2.hops && s1.U == s2.U;
      //return s1.e == s2.e && s1.hops==s2.hops;
   }
};
*/
/*
template<> struct std::hash<TSPTW> {
   std::size_t operator()(const TSPTW& v) const noexcept {
      const auto&  [U,e,t,h] = v;
      return (std::hash<Set>{}(U) << 32) |  // check if this is OK
         (std::hash<int>{}(t) << 16) |
         (std::hash<int>{}(e) << 8) |
         std::hash<int>{}(h);
   }
};
*/

struct Instance {
   size_t nv;
   Matrix<int,2>   d;
   std::vector<TimeWindow> twin;
   Instance() {}
   Set vertices() {
      if (nv >= 256) abort();
      return Set(0,nv-1);      
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
   std::cout << "Distance:\n";
   for(int r = 0; r < d.getDim(0);r++) {
      std::cout << "R(" << std::setw(2) << r << "): ";
      for(int c = 0; c < d.getDim(1);c++) 
         std::cout << d[r][c] << " ";      
      std::cout << "\n";
   }
   Bounds bnds([](const std::vector<int>& inc)  {
   });

   const int depot = 0;
   const int sz = (const int)C.size();

   const auto init = [&C,depot]()   { return TSPTW { C - depot, depot, 0,  0 }; };
   const auto target = [sz,depot]() { return TSPTW { Set(),   depot, 0, sz }; };
   const auto lgf = [sz,&d,&tw](const TSPTW& s,DDContext)  {
      const auto& [U,e,t,hops] = s;
      if (hops >= sz-1) 
         return (t + d[e][depot] <= tw[depot].b) ? Set {depot} : Set{}; // that's the only way to return the depot
      else 
         return filter(U, [&U,e,t,&d,&tw](auto& u){ return t + d[e][u] <= tw[u].b; }); // neither s.e nor depot IN s.U     
   };
   const auto stf = [sz,depot,&d,&tw](const TSPTW& s,const int label) -> std::optional<TSPTW> {
      if (label==depot) {
         return TSPTW { Set(),depot,0,sz}; 
      } else {
         const auto& [U,e,t,hops] = s;
         const int nextT = std::max(t + d[e][label], tw[label].a);
         Set nextU = U - label;
         for(auto u : nextU) 
            if(nextT + d[label][u] > tw[u].b) 
               return std::nullopt;                     
         return TSPTW { nextU, label, nextT, hops + 1};
      }
   };
   const auto scf = [&d](const TSPTW& s,int label) noexcept { // partial cost function 
      //const auto& [U,e,t,hops] = s;
      return d[std::get<1>(s)][label];
   };
   const auto eqs = [sz](const TSPTW& s) noexcept -> bool { 
      //const auto& [U,e,t,hops] = s;
      return std::get<1>(s) == depot && std::get<3>(s) == sz;
   };
   /*
   const auto domClass = [](const TSPTW& a,const TSPTW& b) -> bool {
      [[maybe_unused]] const auto& [aU,ae,at,ah] = a;
      [[maybe_unused]] const auto& [bU,be,bt,bh] = b;
      return aU==bU && ae == be;
      };*/
   const auto sDom = [](const TSPTW& a,const TSPTW& b) -> bool { 
      const auto& [aU,ae,at,ahops] = a;
      const auto& [bU,be,bt,bhops] = b;
      return  (aU == bU) && (ae == be) && at < bt;
      //return ae==be && at < bt;
   };
   const auto merge = [](const TSPTW&a,const TSPTW& b) -> std::optional<TSPTW> {
      return std::nullopt;
   };
   // const auto local = [](const TSPTW& s,LocalContext) -> double {
   //    return 0.0;
   // };

   int* dIn = new int[sz];
   int* dOut= new int[sz];
   for(auto j : C) {
      auto [e1, minIn]  = argmin(C - j,[&d,j](int k) { return d[k][j];});
      auto [e2, minOut] = argmin(C - j,[&d,j](int k) { return d[j][k];});
      dIn[j] = minIn;
      dOut[j] = minOut;
   }
   for(auto j : C) std::cout << j << ":" << dIn[j] << " ";std::cout << "\n";
   for(auto j : C) std::cout << j << ":" << dOut[j] << " ";std::cout << "\n";   
   const auto local = [dIn,dOut,sz,depot](const TSPTW& s,LocalContext) -> double {
      const auto& [U,e,t,hops] = s;
      if (e == depot) return 0;
      int sumIn  = sum(U,[dIn](int j) { return dIn[j];})  + dIn[depot];
      int sumOut = sum(U,[dOut](int j) { return dOut[j];}) + dOut[e];
      return std::max(sumIn,sumOut);   
   };

   
   BAndBRestrictedOnlyNoQ engine(DD<TSPTW,Minimize<double>,
                              Projection<TSPTW,0,1>::Tuple,
                              decltype(target),
                              decltype(lgf),
                              decltype(stf),
                              decltype(scf),
                              decltype(merge),
                              decltype(eqs),
                              decltype(sDom)
                              >::makeDD(init,target,
                                        lgf,stf,scf,
                                         merge,
                                        eqs,
                                        C,
                                        local,
                                        project2<TSPTW,0,1>,
                                        sDom),w);
   engine.search(bnds);
   return 0;
}

