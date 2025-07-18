#include "codd.hpp"
#include "heap.hpp"
#include "searchRestrictedFirst.hpp"
#include "searchRestrictedOnly.hpp"

int d2i(double d) { return (int)(d * 10000); }

struct TimeWindow {
   int a,b;
   friend std::ostream& operator<<(std::ostream& os,const TimeWindow& t) {
      return os << '<' << t.a << ',' << t.b << "> ";
   }
};

struct TSPTW {
   using Set = NatSet<4>;
   Set    U; // unvisited cities
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
      return (std::hash<TSPTW::Set>{}(v.U) << 32) |  // check if this is OK
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
   TSPTW::Set vertices() {
      if (nv >= 256) abort();
      return TSPTW::Set(0,nv-1);      
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

   const auto init = [&C]()   { return TSPTW { C - depot, depot, 0,  0 }; };
   const auto target = [sz]() { return TSPTW { TSPTW::Set(),   depot, 0, sz }; };
   const auto lgf = [sz,&d,&tw](const TSPTW& s,DDContext)  {
      if (s.hops >= sz-1) 
         return (s.t + d[s.e][depot] <= tw[depot].b) ? TSPTW::Set {depot} : TSPTW::Set{}; // that's the only way to return the depot
      else 
         return filter(s.U, [&s,&d,&tw](auto& u){ return s.t + d[s.e][u] <= tw[u].b; }); // neither s.e nor depot IN s.U     
   };
   const auto stf = [sz,&d,&tw](const TSPTW& s,const int label) -> std::optional<TSPTW> {
      if (label==depot) {
         return TSPTW { TSPTW::Set(),depot,0,sz}; 
      } else {
         const int nextT = std::max(s.t+d[s.e][label], tw[label].a);
         TSPTW::Set nextU = s.U - label;
         for(auto u : nextU) 
            if(nextT + d[label][u] > tw[u].b) 
               return std::nullopt;                     
         return TSPTW { nextU, label, nextT, s.hops+1};
      }
   };
   const auto scf = [&d](const TSPTW& s,int label) noexcept { // partial cost function 
      return d[s.e][label];
   };
   const auto eqs = [sz](const TSPTW& s) noexcept -> bool { 
      return s.e == depot && s.hops == sz;
   };
   const auto sDom = [](const TSPTW& a,const TSPTW& b) -> bool { 
      return  (a.U == b.U) && (a.e == b.e) && a.t < b.t;
   };
   BAndBRestrictedOnly engine(DD<TSPTW,Minimize<double>,
                              decltype(target),
                              decltype(lgf),
                              decltype(stf),
                              decltype(scf),
                              AbstractDD::nullmerge_t<TSPTW>,
                              decltype(eqs)
                              >::makeDD(init,target,lgf,stf,scf,AbstractDD::nullmerge<TSPTW>,eqs,C,nullptr,sDom),w);
   engine.search(bnds);
   return 0;
}

