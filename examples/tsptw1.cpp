#include "codd.hpp"
#include "heap.hpp"
#include "search.hpp"
#include "searchRestrictedFirst.hpp"
#include "searchRestrictedOnly.hpp"
#include "searchRestrictedOnlyNoQ.hpp"
#include "searchRestrictedFirstThreaded.hpp"
#include <getopt.h>

int d2i(double d) { return (int)(d * 10000); }

struct TimeWindow {
   int a,b;
   friend std::ostream& operator<<(std::ostream& os,const TimeWindow& t) {
      return os << "[" << t.a << "," << t.b << "]";
   }
};

using Set = NatSet<4>;
using TSPTW = std::tuple<Set,int,int,int>;


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
   const char* fName = nullptr;
   int w = -1;
   std::string solver = "XF";  // Default value

   const option long_opts[] = {
      {"instance", required_argument, 0, 'i'},
      {"width"   , required_argument, 0, 'w'},
      {"solver"  , required_argument, 0, 's'},
      {"help"    , no_argument      , 0, 'h'},
      {0, 0, 0, 0}
   };

   int opt;
   while ((opt = getopt_long(argc, argv, "i:w:s:h", long_opts, nullptr)) != -1) {
      switch (opt) {
         case 'i':
            fName = optarg;
            break;
         case 'w':
            w = std::atoi(optarg);
            if (w < 0) {
               std::cerr << "Error: Width must be a non-negative integer.\n";
               return -1;
            }
            break;
         case 's':
            solver = optarg;
            break;
         case 'h':
            std::cout << "Usage: " << argv[0] << " --instance <file> --width <int> [--solver <name>] [--nolocal] [--nodom]\n"
                      << "  -i, --instance <file>   Path to the instance file (required)\n"
                      << "  -w, --width <int>       Non-negative integer specifying the max width (required)\n"
                      << "  -s, --solver <name>     Name of the solver to use (optional, default: XF)\n"
                      << "  -h, --help              Show this help message and exit\n";
            return 0;
         case '?':
         default:
            return -1;
      }
   }

   if (fName == nullptr || w < 0) {
      std::cerr << "Error: --instance and --width are required.\n";
      return -1;
   }

   std::cout << "Instance file: " << fName << '\n';
   std::cout << "Width: " << w << '\n';
   std::cout << "Solver: " << solver << '\n';

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
      return d[std::get<1>(s)][label];
   };
   const auto smf = [](const TSPTW& s1,const TSPTW& s2) -> std::optional<TSPTW> {
      const auto& [U1,e1,t1,hops1] = s1;
      const auto& [U2,e2,t2,hops2] = s2;
      if (e1 == e2 && hops1 == hops2)  {
         return TSPTW {U1 | U2, e1, hops1, std::min(t1, t2)};
      } else {
         return std::nullopt; // return  the empty optional
      }
   };
   const auto eqs = [sz](const TSPTW& s) -> bool { 
      return std::get<1>(s) == depot && std::get<3>(s) == sz;
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
      const auto& [U,e,t,hops] = s;
      int sumIn = 0,sumOut = 0,n1=0,n2=0;
      for(int i = 0; (i < sz) && (n1 < sz-hops); i++) { 
         if( U.contains(perm1[i]) || perm1[i] == depot) {
            sumIn += dIn[i];
            n1++; 
         }
      }
      for(int i = 0; (i < sz) && (n2 < sz-hops); i++) { 
         if( U.contains(perm2[i]) || perm2[i] == e) {
            sumOut += dOut[i];
            n2++; 
         }
      }
      return std::max(sumIn,sumOut);   
   };
   const auto sDom = [](const TSPTW& a,const TSPTW& b) -> bool { 
      const auto& [aU,ae,at,ahops] = a;
      const auto& [bU,be,bt,bhops] = b;
      //return  (aU == bU) && (ae == be) && at < bt;
      return  at < bt;
      //return ae==be && at < bt;
   };

   if(solver == "XF") {
      BAndB engine(DD<TSPTW,
                      Minimize<double>,
                      Projection<TSPTW,0,1>::Tuple,
                      decltype(target),
                      decltype(lgf),
                      decltype(stf),
                      decltype(scf),
                      decltype(smf),
                      decltype(eqs)>::makeDD(init,target,lgf,stf,scf,smf,eqs,C,local,project2<TSPTW,0,1>,sDom),
                   w);
      engine.search(bnds);
   } else if(solver == "RF") {
      BAndBRestrictedFirst engine(DD<TSPTW,
                                     Minimize<double>,
                                     Projection<TSPTW,0,1>::Tuple,
                                     decltype(target),
                                     decltype(lgf),
                                     decltype(stf),
                                     decltype(scf),
                                     decltype(smf),
                                     decltype(eqs)>::makeDD(init,target,lgf,stf,scf,smf,eqs,C,local,project2<TSPTW,0,1>,sDom),
                                  w);
      engine.search(bnds);
   } else if(solver == "RFT") {
      BAndBRestrictedFirstThreaded engine(DD<TSPTW,
                                             Minimize<double>,
                                             Projection<TSPTW,0,1>::Tuple,
                                             decltype(target),
                                             decltype(lgf),
                                             decltype(stf),
                                             decltype(scf),
                                             decltype(smf),
                                             decltype(eqs)>::makeDD(init,target,lgf,stf,scf,smf,eqs,C,local,project2<TSPTW,0,1>,sDom),
                                          w);
      engine.search(bnds);
   } else if(solver == "RO") {
      BAndBRestrictedOnly engine(DD<TSPTW,Minimize<double>,
                                    Projection<TSPTW,0,1>::Tuple,
                                    decltype(target),
                                    decltype(lgf),
                                    decltype(stf),
                                    decltype(scf),
                                    decltype(smf),
                                    decltype(eqs),
                                    decltype(sDom)
                                    >::makeDD(init,target,lgf,stf,scf,smf,eqs,C,local,project2<TSPTW,0,1>,sDom),
                                 w);
      engine.search(bnds);
   } else if(solver == "RONQ") {
      BAndBRestrictedOnlyNoQ engine(DD<TSPTW,Minimize<double>,
                                       Projection<TSPTW,0,1>::Tuple,
                                       decltype(target),
                                       decltype(lgf),
                                       decltype(stf),
                                       decltype(scf),
                                       decltype(smf),
                                       decltype(eqs),
                                       decltype(sDom)
                                       >::makeDD(init,target,lgf,stf,scf,smf,eqs,C,local,project2<TSPTW,0,1>,sDom),
                                    w);
      engine.search(bnds);
   } else {
      std::cerr << "The solver " << solver << " is unknown.\n"
                << "This model only supports the following solvers:\n"
                << "XF     relaXed First\n"
                << "RF     Restricted First\n"
                << "RFT    Restricted First Threaded\n"
                << "RO     Restricted Only\n"
                << "RONQ   Restricted Only No Queue (CABS-style)";
      return -1;
   }
   
   return 0;
}