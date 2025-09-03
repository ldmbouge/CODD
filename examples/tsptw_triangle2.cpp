#include "codd.hpp"
#include "heap.hpp"
#include "searchRestrictedFirst.hpp"

int d2i(double d) { return (int)(d * 10000); }

struct TimeWindow {
   int a,b;
   friend std::ostream& operator<<(std::ostream& os,const TimeWindow& t) {
      return os << "[" << t.a << "," << t.b << "]";
   }
};

struct TSPTW {
   using Set = NatSet<4>;
   Set pos;  // current city (set for merged nodes, singleton otherwise)
   int hops; // number of visited cities
   int ta;   // earliest and latest times at pos (a and b are same for exact nodes)
   int tb;
   Set must; // cities that are unvisited in all nodes in the prefix of this state
   Set may;  // cities that are visited in some, but not all, nodes in the prefix

   friend std::ostream& operator<<(std::ostream& os,const TSPTW& m) {
      return os << "<" << m.pos << ", " << m.hops << ", [" << m.ta << "," << m.tb << "], " << m.must << ", " << m.may << ">";
   }
};


template<> struct std::equal_to<TSPTW> {
   constexpr bool operator()(const TSPTW& s1,const TSPTW& s2) const {
      return s1.pos  == s2.pos  &&
             s1.hops == s2.hops &&
             s1.ta   == s2.ta   &&
             s1.tb   == s2.tb   &&
             s1.must == s2.must &&
             s1.may  == s2.may   ;
   }
};

template<> struct std::hash<TSPTW> {
   std::size_t operator()(const TSPTW& v) const noexcept {
      return std::rotl(std::hash<TSPTW::Set>{}(v.pos) , 40) |
             std::rotl(std::hash<TSPTW::Set>{}(v.must), 32) | 
             std::rotl(std::hash<TSPTW::Set>{}(v.may) , 24) | 
             std::rotl(std::hash<int>{}(v.hops)       , 16) |
             std::rotl(std::hash<int>{}(v.ta)         ,  8) |
                       std::hash<int>{}(v.tb)               ;
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
   Bounds bnds([](const std::vector<int>& inc)  {
   });

   const int depot = 0;
   const int sz = (const int)C.size();

   const auto init   = [&C] () { return TSPTW { TSPTW::Set{depot},  0, 0, 0, C - depot   , TSPTW::Set{} }; };
   const auto target = [&sz]() { return TSPTW { TSPTW::Set{depot}, sz, 0, 0, TSPTW::Set{}, TSPTW::Set{} }; };
   const auto lgf = [sz,&d,&tw](const TSPTW& s,DDContext)  {
      if (s.hops >= sz-1) { /*s.must.empty() && s.may.empty()*/
         const int a = s.ta + min(s.pos, [&d](const int p){ return d[p][depot]; } );
         return (a <= tw[depot].b) ? TSPTW::Set{depot} : TSPTW::Set{};
      } else {
         // std::cout << (s) << "\n";
         // std::cout << "pre-filter : " << (s.must|s.may) << "\n";
         const auto f =  filter(
            s.must|s.may, 
            [&s,&d,&tw](const auto u){
               const int a = s.ta + min(s.pos - u,[&d,u](const int p){ return d[p][u]; } );
               return a <= tw[u].b && (s.pos.size() > 1 || !s.pos.contains(u));
            }
         );
         //std::cout << "state: " << s << "\nnext: " << f << "\n";
         return f;
      }     
   };
   const auto stf = [sz,&d,&tw,&target](const TSPTW& s,const int label) noexcept -> std::optional<TSPTW> {
      if (label==depot) {
         return target();
      } else {
         const int ta = std::max(s.ta + min(s.pos - label,[&d,label](const int p){ return d[p][label]; } ),
                                 tw[label].a);
         auto newMust = s.must - label;
         if (any(newMust,[&d,&tw,label,ta](int u) { return ta + d[label][u] > tw[u].b;}))
            return std::nullopt; // at least one in the next must violates its time window ub.
         const int tb = (s.ta == s.tb) ? 
            ta :
            std::min(s.tb + max(s.pos - label,[&d,label](const int p){ return d[p][label]; } ),
                     tw[label].b);         
         return TSPTW { TSPTW::Set{label}, s.hops+1,ta,tb,newMust, s.may-label };
      }
   };
   const auto scf = [&d](const TSPTW& s,int label) { // partial cost function 
      return min(s.pos - label,[label,&d](int p) { return d[p][label]; });
   };
   const auto smf = [](const TSPTW& s1,const TSPTW& s2) -> std::optional<TSPTW> {
      if (s1.hops != s2.hops) return std::nullopt;

      const auto newMust = s1.must & s2.must;
      return TSPTW {
         s1.pos | s2.pos,
         s1.hops,
         std::min(s1.ta, s2.ta),
         std::max(s1.tb, s2.tb),
         newMust,
         (s1.must | s1.may | s2.must | s2.may) - newMust
      };
   };
   const auto eqs = [&sz](const TSPTW& s) -> bool { 
      return s.pos.contains(depot) && s.hops == sz;// && s.must.empty(); //&& s.may.empty();
   };
   
   int* dIn = new int[sz];
   int* dOut= new int[sz];
   int* perm1 = new int[sz];
   int* perm2 = new int[sz];
   for(auto j : C) {
      auto [e1, minIn]  = argmin(C - j,[&d,j](int k) { return d[k][j];});
      auto [e2, minOut] = argmin(C - j,[&d,j](int k) { return d[j][k];});
      dIn[j] = minIn;
      dOut[j] = minOut;
      perm1[j] = j;
      perm2[j] = j;
   }
   mergeSortPerm(dIn,  perm1, sz, [](double a, double b) { return a < b; });
   mergeSortPerm(dOut, perm2, sz, [](double a, double b) { return a < b; });

   const auto local = [&d,&dIn,&dOut,&sz,&perm1,&perm2,&tw](const TSPTW& s,LocalContext) -> double {
      int sumIn = 0,sumOut = 0,n1=0,n2=0;
      for(int i = 0; (i < sz) && (n1 < sz-s.hops); i++) { 
         if( s.must.contains(perm1[i]) || perm1[i] == depot) {
            sumIn += dIn[i];
            n1++; 
         }
      }
      for(int i = 0; (i < sz) && (n1 < sz-s.hops); i++) { 
         if( s.may.contains(perm1[i]) || perm1[i] == depot) {
            sumIn += dIn[i];
            n1++; 
         }
      }
      for(int i = 0; (i < sz) && (n2 < sz-s.hops); i++) { 
         if( s.must.contains(perm2[i]) || s.pos.contains(perm2[i])) {
            sumOut += dOut[i];
            n2++; 
         }
      }
      for(int i = 0; (i < sz) && (n2 < sz-s.hops); i++) { 
         if( s.may.contains(perm2[i]) || s.pos.contains(perm2[i])) {
            sumOut += dOut[i];
            n2++; 
         }
      }
      return std::max(sumIn,sumOut);  
   };
   
   const auto sDom = [](const TSPTW& a,const TSPTW& b) -> bool {
      return (b.must <= a.must) && a.may==b.may && a.ta < b.ta; //  && a.hops==b.hops;
   };

   //   BAndBRestrictedFirst engine(DD<TSPTW,Minimize<double>,
   BAndB engine(DD<TSPTW,Minimize<double>, // to minimize
                std::tuple<TSPTW::Set,int>,
                decltype(target),
                decltype(lgf),
                decltype(stf),
                decltype(scf),
                decltype(smf),
                decltype(eqs)
                >::makeDD(init,target,lgf,stf,scf,smf,eqs,C,local,
                          [](const TSPTW& s) { return std::make_tuple(s.pos,s.hops);}, 
                          sDom),w);
   engine.search(bnds);
   return 0;
}
