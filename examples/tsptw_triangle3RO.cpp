#include "codd.hpp"
#include "heap.hpp"
#include "searchRestrictedOnly.hpp"

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
   Set must; // cities that are unvisited in all nodes in the prefix of this state
   Set may;  // cities that are visited in some, but not all, nodes in the prefix
   int hops; // number of visited cities
   int ta;   // earliest and latest times at pos (a and b are same for exact nodes)
   int tb;
   int pad; 
   TSPTW() { memset(this,0,sizeof(TSPTW));}
   TSPTW(const Set& p,const Set& m,const Set& maya,int h,int a,int b) : pos(p),must(m),may(maya),hops(h),ta(a),tb(b),pad(0) {}
   TSPTW(Set&& p,Set&& m,Set&& maya,int h,int a,int b) : pos(p),must(m),may(maya),hops(h),ta(a),tb(b),pad(0) {}
   friend std::ostream& operator<<(std::ostream& os,const TSPTW& m) {
      return os << "<" << m.pos << ", " << m.hops << ", [" << m.ta << "," << m.tb << "], " << m.must << ", " << m.may << ">";
   }
};

template<size_t n>
inline int fast_memcmp(const void *s1, const void *s2) {
   switch(n & 7) {
      case 0: {
         const unsigned long long* p1 = static_cast<const unsigned long long*>(s1);
         const unsigned long long* p2 = static_cast<const unsigned long long*>(s2);
         size_t dwl = n >> 3;
         unsigned long long diff = 0;
         while (diff==0 && dwl--)
            diff = *p1++ - *p2++;
         return diff!=0;
      }
      default: return memcmp(s1,s2,n);
   }
}

template<> struct std::equal_to<TSPTW> {
   constexpr bool operator()(const TSPTW& s1,const TSPTW& s2) const {
      //return fast_memcmp<sizeof(TSPTW)>(&s1,&s2)==0;      
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

   const auto init   = [&C] () { return TSPTW { TSPTW::Set{depot}, C - depot   , TSPTW::Set{} ,0, 0, 0}; };
   const auto target = [&sz]() { return TSPTW { TSPTW::Set{depot}, TSPTW::Set{}, TSPTW::Set{} ,sz, 0, 0}; };
   const auto lgf = [sz,&d,&tw](const TSPTW& s,DDContext)  noexcept {
      if (s.hops >= sz-1) { /*s.must.empty() && s.may.empty()*/
         const int a = min(s.pos, [ta=s.ta,&d](const int p){ return ta + d[p][depot]; } );
         //const int b = max(s.pos, [tb=s.tb,&d](const int p){ return tb + d[p][depot]; } );

         //const int b = min(s.pos, [tb=s.tb,&d](const int p){ return tb + d[p][depot]; } );
         //return (a <= tw[depot].b && b >= tw[depot].a) ? TSPTW::Set{depot} : TSPTW::Set{};
         return (a <= tw[depot].b) ? TSPTW::Set{depot} : TSPTW::Set{};
      } else {
         // std::cout << (s) << "\n";
         // std::cout << "pre-filter : " << (s.must|s.may) << "\n";
         const auto f =  filter(
            s.must|s.may, 
            [&s,&d,&tw](auto& u){ 
               const int a = min(s.pos, [&u](const int p){ return p != u; }, [ta=s.ta,&d,&u](const int p){ return ta + d[p][u]; } );
               //std::cout << a << " <= " << tw[u].b <<"\n";
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
         return TSPTW { TSPTW::Set{label}, newMust, s.may-label,s.hops+1, ta, tb };
      }
   };
   const auto scf = [&d](const TSPTW& s,int label) noexcept { // partial cost function 
      return min(s.pos, [&label](const int p){ return p != label; }, [&label,&d](int p) { return d[p][label]; });
   };
   const auto eqs = [&sz](const TSPTW& s) noexcept -> bool { 
      return s.pos.contains(depot) && s.hops == sz;// && s.must.empty(); //&& s.may.empty();
   };
   
   int* dInNS   = new int[sz];
   int* dIn     = new int[sz];
   int* dOut    = new int[sz];
   int* permIn  = new int[sz];
   int* permOut = new int[sz];

   for(auto j : C) {
      auto allButj = C;allButj.remove(j);
      auto [e1, minIn]  = argmin(allButj,[&d,j](int k) { return d[k][j];});
      auto [e2, minOut] = argmin(allButj,[&d,j](int k) { return d[j][k];});
      dIn[j] = dInNS[j] = minIn;
      dOut[j] = minOut;
      permIn[j] = j;
      permOut[j] = j;
   }
   mergeSortPerm(dIn,  permIn,  sz, [](double a, double b) { return a < b; }); // From smallest to largest
   mergeSortPerm(dOut, permOut, sz, [](double a, double b) { return a < b; }); // From smallest to largest

   const auto local = [dIn,dInNS,dOut,sz,permIn,permOut,&tw,&d](const TSPTW& s,LocalContext) -> double {
      const auto inf = std::numeric_limits<int>::max();
      const auto violatesTW = [ta=s.ta,&tw,dInNS](int p){ return ta + dInNS[p] > tw[p].b; };

      if(any(s.must, violatesTW)) { // Nasty bug here. Sorting was wrong. (flipped >)
         //std::cout << "INF1\n"; 
         return inf; // but for violatesTW to be correct, since it says "If There exist a city in must such that ..."
      }              // the index "p" to the closure is a city name. Can't used dIn. Must use the non-sorted version.

      const int completeTour = (sz-1) - s.hops - s.must.size();
      int mandatoryIn = 0,mandatoryOut = 0;
      // mandatory part
      for(int i = 0; i < sz; i++) {
         mandatoryIn  += dIn [i] * s.must.contains(permIn [i]);
         mandatoryOut += dOut[i] * s.must.contains(permOut[i]);
      }     
      if(s.may.size() > 0) {
         //std::cout << completeTour << "=" << sz << "-" << s.hops << "-" << s.must.size()<< " " << s.must << "\n";
         //std::cout << s.may << " " << s.may.size() <<"-"<< count(s.may, violatesTW) <<"<"<< completeTour << "\n";
         if(s.may.size() - count(s.may, violatesTW) < completeTour) {
            //std::cout << "INF2\n"; 
            return inf;
         }
         int nIn  = 0,nOut = 0;
         for(int i = 0; i < sz && nIn < completeTour; i++) {
            const bool use = s.may.contains(permIn [i]);
            mandatoryIn += dIn[i] * use;
            nIn += use;
         }
         for(int i = 0; i < sz && nOut < completeTour; i++) {
            const bool use = s.may.contains(permOut[i]);
            mandatoryOut += dOut[i] * use;
            nOut += use;
         }
      }
      const int returnToDepot = (mandatoryIn == 0) ?
           min(s.must | s.may | s.pos,[&d](int p){ return d[p][depot]; })
         : min(s.must | s.may        ,[&d](int p){ return d[p][depot]; });

      if(s.ta + mandatoryIn + returnToDepot > tw[depot].b) {
         //std::cout << "INF3\n"; 
         return inf;
      }
      return std::max(mandatoryIn, mandatoryOut) + returnToDepot;
   };
   const auto sDom = [](const TSPTW& a,const TSPTW& b) noexcept -> bool { 
      return (a.must <= b.must) && a.ta < b.ta && a.hops == b.hops && a.pos == b.pos;
   };

   //BAndB engine(DD<TSPTW,Minimize<double>, // to minimize
   BAndBRestrictedOnly engine(DD<TSPTW,Minimize<double>,
                decltype(target),
                decltype(lgf),
                decltype(stf),
                decltype(scf),
                AbstractDD::nullmerge_t<TSPTW>,
                decltype(eqs),
                decltype(local)
                >::makeDD(init,target,lgf,stf,scf,AbstractDD::nullmerge<TSPTW>,eqs,C,local,sDom),w);
   engine.search(bnds);
   return 0;
}
