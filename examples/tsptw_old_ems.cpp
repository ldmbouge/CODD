#include "codd.hpp"
#include "heap.hpp"

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
   const auto lgf = [sz,&C,&d,&tw](const TSPTW& s,DDContext)  {
      if (s.hops >= sz-1) {
         return s.t + d[s.e][depot] <= tw[depot].b ? GNSet {depot} : GNSet{};
      } else {
         GNSet valid; 
         for(auto u: s.U) {
            bool tmp = (u != depot) && (s.e != u) && (s.t + d[s.e][u] <= tw[u].b);
            if( tmp ) valid.insert(u);
         }
         //std::cout << "valid: " << valid << std::endl<< std::endl;
         return valid;
      }     
   };
   const auto stf = [sz,&C,&d,&tw](const TSPTW& s,const int label) -> std::optional<TSPTW> {
      if (label==depot)
         return TSPTW { GNSet{},depot,0,sz}; 
      // else {
      //std::cout << s.U << " \ " << label << " = " << s.U - GNSet{label} << std::endl;
      //std::cout << "max(" << s.t << "+" << d[s.e][label] << "=" << s.t+d[s.e][label] << ", " << tw[label].a << ") = " << std::max(s.t+d[s.e][label], tw[label].a) << std::endl;
      else
         return TSPTW { s.U - GNSet{label}, label, std::max(s.t+d[s.e][label], tw[label].a), s.hops+1};
      //}
   };
   const auto scf = [&d,&tw](const TSPTW& s,int label) { // partial cost function 
      int delta = d[s.e][label];
      if(s.t+delta < tw[label].a) {
         return tw[label].a - s.t;
      } else
         return delta;
   };
   const auto smf = [](const TSPTW& s1,const TSPTW& s2) -> std::optional<TSPTW> {
      if (s1.e == s2.e && s1.hops == s2.hops)  {
         int e = s1.e; int hops = s1.hops;
         return TSPTW {s1.U | s2.U, e, hops, std::min(s1.t, s2.t)};
      } else {
         return std::nullopt; // return  the empty optional
      }
   };
   const auto eqs = [sz](const TSPTW& s) -> bool { 
      //std::cout << s.e << " == " << depot << " && " << s.hops << " == " << sz << std::endl;
      return s.e == depot && s.hops == sz && s.U.empty(); 
   };
   
   struct LocalKey {
      GNSet U;
      int e;

      std::size_t operator()(const LocalKey& lk) const { // I don't love having the key type be the hash type as well...
         return (std::hash<GNSet>{}(lk.U) << 16) |  // check if this is OK
                (std::hash<int>{}(lk.e));
      }
      bool operator==(const LocalKey& other) const { 
         return (U==other.U && e==other.e);
      }
   };
   std::unordered_map<LocalKey, int, LocalKey> localCache; 
   const auto local = [&d,&C,&localCache](const TSPTW& s,LocalContext) -> double {
      LocalKey curr = {s.U - GNSet{depot}, s.e};
      int total = 0;
      while (!curr.U.empty()) {
         auto[next, minD] = argmin(curr.U, [&d, &e=curr.e](int other){ return d[e][other]; });
         total += minD;
         curr.e = next;
         curr.U.remove(next);
      }
      total += d[curr.e][depot];
      return total;
   };

   // int cost = 0;
   // //std::vector<int> labels = { 1, 5, 3, 2, 4, 8, 6, 7, 9, 10, 0  };
   // //std::vector<int> labels = { 1,2,3,4,5,6,7,8,9,10,0  };
   // //std::vector<int> labels = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,0 };
   // std::vector<int> labels = { 1, 2, 3, 4, 16, 6, 7, 8, 9, 10, 11, 5, 13, 14, 15, 12, 17, 0 };
   // auto curr = init();
   // for(int label: labels){
   //    lgf(curr, DDContext::DDRestricted);
   //    cost += scf(curr, label);
   //    auto next = stf(curr, label);
   //    if(next != std::nullopt)
   //       curr = next.value();
   // }
   // std::cout << "value(" << labels << ") = " << cost << std::endl;
   // return 0;

   BAndB engine(DD<TSPTW,Minimize<double>,
                std::tuple<int>,
                decltype(target),
                decltype(lgf),
                decltype(stf),
                decltype(scf),
                decltype(smf),
                decltype(eqs)
                >::makeDD(init,target,lgf,stf,scf,smf,eqs,C,local),w);
   engine.search(bnds);
   return 0;
}

