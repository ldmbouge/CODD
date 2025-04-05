#include "codd.hpp"
#include "uf.hpp"
#include "heap.hpp"

struct TSP {
   GNSet  A; // locations on _all_ paths into state
   int e;
   int hops;
   friend std::ostream& operator<<(std::ostream& os,const TSP& m) {
      return os << "<" << m.A << ',' << m.e << ',' << m.hops << ">";
   }
};

template<> struct std::equal_to<TSP> {
   constexpr bool operator()(const TSP& s1,const TSP& s2) const {
      return s1.e == s2.e && s1.hops==s2.hops && s1.A == s2.A;
   }
};

template<> struct std::hash<TSP> {
   std::size_t operator()(const TSP& v) const noexcept {
      return (std::hash<GNSet>{}(v.A) << 32) |  // check if this is OK
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
   double w;
   friend bool operator==(const GE& e1,const GE& e2) {
      return e1.a == e2.a && e1.b == e2.b;
   }
   friend bool operator<(const GE& e1,const GE& e2) {
      return e1.w < e2.w;
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

enum WeightType {Explicit,EUC_2D,EUC_3D,MAX_2D,MAX_3D,MAN_2D,MAN_3D,
                 CEIL_2D,GEO,ATT,XEAY1,XEAY2,SPECIAL,NA};

enum WeightFormat {Function,FullMatrix,UpperRow,LowerRow,UpperDiagRow,LowerDiagRow,
                   UpperCol,LowerCol,UpperDiagCol,LowerDiagCol,FNA};

enum WeightType convertWT(std::string s) {
   if (s == "EXPLICIT")
      return Explicit;
   else return NA;
}

enum WeightFormat convertWF(std::string s) {
   if (s == "LOWER_DIAG_ROW")
      return LowerDiagRow;
   else if (s == "FULL_MATRIX")
      return FullMatrix;
   else return FNA;
}

std::string readLabel(std::ifstream& f) {
   std::string rv;
   char ch = 0;
   do {
      ch = f.get();
      if (!isspace(ch) && ch != ':')
         rv.push_back(ch);
   } while (ch != ':' && !isspace(ch) && !f.eof());
   if (!f.eof())
      f.unget();
   return rv;
}

Instance readFile(const char* fName)
{
   Instance rv;
   using namespace std;
   ifstream f(fName);
   string label,colon,name,comment,type,edgeWeight;
   enum WeightType wt = NA;
   enum WeightFormat wf = FNA;
   label = readLabel(f);
   std::getline(f,name);
   label = readLabel(f);
   std::getline(f,type);
   label = readLabel(f);
   std::getline(f,comment);
   label = readLabel(f);
   f >> colon >> rv.nv;      
   do {
      std::string back;
      label = readLabel(f);
      if (label=="EDGE_WEIGHT_TYPE") {
         f >> colon >> back;
         wt = convertWT(back);
      } else if (label=="EDGE_WEIGHT_FORMAT") {
         f >> colon >> back;
         wf = convertWF(back);
      } else 
         std::getline(f,back);
   } while (label != "EDGE_WEIGHT_SECTION");
   assert(label == "EDGE_WEIGHT_SECTION");
   std::cout << wt << " : " << wf << "\n";
   rv.d = Matrix<double,2>(rv.nv,rv.nv);
   switch(wf) {
      case FullMatrix: {
         for(auto i = 0u;i < rv.nv;++i)
            for(auto j=0u;j < rv.nv;++j) {
               int val;
               f >> val;
               rv.d[i][j] = val;
            }   
      }break;
      case LowerDiagRow: {
         for(auto i = 0u;i < rv.nv;++i) {
            for(auto j = 0u;j < i;++j) {
               int val;
               f >> val;
               rv.d[i][j] = val;
               rv.d[j][i] = val;
            }
            int val;
            f >> val;
            rv.d[i][i] = val;
         }
      }break;
   }
   f.close();
   std::cout << rv.d << "\n";
   return rv;
}


double mst(Matrix<double,2>& d,GNSet C,GNSet V,int src,int sink,int hops)
{
   GNSet t = (C - V).insert(src).insert(sink);
   UnionFind<int> uf;
   std::map<int,UnionFind<int>::Node::Ptr> vm;
   for(auto v : t) vm[v] = uf.makeSet(v);
   Pool mem;
   Heap<GE> pq(&mem,t.size()*t.size(),[](const GE& e1,const GE& e2) { return e1.w < e2.w;});
   for(auto a : t)
      for(auto b : t)
         if (a!=b) pq.insert(GE {a,b,d[a][b]});
   pq.buildHeap();
   int ne = 0,ts = C.size() - hops;
   double l = 0;
   while (ne != ts && !pq.empty()) {
      auto e = pq.extractMax();
      if (uf.setFor(vm[e.a]) != uf.setFor(vm[e.b])) {
         l += e.w;
         ++ne;
         uf.merge(vm[e.a],vm[e.b]);
      }         
   }
   return l;
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
   Bounds bnds([&d,&bnds](const std::vector<int>& inc)  {
      int pv = 0; // start at depot
      int td = 0;
      int cnt = 0;
      for(auto v : inc) {
         td += d[pv][v];
         pv = v;
         cnt++;
      }
      std::cout << "\n\tCHECKER tour length is " << td << ',' << bnds.getPrimal() << " len:" << cnt << "\n";
   });
   const int depot = 0;
   const int sz = (int)C.size();
   const auto init = []()               { return TSP { GNSet{depot},depot,0};};
   const auto target = [sz,&C]()        { return TSP { C,depot,sz };}; // the optimal bound is unknown (put 0 here)
   const auto lgf = [sz,&C](const TSP& s,DDContext)  {
      if (s.hops >= sz-1)
         return GNSet {depot};
      else 
         return (C - s.A);      
   };
   const auto stf = [depot,sz,&C,&d](const TSP& s,const int label) -> std::optional<TSP> {
      if (label==depot)
         return TSP { C,depot,sz}; //0 is the dummy value for the lb 
      else 
         return TSP { s.A | GNSet{label},label,s.hops + 1};
   };
   const auto scf = [&d](const TSP& s,int label) { // partial cost function 
      return d[s.e][label];
   };
   const auto smf = [](const TSP& s1,const TSP& s2) -> std::optional<TSP> {
      // add test: s1.A and s2.A should not be too different
      //GNSet SymmDiff = (s1.A | s2.A) - (s1.A & s2.A);
      // double sz1 = (s1.A | s2.A).size();
      // double sz2 = (s1.A & s2.A).size();
      // double r = sz1 / (1.0 + sz2);
      if (s1.e == s2.e && s1.hops == s2.hops && abs(s1.A.size() - s2.A.size()) < 5 ) // && SymmDiff.size() <= 10)
         return TSP {s1.A & s2.A, s1.e, s1.hops};
      else return std::nullopt; // return  the empty optional
   };
   const auto eqs = [depot,sz](const TSP& s) -> bool { return s.e == depot && s.hops == sz;};
   const auto local = [depot,&d,&C](const TSP& s,LocalContext) -> double {
      const auto mstVal = mst(d,C,s.A,depot,s.e,s.hops);
      const auto greedyVal = greedy(d,C,s.A,depot,s.e,s.hops);
      //std::cout << "MST:" << mstVal << " \tgreedy: " << greedyVal << "\n";
      return std::max(mstVal,greedyVal);
      //return greedyVal;
   };   

   BAndB engine(DD<TSP,Minimize<double>,
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

