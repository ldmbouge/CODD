#include "codd.hpp"

struct MISP {
   GNSet sel;
   int   n;
   friend std::ostream& operator<<(std::ostream& os,const MISP& m) {
      return os << "<" << m.sel << ',' << m.n << ',' << ">";
   }
};

template<> struct std::equal_to<MISP> {
   bool operator()(const MISP& s1,const MISP& s2) const {
      return s1.n == s2.n && s1.sel == s2.sel;         
   }
};

template<> struct std::hash<MISP> {
   std::size_t operator()(const MISP& v) const noexcept {
      return std::rotl(std::hash<GNSet>{}(v.sel),32) ^ std::hash<int>{}(v.n);
   }
};

struct GE {
   int a,b;
   friend bool operator==(const GE& e1,const GE& e2) {
      return e1.a == e2.a && e1.b == e2.b;
   }
   friend std::ostream& operator<<(std::ostream& os,const GE& e) {
      return os  << e.a << "-->" << e.b;
   }
};


struct Instance {
   int nv;
   int ne;
   std::vector<GE> edges;
   FArray<GNSet> adj;
   Instance() : adj(0) {}
   Instance(Instance&& i) : nv(i.nv),ne(i.ne),edges(std::move(i.edges)) {}
   GNSet vertices() {
      return setFrom(std::views::iota(0,nv));
   }
   auto getEdges() const noexcept { return edges;}
   void convert() {
      adj = FArray<GNSet>(nv+1);
      for(const auto& e : edges) {
         adj[e.a].insert(e.b);
         adj[e.b].insert(e.a);
      }         
   }
};

Instance readFile(const char* fName)
{
   Instance i;
   using namespace std;
   ifstream f(fName);
   while (!f.eof()) {
      char c;
      f >> c;
      if (f.eof()) break;
      switch(c) {
         case 'c': {
            std::string line;
            std::getline(f,line);
         }break;
         case 'p': {
            string w;
            f >> w >> i.nv >> i.ne;
         }break;
         case 'e': {
            GE edge;
            f >> edge.a >> edge.b;
            edge.a--,edge.b--;      // make it zero-based
            assert(edge.a >=0);
            assert(edge.b >=0);
            i.edges.push_back(edge);
         }break;
      }
   }
   f.close();
   i.convert();
   return i;
}

int main(int argc,char* argv[])
{
   // using STL containers for the graph
   if (argc < 2) {
      std::cout << "usage: coloring <fname> <width>\n";
      exit(1);
   }
   const char* fName = argv[1];
   const int w = argc==3 ? atoi(argv[2]) : 64;
   auto instance = readFile(fName);
   
   const GNSet ns = instance.vertices();
   const int top = ns.size();
   const std::vector<GE> es = instance.getEdges();
   std::cout << "TOP=" << top << "\n";

   Bounds bnds([&es](const std::vector<int>& inc)  {
      bool ok = true;    
      for(const auto& e : es) {         
         bool v1In = (e.a < (int)inc.size()) ? inc[e.a] : false;
         bool v2In = (e.b < (int)inc.size()) ? inc[e.b] : false;
         if (v1In && v2In) {
            std::cout << e << " BOTH ep in inc: " << inc << "\n";
            assert(false);
         }
         ok &= !(v1In && v2In);
      }
      std::cout << "CHECKER is " << ok << "\n";
   });
   const auto labels = ns | GNSet { top };     // using a plain set for the labels
   std::vector<int> weight(ns.size()+1);
   weight[ns.size()] = 0;
   for(auto v : ns) weight[v] = 1;
   auto neighbors = instance.adj;

   const auto myInit = [top]() {   // The root state
      GNSet U = {}; // std::views::iota(1,top) | std::ranges::to<std::set>();
      for(auto i : std::views::iota(0,top))
         U.insert(i);
      return MISP { U , 0};
   };
   const auto myTarget = [top]() {    // The sink state
      return MISP { GNSet {},top};
   };
   const auto lgf = [](const MISP& s)  {
      return Range::close(0,1);
   };
   auto myStf = [top,&neighbors](const MISP& s,const int label) -> std::optional<MISP> {
      if (s.n >= top) {
         if (s.sel.empty()) 
            return MISP { GNSet {}, top};
         else return std::nullopt;
      } else {
         if (!s.sel.contains(s.n) && label) return std::nullopt; // we cannot take n (label==1) if not legal.
         GNSet out = s.sel;
         out.remove(s.n);   // remove n from state
         if (label) out.diffWith(neighbors[s.n]); // remove neighbors of n from state (when taking n -- label==1 -- )    
         const bool empty = out.empty();  // find out if we are done!
         return MISP { std::move(out),empty ? top : s.n + 1}; // build state accordingly
      }
   };
   const auto scf = [weight](const MISP& s,int label) { // cost function 
      return label * weight[s.n];
   };
   const auto smf = [](const MISP& s1,const MISP& s2) -> std::optional<MISP> { // merge function
      return MISP {s1.sel | s2.sel,std::min(s1.n,s2.n)};
   };
   const auto eqs = [](const MISP& s) -> bool {
      return s.sel.size() == 0;
   };
   const auto local = [&weight](const MISP& s) -> double {
      return sum(s.sel,[&weight](auto v) { return weight[v];});
   };      

   BAndB engine(DD<MISP,Maximize<double>, // to maximize
                //decltype(myInit), 
                decltype(myTarget), // MISP(*)(),
                decltype(lgf),
                decltype(myStf),
                decltype(scf),
                decltype(smf),
                decltype(eqs),
                decltype(local)
                >::makeDD(myInit,myTarget,lgf,myStf,scf,smf,eqs,labels,local),w);
   engine.search(bnds);
   return 0;
}

