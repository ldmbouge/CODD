#include "dd.hpp"
#include "util.hpp"
#include <concepts>
#include <iostream>
#include <fstream>
#include <set>
#include <optional>
#include <ranges>
#include <algorithm>
#include <map>
#include "search.hpp"

struct MISP {
   GNSet sel;
   friend std::ostream& operator<<(std::ostream& os,const MISP& m) {
      return os << "<" << m.sel << ">";
   }
};

template<> struct std::equal_to<MISP> {
   bool operator()(const MISP& s1,const MISP& s2) const {
      return s1.sel == s2.sel;
   }
};

template<> struct std::hash<MISP> {
   std::size_t operator()(const MISP& v) const noexcept {
      return std::hash<GNSet>{}(v.sel);
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
         std::cout << "E: " << e << "\n";
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
            std::cout << edge << "\n";
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
   std::cout << "EDGES:" << es << "\n";
   std::cout << "VERTICES:" << ns << "\n";
   std::cout << "TOP=" << top << "\n";

   Bounds bnds;
   const auto labels = ns | GNSet { top };     // using a plain set for the labels
   std::vector<int> weight(ns.size()+1);
   weight[ns.size()] = 0;
   for(auto v : ns) weight[v] = 1;   
   std::map<int,GNSet> neighbors {};  // computing the neigbhors using STL (not pretty)
   for(int i : ns) {
      neighbors[i] = filter(ns,[i,&es](auto j) {
         return j==i || member(es,[e1=GE {i,j},e2=GE {j,i}](auto e) { return e==e1 || e==e2;});
      });
      std::cout << i << " -> " << neighbors[i] << std::endl;
   }
   const auto myInit = [top]() {   // The root state
      GNSet U = {}; // std::views::iota(1,top) | std::ranges::to<std::set>();
      for(auto i : std::views::iota(0,top))
         U.insert(i);
      std::cout << "ROOT:" << U << "\n";
      return MISP { U };
   };
   const auto myTarget = []() {    // The sink state
      return MISP { GNSet {} };
   };
   const auto lgf = [top](const MISP& s) -> Range {
      return Range::close(1,top);
   };
   auto myStf = [top,&neighbors](const MISP& s,const int label) -> std::optional<MISP> {
      if (label == top)
         return MISP { GNSet {}}; // head to sink
      else if (s.sel.contains(label)) {
         return MISP {
            //s.sel \ neighbors[label]
            filter(s.sel,[label,nl = neighbors[label]](int i) {
               return !nl.contains(i);
            })
         };
      } else return std::nullopt;  // return the empty optional 
   };
   const auto scf = [top,weight](const MISP& s,int label) { // cost function 
      return weight[label];
   };
   const auto smf = [](const MISP& s1,const MISP& s2) -> std::optional<MISP> { // merge function
      using namespace std;
      return MISP {s1.sel & s2.sel};

      if (max(s1.sel) == max(s2.sel))
         return MISP {s1.sel | s2.sel};
      else return std::nullopt; // return  the empty optional
   };
   const auto eqs = [](const MISP& s) -> bool {
      return s.sel.size() == 0;
   };

   BAndB engine(DD<MISP,Maximize<double>, // to maximize
                //decltype(myInit), 
                decltype(myTarget), // MISP(*)(),
                decltype(lgf),
                decltype(myStf),
                decltype(scf),
                decltype(smf),
                decltype(eqs)                   
                >::makeDD(myInit,myTarget,lgf,myStf,scf,smf,eqs,labels),w);
   engine.search(bnds);
   return 0;
}

