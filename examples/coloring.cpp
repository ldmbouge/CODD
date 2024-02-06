#include "dd.hpp"
#include "util.hpp"
#include "search.hpp"
#include <concepts>
#include <iostream>
#include <fstream>
#include <set>
#include <optional>
#include <algorithm>
#include <map>

//typedef std::vector<GNSet> Legal;
typedef FArray<GNSet> Legal;

struct COLOR {
   Legal s;
   int last;
   int vtx;
   COLOR(const COLOR& c) : s(c.s),last(c.last),vtx(c.vtx) {}
   COLOR(COLOR&& c) : s(std::move(c.s)),last(c.last),vtx(c.vtx) {}
   COLOR(Legal&& s,int l,int v) : s(std::move(s)),last(l),vtx(v) {}
   friend std::ostream& operator<<(std::ostream& os,const COLOR& m) {
      return os << "<" << m.s << ',' << m.last << ',' << m.vtx << ">";
   }
};

template<> struct std::equal_to<COLOR> {
   constexpr bool operator()(const COLOR& s1,const COLOR& s2) const {
      return s1.last == s2.last && s1.vtx==s2.vtx && s1.s == s2.s;
   }
};

template<> struct std::hash<COLOR> {
   std::size_t operator()(const COLOR& v) const noexcept {
      return (std::hash<Legal>{}(v.s) << 16) ^
         (std::hash<int>{}(v.last) << 8) ^
         std::hash<int>{}(v.vtx);
   }
};

struct GE {
   int a,b;
   friend bool operator==(const GE& e1,const GE& e2) {
      return e1.a == e2.a && e1.b == e2.b;
   }
   friend bool operator<(const GE& e1,const GE& e2) {
      return e1.a < e2.a || (e1.a == e2.a && e1.b < e2.b);
   }
   friend std::ostream& operator<<(std::ostream& os,const GE& e) {
      return os  << e.a << "-->" << e.b;
   }
};

struct Instance {
   int nv;
   int ne;
   std::set<GE> edges;
   FArray<GNSet> adj;
   Instance() : adj(0) {}
   Instance(Instance&& i) : nv(i.nv),ne(i.ne),edges(std::move(i.edges)) {}
   GNSet vertices() {
      return setFrom(std::views::iota(0,nv+1));
   }
   void convert() {
      adj = FArray<GNSet>(nv);
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
            std::cout << edge << "\n";
            assert(edge.a >=0);
            assert(edge.b >=0);
            i.edges.insert(edge);
         }break;
      }
   }
   f.close();
   i.convert();
   return i;
}

int main(int argc,char* argv[])
{
   if (argc < 2) {
      std::cout << "usage: coloring <fname> <width>\n";
      exit(1);
   }
   const char* fName = argv[1];
   const int w = argc==3 ? atoi(argv[2]) : 64;
   Instance instance = readFile(fName);
   std::cout << "read instance:" << instance.nv << " " << instance.ne << "\n";
   std::cout << instance.edges << "\n";
   std::cout << "Width=" << w << "\n";
   // using STL containers for the graph
   const GNSet ns = instance.vertices();
   const std::set<GE> es = instance.edges;
   const FArray adj = instance.adj;
   const int K = ns.size();
   Bounds bnds;
   const auto labels = setFrom(std::views::iota(1,K+1));     // using a plain set for the labels
   const auto init = [ns,labels]() {   // The root state
      Legal A(ns.size(), GNSet {});      
      for(auto v : ns)
         A[v] = labels;
      return COLOR { std::move(A),0,0 };
   };
   const auto target = [K]() {    // The sink state
      return COLOR { Legal{},0,K};
   };
   const auto stf = [K,&adj](const COLOR& s,const int label) -> std::optional<COLOR> {
      if (s.vtx < K && label <= s.last+1 && s.s[s.vtx].contains(label)) {
         if (s.vtx+1 == K)
            return COLOR { Legal {},0, K };
         Legal B = s.s;
         for(auto vIdx : adj[s.vtx])
            if (s.s[vIdx].contains(label))
               B[vIdx].remove(label);
         /*
         for(auto vIdx = s.vtx+1;vIdx < K;vIdx++)
            if (adj[s.vtx].contains(vIdx) && s.s[vIdx].contains(label))
               B[vIdx].remove(label);
         */
         B[s.vtx] = GNSet {label};
         return COLOR { std::move(B), std::max(label,s.last), s.vtx + 1};
      } 
      else return std::nullopt;  // return the empty optional 
   };
   const auto scf = [](const COLOR& s,int label) { // partial cost function 
      return std::max(0,label -  s.last);
   };
   const auto smf = [K](const COLOR& s1,const COLOR& s2) -> std::optional<COLOR> {
      if (s1.last == s2.last && s1.vtx == s2.vtx) {
         Legal B(s1.s);
         for(auto i=0;i < K;i++) 
            B[i].unionWith(s2.s[i]);         
         return COLOR { std::move(B) , s1.last, s1.vtx};
      }
      else return std::nullopt; // return  the empty optional
   };
   const auto eqs = [K](const COLOR& s) -> bool {
      return s.vtx == K;
   };

   std::cout << "LABELS:" << labels << "\n";

   BAndB engine(DD<COLOR,std::less<double>, // to minimize
                ///decltype(init), 
                decltype(target), 
                decltype(stf),
                decltype(scf),
                decltype(smf),
                decltype(eqs)
                >::makeDD(init,target,stf,scf,smf,eqs,labels),w);
   engine.search(bnds);
   return 0;
}

