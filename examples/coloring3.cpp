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

typedef FArray<unsigned short> Legal; // colors assigned to vertices. Either 0 or a value in 1..65K

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
      return setFrom(std::views::iota(0,nv));
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

Instance readPyFile(const char* fName, int& ub)
{
   Instance i;
   using namespace std;
   int nbe = 0;
   ifstream f(fName);
   while (!f.eof()) {
      char c;
      f >> c;
      if (f.eof()) break;
      switch(c) {
         case 'N': {
            string w;
            f >> w; // read '='
            f >> i.nv;
         }break;
         case 'E': {
            string w;
            char tmp, last;
            f >> w;    // read '='
            f >> last; // read '{'
            while (last != '}') {
               int u, v;
               f >> tmp >> u >> tmp >> v >> tmp >> last;
               GE edge;
               edge.a = u, edge.b = v;
               //std::cout << edge << "\n";
               assert(edge.a >=0);
               assert(edge.b >=0);
               i.edges.insert(edge);
               nbe++;
            }
         }break;
         case 'K': {
            string w;
            f >> w;
            f >> ub;
         }break;
      }
   }
   i.ne = nbe;
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
   int UB = -1;
   std::cout << "FILE:" << fName << "\n";
   //Instance instance = readFile(fName);
   Instance instance = readPyFile(fName, UB);
   std::cout << "read instance:" << instance.nv << " " << instance.ne << "\n";
   // std::cout << instance.edges << "\n";
   std::cout << "Width=" << w << "\n";
   // using STL containers for the graph
   const GNSet ns = instance.vertices();
   const std::set<GE> es = instance.edges;
   const FArray adj = instance.adj;
   const int K = ns.size();
   std::cout << "NS:" << ns << " |NS|=" << K << "\n";
   
   if (UB < 0) UB = K;
   std::cout << "upper bound is " << UB << "\n";
   Bounds bnds([&es,K](const std::vector<int>& inc)  {
      bool aBad = false;
      //std::cout << "INC: " << inc << "\n";
      for(const auto& e : es) {
         //std::cout << "EDGE:" << e << "\n";
         if (inc[e.a] == inc[e.b]) {
            std::cout << e << " COL:" << inc << "\n";
            assert(false);
         }
         aBad = aBad || (inc[e.a] == inc[e.b]);
         assert(!aBad);
      }
      std::map<int,int> h;
      for(int v = 0;v < K;v++) 
         h[inc[v]] = h.contains(inc[v]) ? h[inc[v]] + 1 : 1;
      for(const auto& e : h) 
         std::cout << "(" << e.first << " : " << e.second << ") ";
      std::cout << "\n";
      std::cout << "CHECKER is " << !aBad << "\n";
   });
   //bnds.setPrimal(UB);
   // [LDM] : changed labels to go to UB (rather than K) (halved the runtime)
   // Well, UB is given in the file, but it actually tends to be the optimal ;-) So it's cheating a bit
   // to start with that value (it saves memory, it does not provide a witness). We can switch back to K+1
   // rather than UB+1. Or compute an actual greedy coloring at first to get an UB. For Some benchmark, that
   // would be far less than K.
   const auto labels = setFrom(std::views::iota(1,UB+1));     // using a plain set for the labels
   const auto init = [K,labels]() {   // The root state
      Legal A(K, 0);      // Noboby is colored yet (0 to everyone)
      return COLOR { std::move(A),0,0 };
   };
   const auto target = [K]() {    // The sink state
      return COLOR { Legal{},0,K};
   };
   const auto lgf = [K,&labels,&bnds,&adj](const COLOR& s) -> GNSet {
      if (s.vtx >= K+1)
         return GNSet(); 
      GNSet valid(labels);
      for(auto a : adj[s.vtx])  // [WvH] We could limit adj[i] to indices < i because of the fixed variale ordering, right? [LDM]. Yes, of course!
         if (s.s[a] != 0)
            valid.remove(s.s[a]);
      auto ub = std::min({(int)bnds.getPrimal(),s.last+1});
      valid.removeAbove(ub);
      return valid;
   };
   const auto stf = [K](const COLOR& s,const int label) -> std::optional<COLOR> {
      if (s.vtx+1 == K)
         return COLOR { Legal {},0, K };
      Legal B = s.s;
      B[s.vtx] = label;
      return COLOR { std::move(B), std::max(label,s.last), s.vtx + 1};
   };
   const auto scf = [](const COLOR& s,int label) { // partial cost function 
      return std::max(0,label -  s.last);
   };
   const auto smf = [](const COLOR& s1,const COLOR& s2) -> std::optional<COLOR> { 
      if (s1.vtx == s2.vtx && s1.last == s2.last) {
         Legal B(s1.s);
         int cnt = 0;  // number of shared labels
         for(auto i=0;i < s1.vtx+1;i++) {
            B[i] = (s1.s[i] == s2.s[i]) ? s1.s[i] : 0;
            if (s1.s[i] == s2.s[i]) cnt+=1;
         }
         if (cnt >= 0.5*(s1.vtx)) 
            return COLOR { std::move(B) , s1.last, s1.vtx};
         else return std::nullopt; // colorings are too dissimilar to merge
      }
      else return std::nullopt; // can't merge those
   };
   const auto eqs = [K](const COLOR& s) -> bool {
      return s.vtx == K;
   };

   std::cout << "LABELS:" << labels << "\n";

   BAndB engine(DD<COLOR,Minimize<double>, // to minimize
                ///decltype(init), 
                decltype(target),
                decltype(lgf),
                decltype(stf),
                decltype(scf),
                decltype(smf),
                decltype(eqs)
                >::makeDD(init,target,lgf,stf,scf,smf,eqs,labels),w);
   engine.setTimeLimit([](double elapsed) { return elapsed >= 600000;});
   engine.search(bnds);
   return 0;
}

