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
      return (std::hash<Legal>{}(v.s) << 24) |
         (std::hash<int>{}(v.last) << 16) |
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
   Instance() {}
   Instance(Instance&& i) : nv(i.nv),ne(i.ne),edges(std::move(i.edges)) {}
   GNSet vertices() {
      return setFrom(std::views::iota(0,nv+1));
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
            i.edges.insert(edge);
         }break;
      }
   }
   f.close();
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
   const auto stf = [K,es](const COLOR& s,const int label) -> std::optional<COLOR> {
      if (s.vtx < K && s.s[s.vtx].contains(label) && label <= s.last+1) {
         Legal B = s.s;
         for(auto vIdx = s.vtx+1;vIdx < (int)B.size();vIdx++)
            if ((es.contains(GE {s.vtx,vIdx}) || es.contains(GE {vIdx,s.vtx})) &&
                s.s[vIdx].contains(label))
               B[vIdx].remove(label);
         B[s.vtx] = GNSet {label};
         if (s.vtx+1 == K)
            return COLOR { Legal {},0, K };
         else
            return COLOR { std::move(B), std::max(label,s.last), s.vtx + 1};
      } 
      else return std::nullopt;  // return the empty optional 
   };
   const auto scf = [](const COLOR& s,int label) { // partial cost function 
      return std::max(0,label -  s.last);
   };
   const auto smf = [K](const COLOR& s1,const COLOR& s2) -> std::optional<COLOR> {
      if (s1.last == s2.last && s1.vtx == s2.vtx) {
         Legal B(s1.s.size(),GNSet {});
         for(auto i=0;i < K;i++) 
            B[i] = s1.s[i] | s2.s[i];         
         return COLOR {B , s1.last, s1.vtx};
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

