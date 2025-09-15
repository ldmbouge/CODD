#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cstdio>
#include <fstream>
#include "util.hpp"

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

class MGraph {
   int _V;
   FArray<GNSet> _adj;
   GNSet _av;
public:
   MGraph(const FArray<GNSet>& adj)
      : _V(adj.size()),_adj(adj),_av(0,_V-1)       {}
   MGraph(FArray<GNSet>&& adj,const GNSet& v)
      : _V(adj.size()),_adj(std::move(adj)),_av(v) {}
   MGraph maskWith(const GNSet& s) {
      FArray<GNSet> na(_V);
      for(auto v : s) {
         na[v] = _adj[v];
         na[v].interWith(s);
      }
      return MGraph(std::move(na),s);
   }
   const auto& vertices() { return _av;}
   const auto& adjacent(int v) { return _adj[v];}
   friend std::ostream& operator<<(std::ostream& os,const MGraph& g) {
      os << "G(";
      for(auto v : g._av) 
         os << "(" << v << " -> " << g._adj[v] << ") ";      
      os << ",MASK:" << g._av << ")";
      return os;
   }
};

int t0(Instance& instance) {

   int n = instance.nv;  
   auto adj = instance.adj;

   Matrix<double,2> lambda(n,n);
   for(int i=0;i < n;i++)
      for(int j=0;j < n;j++)
         lambda[i][j] = (i)*1000 + j;
      
   GNSet S({0,1,2,5});
   MGraph m0(adj);   
      
   std::cout  << adj << "\n";
   std::cout << S << "\n";
   std::cout << m0 << "\n";

   double lagW[n]; // stack allocated array of "n" doubles 
   for(int i=0;i < n;i++) lagW[i] = 0; // init all of them to zero
   //mask
   for(auto v : S) adj[v].interWith(S);

   // update lagW
   double sl = 0.0;
   for(auto i : S) {
      for(auto j : adj[i]) {
         const auto lv = lambda[i][j];
         lagW[i] -= lv;
         lagW[j] -= lv;
         sl += lv;
      }
   }

   MGraph m1 = m0.maskWith(S);
   double lagW2[n]; // stack allocated array of "n" doubles
   std::fill(lagW2,lagW2+n,0); // initialize the whole bit to 0
   double sl2 = 0.0;
   // update lagW
   for(auto i : m1.vertices()) {
      for(auto j : m1.adjacent(i)) {
         const auto lv = lambda[i][j];
         lagW2[i] -= lv;
         lagW2[j] -= lv;
         sl2 += lv;
      }
   }
   std::cout << "M1:" <<  m1 << "\n";

   
   // print lagW
   std::cout << "lagW = [";
   for(int i=0;i < n;i++) 
      std::cout << lagW[i] << (i < n-1 ? "," : "");
   std::cout << "]\n";
   std::cout << "SL  =" << sl << "\n";
   std::cout << "SL2 =" << sl2 << "\n";
   //print
   for(auto v : S) 
      std::cout  << "S[" << v << "]=" << adj[v] << "\n";   
   return 0;   
}


int main()
{
   auto instance = readFile("../data/misp/hamming10-2.clq");
   t0(instance);
}
