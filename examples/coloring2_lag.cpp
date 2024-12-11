#include "codd.hpp"
#include <vector>
#include <set>
#include <unordered_set>
#include <algorithm>
#include <queue>


typedef FArray<unsigned short> Legal; // colors assigned to vertices. Either 0 or a value in 1..65K

struct COLOR {
   Legal s;
   int last;   // last used color
   int vtx;    // current vertex (previous vertices have all been colored)
   COLOR(const COLOR& c) : s(c.s),last(c.last),vtx(c.vtx) {}
   COLOR(COLOR&& c) : s(std::move(c.s)),last(c.last),vtx(c.vtx) {}
   COLOR(Legal&& s,int l,int v) : s(std::move(s)),last(l),vtx(v) {}
   COLOR& operator=(const COLOR& c) { s = c.s;last = c.last; vtx = c.vtx;return *this;}
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

// Lagrangian Relaxation - Start

// Function to find the vertex with the highest degree among unvisited vertices
int findHighestDegreeVertex(const FArray<GNSet>& adj, const std::vector<bool>& visited, const GNSet &S) {
   int maxDegree = -1, vertex = -1;

   for (auto i : S) {
      if (!visited[i]) { // Only consider unvisited vertices
         GNSet remAdj = adj[i]&S;
         int degree = sum(remAdj,[&visited](auto neighbor) { return !visited[neighbor];});
         if (degree > maxDegree) {
            maxDegree = degree;
            vertex = i;
         }
      }
   }
   return vertex;
}

// Function to check if an extended  set of vertices clique UNION {vertex} remains a clique
bool isClique(const FArray<GNSet>& adj, const std::vector<int>& clique, int vertex) {
   for(const auto v : clique)
      if (!adj[v].contains(vertex))
         return false;
   return true;
}

// Function to find a maximal clique starting from the highest-degree vertex
std::vector<int> findMaximalClique(const FArray<GNSet>& adj, std::vector<bool>& visited, const GNSet &S) {
   std::vector<int> clique;
   
   // Start with the vertex of the highest degree
   int start = findHighestDegreeVertex(adj, visited, S);
   if (start == -1) return clique; // No more vertices to process
   
   clique.push_back(start);
   visited[start] = true;
   
   // Grow the clique by adding the highest-degree neighbor that maintains the clique property
   while (true) {
      int bestCandidate = -1, maxDegree = -1;
      // for (int i = 0; i < n; i++) {
      for (auto i : S ){
         if (!visited[i] && isClique(adj, clique, i)) {
            if (!adj[start].contains(i)) {
               std::cout << "ERROR in findMaximalClique" << std::endl;
               exit(1);
            }
            GNSet remAdj = adj[i]&S;
            int degree = sum(remAdj,[&visited](auto neighbor) { return !visited[neighbor];});
            if (degree > maxDegree) {
               maxDegree = degree;
               bestCandidate = i;
            }
         }
      }
      if (bestCandidate == -1) break; // No more vertices can be added
      clique.push_back(bestCandidate);
      visited[bestCandidate] = true;
   }
   return clique;
}

// Function to compute a maximal clique partition
struct  clqPartition {
   int nc;  // number of cliques
   std::vector<int> CliqueOfVertex;
   std::vector<std::vector<int>> Cliques;
   friend std::ostream& operator<<(std::ostream& os,const clqPartition& ClqPart) {
      os << "There are " << ClqPart.nc << " cliques\n";
      for (int i = 0; i < (int)ClqPart.Cliques.size(); ++i) {
         os << "Clique " << i << ": ";
         for (int v : ClqPart.Cliques[i])
            os << v << " ";
         os << std::endl;
      }
      os << "clique per vertex:" << std::endl;
      for (int v=0; v<(int)ClqPart.CliqueOfVertex.size(); v++)
         os << v << ": " << ClqPart.CliqueOfVertex[v] << std::endl;
      return os;
   }
};

clqPartition maximalCliquePartition(const FArray<GNSet>& adj, const GNSet &S) {
   const int n = adj.size();
   std::vector<bool> visited(n, false);
   std::vector<std::vector<int>> cliquePartition;

   while (true) {
      // Find a maximal clique
      std::vector<int> clique = findMaximalClique(adj, visited, S);
      if (clique.empty()) break; // No more cliques to find
      cliquePartition.push_back(clique); // Add the clique to the partition
   }

   clqPartition ClqP;
   ClqP.nc = cliquePartition.size();
   ClqP.CliqueOfVertex = std::vector<int>(n, -1);
   for (int i=0; i<(int)cliquePartition.size(); i++)
      for (int v : cliquePartition[i])
         ClqP.CliqueOfVertex[v] = i;
   ClqP.Cliques = cliquePartition;
   return ClqP;
}

void exportCliquePartition(const std::vector<std::vector<int>>& cliques, const std::string &fileName) {
   std::ofstream outFile(fileName.c_str()); // Open file for writing
   if (!outFile.is_open()) {
      std::cerr << "Error: Unable to open file " << fileName << " for writing." << std::endl;
      return;
   }
   // Write the number of cliques
   outFile << "n " << cliques.size() << "\n";
   // Write each clique
   for (size_t i = 0; i < cliques.size(); ++i) {
      outFile << "c";
      for (size_t j = 0; j < cliques[i].size(); ++j)
         outFile << " " << cliques[i][j] + 1;  // add 1 to match DIMACS format
      outFile << "\n";
   }
   outFile.close(); // Close the file
   std::cout << "Cliques successfully exported to " << fileName << std::endl;
}

// Solve clique subproblem: Select vertex with largest weight in each clique
double solveCliqueSubproblem(const std::vector<std::vector<int>>& cliques, const std::vector<double>& weight, std::vector<int> &x, const GNSet &S) {
   double primalSolution = 0.0;
   for(const auto& clique : cliques) {
      double maxWeight = 0; // only include positive weights for MISP
      int maxV = -1;
      for (int i=0; i<(int)clique.size(); i++) {
         int v = clique[i];
         if (S.contains(v) && (weight[v] > maxWeight)) {
            maxWeight = weight[v];
            maxV = v;
         }
      }
      primalSolution += maxWeight;
      if (maxV >= 0)
         x[maxV] = 1;
      // else do nothing: there is no vertex with positive weight.
   }
   return primalSolution;
}

double CliqueBound(const FArray<GNSet>& adj, const GNSet& S) {
   auto cliquePartition = maximalCliquePartition(adj, S);
   double maxC = 0.0;
   // int cnt = 0;
   for(const auto& clique : cliquePartition.Cliques) {
      // cnt += clique.size();
      if ((int)clique.size() > maxC) maxC = (double)clique.size();
   }
   // assert(cnt==S.size());
   return maxC;
}

// Lagrangian Relaxation - End


int main(int argc,char* argv[])
{
   if (argc < 2) {
      std::cout << "usage: coloring <fname> <width> <LB>\n";
      std::cout << "where optional <LB> = 0 (do nothing) \n";
      std::cout << "                    = 1 (maximal clique) \n";
      std::cout << "                    = 2 (Lagrangian; default) \n";
      std::cout << "                    = 3 (Lagrangian recomputed) \n";
      exit(1);
   }
   std::cout << "ARGC:" << argc << "\n";
   const char* fName = argv[1];
   const int w = argc>=3 ? atoi(argv[2]) : 64;
   int LBopt = std::clamp(argc>=4 ? atoi(argv[3]) : 1,0,3);

   std::cout << "FILE:" << fName << "\n";
   std::cout << "Width=" << w << "\n";
   std::cout << "LBopt = " << LBopt << std::endl;

   int UB = -1;
   Instance instance = readPyFile(fName, UB);
   std::cout << "read instance:" << instance.nv << " " << instance.ne << "\n";
   const GNSet ns = instance.vertices();
   const std::set<GE> es = instance.edges;
   const FArray adj = instance.adj;
   const int K = ns.size();
   std::cout << "NS:" << ns << " |NS|=" << K << "\n";
   
   if (UB < 0) UB = K;
   std::cout << "upper bound is " << UB << "\n";
   Bounds bnds([&es](const std::vector<int>& inc)  {
      bool aBad = false;
      for(const auto& e : es) {
         if (inc[e.a] == inc[e.b]) {
            std::cout << e << " COL:" << inc << "\n";
            assert(false);
         }
         aBad = aBad || (inc[e.a] == inc[e.b]);
         assert(!aBad);
      }
      std::cout << "CHECKER is " << !aBad << "\n";
   });
   
   // Lagrangian Relaxation
   std::cout << "Computing clique partition" << "\n";
   auto cliquePartition = maximalCliquePartition(instance.adj, ns);
   // std::cout << cliquePartition;
   // std::string fOutName = std::string(fName) + ".clique_partition";
   // exportCliquePartition(cliquePartition.Cliques,fOutName);
   std::cout << "Number of cliques = " << cliquePartition.Cliques.size() << "\n";
   
   double cb = CliqueBound(instance.adj, ns);
   std::cout << "lower bound (maximum clique) = " << cb << std::endl;
   
   // [LDM] : changed labels to go to UB (rather than K) (halved the runtime)
   const auto labels = setFrom(std::views::iota(1,UB+1));     // using a plain set for the labels
   const auto init = [K]()     { return COLOR { Legal(K,0),0,0 };};
   const auto target = [K]()   { return COLOR { Legal{},0,K};};
   const auto lgf = [K,&labels,&bnds,&adj](const COLOR& s) -> GNSet {
      if (s.vtx >= K+1)
         return GNSet(); 
      GNSet valid(labels);
      for(auto a : adj[s.vtx])
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
   const auto scf = [](const COLOR& s,int label) { return std::max(0,label -  s.last);};
   const auto smf = [](const COLOR& s1,const COLOR& s2) -> std::optional<COLOR> {
      if (s1.last == s2.last && s1.vtx == s2.vtx) {
         Legal B(s1.s);
         for(auto i=0;i < s1.vtx + 1;i++) 
            B[i] = (s1.s[i] == s2.s[i]) ? s1.s[i] : 0;
         return COLOR { std::move(B) , s1.last, s1.vtx};
      }
      else return std::nullopt; // return  the empty optional
   };
   const auto eqs = [K](const COLOR& s) -> bool { return s.vtx == K;};
   
   std::function<double(const COLOR&,LocalContext)> local;
   switch(LBopt) {
      case 0: local = [](const COLOR& s,LocalContext) -> double {
         return 0.0;
      }; break;
      case 1: local = [&instance,&K,&bnds](const COLOR& s,LocalContext) -> double {
         GNSet S(s.vtx,K-1); // uncolored vertices: from current vertex s.vtx until the last vertex K-1.
         if (S.size()==0) {
            // std::cout << "ERROR: no more vertices left in localBound" << std::endl;
            return 0.0;
         } else if (S.size() <= bnds.getDual()) {  // no chance to find a larger dual bound
            return bnds.getDual();
         }
         double cb = CliqueBound(instance.adj, S);
         //double lb = std::max(cb, bnds.getDual());
         //double ub = bnds.getPrimal();
         //if (cb > bnds.getDual()) {
         //   std::cout << "\t--> (current dual = " << bnds.getDual() << ".)   new dual heur: " << cb << " dual: " << lb << " primal: " << ub << " -- better bound\n";
         //}
         // return std::max(cb, bnds.getDual());
         return bnds.getDual();
      };break;
      default:
         throw std::invalid_argument("Invalid LBopt value! Must be 0, 1, or 2.");
   }
   

   BAndB engine(DD<COLOR,Minimize<double>, // to minimize
                decltype(target),
                decltype(lgf),
                decltype(stf),
                decltype(scf),
                decltype(smf),
                decltype(eqs)
                >::makeDD(init,target,lgf,stf,scf,smf,eqs,labels,local),w);
   engine.setTimeLimit([](double elapsed) { return elapsed >= 120000;});
   engine.search(bnds);
   return 0;
}

