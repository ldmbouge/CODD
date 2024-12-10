#include "codd.hpp"
#include <vector>
#include <set>
#include <unordered_set>
#include <algorithm>
#include <queue>

using namespace std;

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

// Function to find the vertex with the highest degree among unvisited vertices
int findHighestDegreeVertex(const FArray<GNSet>& adj, const vector<bool>& visited, const GNSet &S) {
   int maxDegree = -1, vertex = -1;

   for (auto i : S) {
//   GNSet remAdj = adj[i]&S;
   //   for (int i = 0; i < (int)adj.size()-1; ++i) { // adj.size - 1 : omit last (top) element
   //for (auto i : remAdj ) {
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
bool isClique(const FArray<GNSet>& adj, const vector<int>& clique, int vertex) {
   for(const auto v : clique)
      if (!adj[v].contains(vertex))
         return false;
   return true;
}

// Function to find a maximal clique starting from the highest-degree vertex
vector<int> findMaximalClique(const FArray<GNSet>& adj, vector<bool>& visited, const GNSet &S) {
   vector<int> clique;
   //   const int n = adj.size()-1; // omit last entry (top)
   
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
               cout << "ERROR in findMaximalClique" << endl;
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
   vector<int> CliqueOfVertex;
   vector<vector<int>> Cliques;
   friend std::ostream& operator<<(std::ostream& os,const clqPartition& ClqPart) {
      os << "There are " << ClqPart.nc << " cliques\n";
      for (int i = 0; i < (int)ClqPart.Cliques.size(); ++i) {
         os << "Clique " << i << ": ";
         for (int v : ClqPart.Cliques[i])
            os << v << " ";
         os << endl;
      }
      os << "clique per vertex:" << endl;
      for (int v=0; v<(int)ClqPart.CliqueOfVertex.size(); v++)
         os << v << ": " << ClqPart.CliqueOfVertex[v] << endl;
      return os;
   }
};

clqPartition maximalCliquePartition(const FArray<GNSet>& adj, const GNSet &S) {
   const int n = adj.size()-1;  // omit the last entry (representing 'top')
   vector<bool> visited(n, false);
   vector<vector<int>> cliquePartition;

   while (true) {
      // Find a maximal clique
      vector<int> clique = findMaximalClique(adj, visited, S);
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
double solveCliqueSubproblem(const vector<vector<int>>& cliques, const std::vector<double>& weight, std::vector<int> &x, const GNSet &S) {
   double primalSolution = 0.0;
   // already initialized x to 0 when passed to this function.
   //for (int i=0; i<x.size(); i++) { x[i] := 0; }
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

// Define a default hash function for std::pair<int, int>
template<> struct std::hash<std::pair<int,int>> {
   std::size_t operator()(const std::pair<int, int>& p) const {
      return std::hash<int>()(p.first) ^ (std::hash<int>()(p.second) << 1);
   }
};

// Define a custom hash function
struct PairHash {
    size_t operator()(const std::pair<int, int>& pair) const {
        auto hash1 = std::hash<int>{}(pair.first);
        auto hash2 = std::hash<int>{}(pair.second);
        return hash1 ^ (hash2 << 1); // Combine the hashes
    }
};
using PairMap = std::unordered_map<std::pair<int, int>, double, PairHash>;// using PairMap = std::unordered_map<std::pair<int, int>, double>; // use the default hash

// Update Lagrangian multipliers
bool updateMultipliers(PairMap& lambdas, const vector<int>& x, double alpha, const std::unordered_set<std::pair<int, int>> &edgeSet) {
   bool foundViolation = false;
   // cout << "entering updateMultipliers with edgeSet.size() = " << edgeSet.size() << endl;
   for (const auto& [i,j]  : edgeSet) {
      if (lambdas.count({i,j}) > 0)
         lambdas[{i, j}] = std::max(0.0, lambdas[{i, j}] + alpha*(x[i] + x[j] - 1));
      else
         lambdas[{i, j}] = std::max(0.0, alpha*(x[i] + x[j] - 1));
      foundViolation |= (x[i] + x[j] > 1);
      if (lambdas[{i, j}] == 0.0)
         lambdas.erase({i,j}); // eliminate this lambda from being used in the Lagrangian iteration
   }
   return (!foundViolation); // return whether a primal solution was found
}

void printLagrangianInfo(const PairMap &lambdas, const std::vector<double>& LagWeights, const vector<int> &x,
                         int k, double alpha, double primalSolution) {
    cout << "** Lagrangian iteration " << k << ": " << endl;
    cout << "   Multipliers = ";
    for (const auto& [edge, lambda] : lambdas) {
        cout << "(" << edge.first << "," << edge.second << ") = " << lambda << ", ";
    }
    cout << "\n";
    cout << "   LagWeights = " << LagWeights << "\n";
    cout << "   Solution    = " << x << "\n";
    cout << "   Intermediate Lagrangian bound = " << primalSolution << "\n";
    cout << "   alpha      = " << alpha << "\n";
}

// Subgradient descent for Lagrangian dual
double lagrangianDual(const FArray<GNSet>& adj, const std::vector<double>& weight,
                      const clqPartition& ClqPart, int maxIter, double alpha0,
                      PairMap &lambdas, const bool reOpt, const GNSet &S) {

   // reOpt = true means we reOptimize the Lagrangian Dual using state S.
   // We can use the existing lambdas to initialize the search.

   // std::cout << "Computing Lagrangian Dual with reOpt = " << reOpt << " and S = " << S << std::endl;

   std::set<std::pair<int, int>> edgeSet;  // the set of edges between Cliques

   // Define the appropriate lambdas: these will be dynamically updated and only defined for edges with non-zero lambda.
   // Here, we initialize them for all edges with i < j to eliminate symmetry.
   for (int c = 0; c < (int)ClqPart.Cliques.size(); c++) {
      for (auto i : ClqPart.Cliques[c]) {
         if (S.contains(i)) {
            for (auto j : adj[i]) {
               if (S.contains(j) && (ClqPart.CliqueOfVertex[j] != c) && (i<j)) {
                  // lambdas[{i,j}] = 0.0;
                  edgeSet.insert({i,j});
               }
               else if (lambdas.count({i,j}) > 0) {
                  lambdas.erase({i,j});
               }
            }
         }
      }
   }

   if (edgeSet.size() == 0) {
      // std::cout << "Cliques are disconnected.  Can report dual immediately: the number of cliques." << std::endl;
      return 1.0*(int)ClqPart.Cliques.size();
   }

   // // Using small nonzero initial values for Lambda can be helpful.
   double init = 1.0*(int)ClqPart.Cliques.size()/edgeSet.size(); // the number of cliques is an upper bound; divide by #edges.
   if (!reOpt) {
      for (const auto& edge : edgeSet)
         lambdas[edge] = 0.01*init; // large init value is often too aggressive
   }

   double dualBound = numeric_limits<double>::max();
   std::vector<double> LagWeights = weight;  // Need a local copy because these will be updated

   double alpha = alpha0;
   int k = 0;
   int unchanged_iters = 0; // to identify stalling (objective doesn't change)
   clock_t lagStart = clock();
   clock_t outStart = clock();
   for (k = 0; k < maxIter; ++k) {
      // Update the weights
      LagWeights = weight;
      // update only those lambdas that are in the active edgeSet (needed when reoptimizing)
      for (const auto& [edge, value] : lambdas){
        LagWeights[edge.first] -= value;
        LagWeights[edge.second] -= value;
      }

      vector<int> x(weight.size(), 0);
      double primalSolution = solveCliqueSubproblem(ClqPart.Cliques, LagWeights, x, S);
      double lsum = 0.0;
      // sum only those lambdas that are in the active edgeSet (needed when reoptimizing)
      for (const auto& [edge, value] : lambdas)
         lsum += value;

      primalSolution += lsum;
      // initialize for Polyak step size
      if (k==0)
         dualBound = std::min(dualBound, primalSolution);

      // cout << "iteration k = " << k << ": bound = " << dualBound << ", alpha = " << alpha << endl;
      // printLagrangianInfo(lambdas, LagWeights, x, k, alpha, primalSolution);

      // Update stepsize alpha and multipliers lambda
      // double eps = 0.00001;
      // alpha = alpha0; // constant.  Added for testing purposes.  Not effective in general.
      // alpha = alpha0/(k+1);   // a bit too aggressive; becomes too small too quickly
      // alpha = alpha0/sqrt(k + 1);  // works well on smaller graphs
      // if (alpha < eps) break;  // additional stopping criterion.

      // Polyak-style stepsize update: works well on larger graphs
      // gradient = (x[edge.first] + x[edge.second] - 1), so gradient^2 = 1 if xi = xj and 0 otherwise
      // int gradient = sum(edgeSet,[&x](const auto& edge) { return x[edge.first]==x[edge.second];});
      // std::set<std::pair<int, int>> edgeUpdateSet; // set of edges (i,j) for which the gradient is non-zero
      std::unordered_set<std::pair<int, int>> edgeUpdateSet; // set of edges (i,j) for which the gradient is non-zero
      int gradient = 0;
      for (const auto& edge : edgeSet){
         if (x[edge.first]==x[edge.second]) {
            edgeUpdateSet.insert(edge);
            gradient++;
         }
      }
      if (gradient == 0) //cout << "gradient = 0: break" << "\n";
         break;

      alpha = alpha0/gradient;  // ignore any change in the objective change
      /*
      // Taking into account objective information: leads to very small updates
      double IterDiff = std::max(0.0, dualBound - primalSolution);
      if (IterDiff <= eps) {
      alpha = alpha0/gradient;  // ignore the objective change
      }
      else alpha = IterDiff/gradient;
      */

      // primalSol = true if a primal solution is found with gradient = 0)
      // Note: updateMultipliers will remove lambdas that are set to zero.
      bool primalSol = updateMultipliers(lambdas, x, alpha, edgeUpdateSet);

      clock_t lagEnd = clock();
      double elapsed = double(lagEnd - lagStart) / CLOCKS_PER_SEC;
      double outElapsed = double(lagEnd - outStart) / CLOCKS_PER_SEC;
      if (outElapsed > 1.5) {
         std::cout << "Lagrangian bound = " << dualBound << "\t alpha = " << alpha << "\t time: " << elapsed << "s" << "\n";
         outStart = clock();
      }
      // Check if gradient = 0 (and break). The primal solution is not expected to be good though.
      if (primalSol) {
         // double val = 0.0;
         // for (int i=0; i<(int)x.size(); i++)
         //    val += weight[i]*x[i];
         // cout << "Lagrangian found primal solution with value \n";
         dualBound = min(dualBound, primalSolution);
         // std::cout << "Updated Lagrangian bound = " << dualBound << "\n";
         break;
      }
      // Compute the Lagrangian dual bound
      if (primalSolution < dualBound) {
         dualBound = primalSolution;
         unchanged_iters = 0;
      }
      else if (++unchanged_iters > 0.05*maxIter)
         break;
   }

   const double epsilon = 0.00001; // precision error
   return std::floor(dualBound + epsilon);
}


double CliqueBound(int nbCliques, const vector<int>& cliqueOfVertex, GNSet S, std::vector<double> &weight) {
   std::vector<double> clqWeight(nbCliques, 0.0);
   for (auto v : S)
      clqWeight[ cliqueOfVertex[v] ] = std::max(clqWeight[ cliqueOfVertex[v] ],weight[v]);
   return sum<double>(clqWeight,[](auto cw) { return cw;});
}

//__attribute__((noinline)) double sumOfLambdas(const auto& lambdas, const GNSet& S, std::vector<double>& LagWeight) {
//   return sum<double>(lambdas,[&S,&LagWeight](const auto& edge,auto lambda){
//      const auto& [i,j] = edge;
//      if (S.contains(i) && S.contains(j)) {
//         LagWeight[i] -= lambda;
//         LagWeight[j] -= lambda;
//         return lambda;
//      } else return 0.0;
//   });
//}


double LagBound(int nbCliques, const vector<int>& cliqueOfVertex, const GNSet& S, const std::vector<double> &weight, const PairMap &lambdas) {

   if (S.size()<=1) {
      return weight[*S.begin()];
   }
   
   std::vector<double> LagWeight = weight;

   double sumLambdas = 0.0;
   int n = S.size();
   //if (n*(n-1)*0.5 < lambdas.size())
   //   std::cout << "Iterate over i<j in S: n(n-1)/2 = " << n*(n-1)*0.5 << ".  Iterate over |lambdas| = " << lambdas.size() << std::endl;
   if (n*(n-1)*0.5 < lambdas.size()) {
      for (auto i = S.begin(); i != std::prev(S.end()); ++i) {
         for (auto j = std::next(i); j != S.end(); ++j) {
            std::pair<int, int> edge = (*i < *j) ? std::make_pair(*i, *j) : std::make_pair(*j, *i);
            auto it = lambdas.find(edge);
            if (it != lambdas.end()) {  // Check if the edge exists
               double value = it->second;  // Access the value directly
               LagWeight[*i] -= value;
               LagWeight[*j] -= value;
               sumLambdas += value;
            }
         }
      }
   }
   else {
      sumLambdas = sum<double>(lambdas,[&S,&LagWeight](const auto& edge,auto lambda){
         const auto& [i,j] = edge;
         if (S.contains(i) && S.contains(j)) {
            LagWeight[i] -= lambda;
            LagWeight[j] -= lambda;
            return lambda;
         } else return 0.0;
      });
   }
   
   std::vector<double> clqWeight(nbCliques, 0.0);
   for (auto v : S)
      clqWeight[cliqueOfVertex[v]] = std::max(LagWeight[v],clqWeight[cliqueOfVertex[v]]);
   // Add the constant lambdas (if both i and j in S) and the clique max
   // lagrangian multiplier weight
   const double lb = sumLambdas + sum<double>(clqWeight,[](auto w) { return w;});
   const double epsilon = 0.00001; // precision error
   return std::floor(lb + epsilon);
}

int main(int argc,char* argv[])
{
   // using STL containers for the graph
   if (argc < 2) {
      std::cout << "usage: misp <fname> <width> <LB>\n";
      std::cout << "where optional <LB> = 0 (sum of weights) \n";
      std::cout << "                    = 1 (clique partition) \n";
      std::cout << "                    = 2 (Lagrangian; default) \n";
      std::cout << "                    = 3 (Lagrangian recomputed) \n";
      exit(1);
   }
   std::cout << "ARGC:" << argc << "\n";
   const char* fName = argv[1];
   const int w = argc>=3 ? atoi(argv[2]) : 64;
   int LBopt = std::clamp(argc>=4 ? atoi(argv[3]) : 2,0,3);

   std::cout << "w = " << w << ", LBopt = " << LBopt << std::endl;

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
   std::vector<double> weight(ns.size()+1);       // make these doubles
   weight[ns.size()] = 0.0;
   for(auto v : ns) weight[v] = 1.0;
   auto neighbors = instance.adj;

   cout << "Computing clique partition" << "\n";
   auto cliquePartition = maximalCliquePartition(instance.adj, ns);
   // std::cout << cliquePartition;
   // std::string fOutName = std::string(fName) + ".clique_partition";
   // exportCliquePartition(cliquePartition.Cliques,fOutName);
   cout << "Number of cliques = " << cliquePartition.Cliques.size() << "\n";
   // Initialize Lagrangian relaxation
   std::vector<double> LagWeight = weight;
   // for (auto i : ns) LagWeight[i] = 1.0*weight[i];
   int maxIter = std::min(50*ns.size(),10000);
   double alpha0 = 1.0; // this constant is more robust than a weight-based initialization
   PairMap lambdas; // Lagrangian multipliers

   double dualBound = 1.0*cliquePartition.Cliques.size();
   if (LBopt >= 2) {
      double lagBound = lagrangianDual(instance.adj, LagWeight, cliquePartition, maxIter, alpha0, lambdas, false, ns);
      cout << "Lagrangian dual bound = " << lagBound << "\n";
      if (lagBound >= dualBound) {
          //cout << "switching to cliques local bound" << "\n";
          //LBopt = 1;
      } else dualBound = lagBound;
   }

   //  DP Model
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
   const auto scf = [&weight](const MISP& s,int label) { // cost function
      return label * weight[s.n];
   };
   const auto smf = [](const MISP& s1,const MISP& s2) -> std::optional<MISP> { // merge function
      return MISP {s1.sel | s2.sel,std::min(s1.n,s2.n)};
   };
   const auto eqs = [](const MISP& s) -> bool {
      return s.sel.size() == 0;
   };
   std::function<double(const MISP&,LocalContext)> local;
   clqPartition DDcliquePartition = cliquePartition;  // clique partition to be used inside DD
   PairMap DDlambdas = lambdas;                       // multipliers to be used inside the DD
   bool DDreOpt = false;                              // flag to ensure that the DDcliquePartition and DDlambdas can be used
   switch(LBopt) {
      case 0: local = [&weight](const MISP& s,LocalContext) -> double {
         return sum(s.sel,[&weight](auto v) { return weight[v];});
      };break;
      case 1: local = [&weight,&cliquePartition](const MISP& s,LocalContext) -> double {
         return CliqueBound(cliquePartition.nc, cliquePartition.CliqueOfVertex, s.sel, weight);
      };break;
      case 2: local = [&weight,&cliquePartition,&lambdas,&instance](const MISP& s,LocalContext) -> double {
         return LagBound(cliquePartition.nc, cliquePartition.CliqueOfVertex, s.sel, weight, lambdas);
      };break;
      case 3: local = [&weight,&cliquePartition,&lambdas,&instance,&LagWeight,&bnds,&DDlambdas,&DDcliquePartition,&DDreOpt](const MISP& s,LocalContext lc) -> double {
         switch(lc) {
            case BBCtx: {
               return LagBound(cliquePartition.nc, cliquePartition.CliqueOfVertex, s.sel, weight, lambdas);
            }
            case DDInit: {
               // Use maxIter = 50 for quick reoptimization
               // cout << "DDInit: dual:" << bnds.getDual() << "\n";
               DDcliquePartition = maximalCliquePartition(instance.adj, s.sel);
               DDlambdas = lambdas; // start from original lambdas each time - only used when flag reOpt = true
               double nd = lagrangianDual(instance.adj, LagWeight, DDcliquePartition, 25, 1.0, DDlambdas, false, s.sel);
               // cout << "\t--> (current dual = " << bnds.getDual() << ".) \t bnd: " << bnds.getG() << " new dual heur:" << nd << " dual:" << bnds.getG() + nd << "\n";
               if (bnds.getG() + nd >= bnds.getDual()) {
                  // no improvement--revert to original setting
                  DDcliquePartition = cliquePartition;
                  DDlambdas = lambdas;
                  DDreOpt = false;
                  return bnds.getDual();
               }
               DDreOpt = true;
               // else cout << "\t ==FOUND A BETTER DUAL BOUND==" << endl;
               // if (nd <= bnds.getPrimal()) cout << "\t Updated bound can prune this node" << endl;
               return nd;
            }
            case DDCtx: {
               if (DDreOpt) return LagBound(DDcliquePartition.nc, DDcliquePartition.CliqueOfVertex, s.sel, weight, DDlambdas);
               else return LagBound(cliquePartition.nc, cliquePartition.CliqueOfVertex, s.sel, weight, lambdas);
               //return LagBound(cliquePartition.nc, cliquePartition.CliqueOfVertex, s.sel, weight, lambdas);
            }
         }
      };break;
      default:
         throw std::invalid_argument("Invalid LBopt value! Must be 0, 1, or 2.");
   }
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
