#include "codd.hpp"
#include "searchRestrictedFirst.hpp"
#include "searchRestrictedOnly.hpp"
#include <cmath>

struct SKS {
   int n;           // variable index
   int c;           // remaining capacity
   friend std::ostream& operator<<(std::ostream& os,const SKS& m) {
      return os << "<" << m.n << ',' << m.c << ">";
   }
};

template<> struct std::equal_to<SKS> {
   constexpr bool operator()(const SKS& s1,const SKS& s2) const {
      return s1.n == s2.n && s1.c == s2.c;
   }
};

template<> struct std::hash<SKS> {
   std::size_t operator()(const SKS& v) const noexcept {
      return std::rotl(std::hash<int>{}(v.n),32) ^
         std::hash<int>{}(v.c);
   }
};

struct Item {
   int w; // weight
   int p; // profit
};

struct Instance {
   int I;
   int capa;
   Item* items;
   std::vector<int> weight;
   std::vector<int> profit;
   Instance() {}
   Instance(Instance&& i)
      : I(i.I),capa(i.capa),
        weight(std::move(i.weight)),
        profit(std::move(i.profit))
   {}
   int getUB() {
      int rC = capa;
      int val = 0;
      int cnt = 0;
      while (rC >= 0) {
         if (weight[cnt] <= rC) {
            val += profit[cnt];
            rC  -= weight[cnt];
            cnt++;
         } else break;
      }
      return val;
   }
};

Instance readFile(const char* fName)
{
   Instance i;
   using namespace std;
   ifstream f(fName);
   f >> i.I >> i.capa;
   i.items = new Item[i.I];
   int count = 0;
   while (!f.eof()) {
      if (f.eof()) break;
      Item k;
      f >> k.p >> k.w;
      i.items[count] = k;
      if (count++ >= i.I) break;
   }
   f.close();
   mergeSort(i.items,i.I,[](const auto& a,const auto& b) {
      const double ar = -((double)a.p)/((double)a.w);
      const double br = -((double)b.p)/((double)b.w);
      return (ar == br) ? a.p <= b.p : ar <= br;
   });
   for(auto k=0;k < i.I;k++) {
      auto ik = i.items[k];
      i.weight.push_back(ik.w);
      i.profit.push_back(ik.p);
   }
   return i;
}

int main(int argc,char* argv[])
{
   if (argc < 2) {
      std::cout << "Usage knapsack <fname> <width>\n";
      exit(1);
   }
   const char* fName = argv[1];
   const int width = argc>=2 ? atoi(argv[2]) : 64;

   Instance instance = readFile(fName);
   const int I = instance.I;
   const int capa = instance.capa;
   const auto& w = instance.weight;
   const auto& p = instance.profit; 
   Bounds bnds([&w,&p,capa,I](const std::vector<int>& inc)  {
      double v = 0.0;
      int   rc = capa;
      for(int i=0;i < I;i++) {
         if (inc[i]) {
            rc -= w[i];
            v  += p[i];
         }
      }
      std::cout << "CHECKER:" << v << " RC:" << rc << "\n";
   });
   const auto labels = GNSet(0,1);     // using a plain set for the labels
   bnds.setPrimal(instance.getUB());
   std::cout << "CAPA:" << capa << "\n";
   const auto init = [capa]()         { return SKS {0,capa};};
   const auto target = [I]()          { return SKS {I,0};};
   const auto lgf = [w](const SKS& s,DDContext) { return  decreasing(Range::close(0,s.c >= w[s.n]));};
   const auto stf = [I,&w](const SKS& s,const int label) -> std::optional<SKS> {
      if (s.n < I-1) {
         return SKS { s.n+1,s.c - label * w[s.n] };
      } else return SKS { I, 0};
   };
   const auto local = [I,&w,&p](const SKS& s,LocalContext) -> double {
      double nn = 0;
      int    rc = s.c;
      for(auto i=s.n; i < I;i++) 
         if (w[i] <= rc) {
            rc -= w[i];
            nn += p[i];
         } else {            
            nn += std::floor(((double)rc / w[i]) * p[i]);
            break;
         }      
      return nn;
   };

   const auto scf = [p](const SKS& s,int label) { return p[s.n] * label;};
   const auto smf = [capa](const SKS& s1,const SKS& s2) -> std::optional<SKS> {
      //return SKS { std::max(s1.n,s2.n),std::max(s1.c,s2.c) };       
      //if (abs(s1.c - s2.c) <= (0.003 * capa)/100)
      if (abs(s1.c - s2.c) <= 1)
         //      if (s1.c == s2.c)
         return SKS { std::max(s1.n,s2.n),std::max(s1.c,s2.c) };       
      else return std::nullopt; // return  the empty optional
   };
   const auto sEq = [I](const SKS& s) -> bool              { return s.n == I;};

   const auto sDom = [](const SKS& a,const SKS& b) -> bool { 
      return  a.c >= b.c;
   };
      
   BAndB engine(DD<SKS,Maximize<double>,
               decltype(target),
               decltype(lgf),
               decltype(stf),
               decltype(scf),
               decltype(smf),
               decltype(sEq)
               >::makeDD(init,target,lgf,stf,scf,smf,sEq,labels,local,sDom),width);
   engine.search(bnds);

   return 0;
}

