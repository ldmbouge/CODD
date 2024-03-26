#include "dd.hpp"
#include "util.hpp"
#include "search.hpp"
#include <concepts>
#include <iostream>
#include <fstream>
#include <set>
#include <optional>
#include <ranges>
#include <algorithm>
#include <map>
#include <math.h>

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
   for(auto k=0;k < i.I;k++) {
      auto ik = i.items[k];
      std::cout << "(" << ik.w << "/" << ik.p << ")" << " ";
      assert(ik.w != 0);
      assert(ik.p != 0);
   }
   std::cout << "\n";
   mergeSort(i.items,i.I,[](const auto& a,const auto& b) {
      const double ar = -((double)a.p)/((double)a.w);
      const double br = -((double)b.p)/((double)b.w);
      return ar <= br;
   });
   std::cout << "Sorted by ratio... ----------------------------------------------------\n";
   for(auto k=0;k < i.I;k++) {
      auto ik = i.items[k];
      i.weight.push_back(ik.w);
      i.profit.push_back(ik.p);
      std::cout << ik.w << " " << ik.p << "\n";
   }
   return i;
}

int main(int argc,char* argv[])
{
   if (argc < 3) {
      std::cout << "Usage gruler <#marks> <#ubLen> <maxWidth>\n";
      exit(1);
   }
   const char* fName = argv[1];
   const int w = argc==3 ? atoi(argv[2]) : 64;
   Instance instance = readFile(fName);
   const int I = instance.I;
   const int capa = instance.capa;
   const auto& weight = instance.weight;
   const auto& profit = instance.profit; 
   const int UB = instance.getUB();
   std::cout << "I    =" << I << "\n";
   std::cout << "capa =" << capa << "\n";
   std::cout << "W    =" << weight << "\n";
   std::cout << "P    =" << profit << "\n";
   std::cout << "UB   =" << UB << "\n";
   Bounds bnds;
   const auto labels = GNSet(0,1);     // using a plain set for the labels
   bnds.setPrimal(UB);
   const auto init = [capa]() {   // The root state      
      return SKS {0,capa};
   };
   const auto target = [I]() {    // The sink state
      return SKS {I,0};
   };
   const auto lgf = [weight](const SKS& s) {
      auto r = Range::close(s.c >= weight[s.n],0);
      //auto r = Range::close(0,s.c >= weight[s.n]);
      return r;
   };
   const auto stf = [I,&weight](const SKS& s,const int label) -> std::optional<SKS> {
      if (s.n < I-1) {
         return SKS { s.n+1,s.c - label * weight[s.n] };
      } else return SKS { I, 0};
   };
   auto local = [I,&weight,&profit](const SKS& s) -> double {
      double nn = 0;
      int    rc = s.c;
      for(auto i=s.n; i < I;i++) 
         if (weight[i] <= rc) {
            rc -= weight[i];
            nn += profit[i];
         } else {
            nn += std::floor(((double)rc / weight[i]) * profit[i]);
            break;
         }      
      return nn;
   };      
   const auto scf = [profit](const SKS& s,int label) { // partial cost function 
      return profit[s.n] * label;
   };
   const auto smf = [capa](const SKS& s1,const SKS& s2) -> std::optional<SKS> {
      if (abs(s1.c - s2.c) <= (2 * capa)/100) 
         return SKS { std::max(s1.n,s2.n),std::max(s1.c,s2.c) };       
      else return std::nullopt; // return  the empty optional
   };
   const auto sEq = [I](const SKS& s) -> bool {
      return s.n == I;
   };
   const auto sDom = [](const SKS& a,const SKS& b) -> bool {
      return  a.c >= b.c;
   };
   
   BAndB engine(DD<SKS,Maximize<double>, // to maximize               
                decltype(target),
                decltype(lgf),
                decltype(stf),
                decltype(scf),
                decltype(smf),
                decltype(sEq),
                decltype(local)
                >::makeDD(init,target,lgf,stf,scf,smf,sEq,labels,local,sDom),w);
   engine.search(bnds);
   return 0;
}

