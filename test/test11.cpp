#include <queue>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cstdio>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include "util.hpp"
#include "heap.hpp"
#include "search.hpp"
#include "node.hpp"
#include "RuntimeMonitor.hpp"
#include "pool.hpp"

struct MyNode {
   int value;
   bool valid;
   friend std::ostream& operator<<(std::ostream& os,const MyNode& q) {
      return os << "Node[" << q.value << "(" << q.valid << ")]";
   }
};

void testQueue(Bounds& bnds, AbstractDD::Ptr _theDD)
{
    auto bbPool = _theDD->makeNDAllocator();
    auto hOrder = [](const MyNode& a,const MyNode& b) {
        return a.value < b.value;
    };
    ThreadSafeHeap<MyNode, decltype(hOrder)> q(bbPool->get(),64000,hOrder);

    q.insertHeap({ 5, false});
    q.insertHeap({15, false});
    q.insertHeap({25, true});
    q.insertHeap({10, false});
    q.insertHeap({20, false});
    q.insertHeap({30, true});

    for(size_t i = 0; i < q.size(); i++)
       std::cout << *q[i] << "\n";

    auto bestValid = q.extractMax();
    if(bestValid.has_value())
       std::cout << bestValid.value() << "\n";

    for(size_t i = 0; i < q.size(); i++)
       std::cout << *q[i] << "\n";

    return;
};

int main(int argc,char* argv[]) {
   Bounds bnds([](const std::vector<int>& inc)  {
   });
   const auto init = []() {return 0;};
   const auto target = []() {return 0;};
   const auto lgf = [](const int& s,DDContext)  {return 0;};
   const auto stf = [](const int& s,const int label) -> std::optional<int> {return std::nullopt;};
   const auto scf = [](const int& s,int label) {return 0;};
   const auto smf = [](const int& s1,const int& s2) -> std::optional<int> {return std::nullopt;};
   const auto eqs = [](const int& s) -> bool {return false;};
   const auto local = [](const int& s,LocalContext) -> double {return 0;};
   const auto sDom = [](const int& a,const int& b) -> bool {return false;};
   const auto project = [](const int& t) -> std::tuple<int> { return std::make_tuple(0);};
   testQueue(bnds,
             DD<int,Minimize<double>,
             std::tuple<int>,
             decltype(target),
             decltype(lgf),
             decltype(stf),
             decltype(scf),
             decltype(smf),
             decltype(eqs),
             decltype(sDom)
             >::makeDD(init,target,lgf,stf,scf,smf,eqs,0,local,project,sDom));
   return 0;
}
