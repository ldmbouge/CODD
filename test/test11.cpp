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

template <class T,typename Ord = bool(*)(const T&,const T&)> class ThreadSafeHeap {

public:
   ThreadSafeHeap(Pool::Ptr p,int sz,Ord ord): _heap(p, sz, ord), _ord(ord) {}

   typedef Heap<T,Ord>::LocType LocType;

   void insertHeap(T item) {
      std::lock_guard<std::mutex> lock(_mtx); 
      _heap.insertHeap(item); 
      _cv.notify_one();
      // unlock _mtx
   }
   template <typename U, typename TRANS = std::optional<T>(*)(const U&)>
   void insertAllIf(std::vector<U> toAdd, TRANS transformer) {
      std::lock_guard<std::mutex> lock(_mtx); 
      for(auto u: toAdd) {
         std::optional<T> t = transformer(u);
         if(t.has_value()) _heap.insertHeap(t.value());
      }
      _cv.notify_one();
      // unlock _mtx
   }
   std::optional<T> extractMax() { 
      std::unique_lock<std::mutex> lock(_mtx);
      if(empty()) return std::nullopt;
      std::cout << _heap.size() << "(" << empty() << ")" << "\n";
      auto tmp = _heap.extractMax();
      std::cout << _heap.size() << "(" << empty() << ")" << "\n\n";
      return tmp;
      // unlock _mtx
   }

   template <typename PRED = bool(*)(const T&)>
   std::optional<T> extractFirstValid(PRED isValid) {
        auto indexOrd = [this](int a, int b) { return !_ord(**_heap[a], **_heap[b]); }; //std queue used opposite default order
        std::priority_queue<int, std::vector<int>, decltype(indexOrd)> frontier(indexOrd);

        std::unique_lock<std::mutex> lock(_mtx);
        _cv.wait(lock, [&]() { return !empty(); });

        frontier.push(0);
        while (!frontier.empty()) {
            //std::cout << frontier.__get_container() << "\n";
            int i = frontier.top();
            frontier.pop();

            LocType* currLoc = _heap[i];
            //std::cout << i << " " << *currLoc << "\n";
            if (isValid(currLoc->value())) {
                _heap.remove(currLoc);
                return currLoc->value();
                //unlock
            }

            unsigned int left  = 2*i + 1;
            unsigned int right = 2*i + 2;

            if (left  < size()) frontier.push(left );
            if (right < size()) frontier.push(right);
            //std::cout << frontier.__get_container() << "\n";
      }
      return std::nullopt;
      //unlock
   }
   LocType* operator[](int i) {
      std::unique_lock<std::mutex> lock(_mtx);
      return _heap[i];
      // unlock _mtx
   }
   // T remove(LocType* at) {
   //    std::unique_lock<std::mutex> lock(_mtx);
   //    _cv.wait(lock, [&]() { return !empty(); });
   //    return _heap.remove(at);
   //    // unlock _mtx 
   // }
   template <typename PRED>
   void filter(PRED&& p) {
      //std::cout << "filtering...\n";
      size_t i = 0;
      std::unique_lock<std::mutex> lock(_mtx, std::defer_lock); // create, but don't lock, the mutex
      while(true) {
         lock.lock();
         _cv.wait(lock, [&]() { return !empty() || _done; });
         if(_done) {
            //std::cout << "done culling\n";
            return;
         }
         LocType* at;
         do {
            if(i >= size()) i = 0;
            at = _heap[i];
         } while(at->value().getState() != T::State::OPEN);
         at->value().setState(T::State::STOLEN);
         lock.unlock();
         if(p(at->value())) {
            lock.lock();
            _heap.remove(at);
         } else {
            lock.lock();
            at->value().setState(T::State::VETTED);
         }
         lock.unlock();
      }
   } 
   bool empty() { return _heap.empty(); }
   unsigned size() { return _heap.size(); }
   void setDone() {
      _done = true;
      _cv.notify_all();
   }
private:
   Heap<T,Ord> _heap;
   Ord _ord;
   std::mutex _mtx;
   std::condition_variable _cv;
   std::atomic_bool _done = false;
};

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

    auto bestValid = q.extractFirstValid([](const MyNode& n){ return n.valid; });
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

   testQueue(bnds, DD<int,Minimize<double>,
                decltype(target),
                decltype(lgf),
                decltype(stf),
                decltype(scf),
                decltype(smf),
                decltype(eqs),
                decltype(local)
                >::makeDD(init,target,lgf,stf,scf,smf,eqs,0,local,sDom));
   return 0;
}
