/*
 * ddOpt is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License  v3
 * as published by the Free Software Foundation.
 *
 * ddOpt is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY.
 * See the GNU Lesser General Public License  for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with mini-cp. If not, see http://www.gnu.org/licenses/lgpl-3.0.en.html
 *
 * Copyright (c)  2023. by Laurent Michel.
 */


#ifndef __MINICPP_HEAP_H
#define __MINICPP_HEAP_H

#include <algorithm>
#include <iostream>
#include <functional>
#include <mutex>
#include <condition_variable>
#include "store.hpp"

template <class T,typename Ord = bool(*)(const T&,const T&)> class Heap {
public:
   class Location {
      T _val;
      int _p;
      friend class Heap<T,Ord>;
      Location() {}
      Location(const T& val,int p) : _val(val),_p(p) {}
   public:
      T value() const noexcept { return _val;}
      T operator*() const noexcept { return _val;}
      friend std::ostream& operator<<(std::ostream& os,const Location& l) {
         return os << "<" << l._p << ":" << l._val << ">";
      }
   };
   typedef Location LocType;
private:
   Pool::Ptr     _pool; // where memory comes from
   Location**    _data; // location pointers
   Location**    _lseg; // list of location segments
   // current size of segment and insertion point
   int            _mxs; // maximal size
   int             _at; // actual size (where next element goes)
   // segments for location allocation
   int             _ms; // maximum number of segments 
   int            _nbs; // number of segments
   Ord            _ord;
   void resize() noexcept {
      const int newSize = _mxs << 1; // double
      Location* newSeg = new (_pool) Location[_mxs];
      Location** nd    = new (_pool) Location*[newSize];
      for(int i=0u;i < _at;i++) {
         nd[i] = _data[i];
      }
      for(int i=_mxs; i < newSize;i++) {
         nd[i] = newSeg + i - _mxs;
         nd[i]->_p = i;
      }
      if (_nbs >= _ms) { // make sure we have room to track segment
         Location** nls = new (_pool) Location*[_ms << 1];
         for(auto i=0;i < _ms;i++)
            nls[i] = _lseg[i];
         _lseg = nls;
         _ms <<= 1;
      }
      _lseg[_nbs++] = newSeg;
      _mxs = newSize;
      _data = nd;
   }
   void sendToRoot(int p) noexcept {
      while(p>0) {
         int gp = p / 2;
         if (gp) {
            std::swap(_data[p],_data[gp]);
            _data[p]->_p = p;
            _data[gp]->_p = gp;
         }
         p = gp;
      }
   }

   void refloat(int p) noexcept {
      while(p>0) {
         int l = p * 2,r = p * 2 + 1;
         int best;
         if (l < _at && _ord(_data[l]->_val,_data[p]->_val))
            best = l;
         else best = p;
         if (r < _at && _ord(_data[r]->_val,_data[best]->_val))
            best = r;
         if (best != p)  {
            std::swap(_data[p],_data[best]);
            _data[p]->_p = p;
            _data[best]->_p = best;
         }
         p = p / 2;
      }
   }
   void heapify(int p) noexcept {
      while(true) {
         int l = p * 2,r = p * 2 + 1;
         int best;
         if (l < _at && _ord(_data[l]->_val,_data[p]->_val))
            best = l;
         else best = p;
         if (r < _at && _ord(_data[r]->_val,_data[best]->_val))
            best = r;
         if (best != p) {
            std::swap(_data[p],_data[best]);
            _data[p]->_p = p;
            _data[best]->_p = best;
            p = best;
         } else break;
      }
   }
public:
   Heap(Pool::Ptr p,int sz,Ord ord)
      : _pool(p),_mxs(sz),_at(1),_ms(32),_nbs(0),_ord(ord)
   {
      _lseg = new (_pool) Location*[_ms];
      auto cs = _lseg[_nbs++] = new (_pool) Location[_mxs];
      _data = new (_pool) Location*[_mxs];
      for(int i=0u;i<_mxs;++i) {         
         _data[i]  = cs + i;
         *_data[i] = Location { T(), i };
      }
   }
   void clear() noexcept { _at = 1;}
   unsigned size() const noexcept { return (unsigned)(_at - 1);}
   bool empty() const noexcept { return _at == 1;}
   Location* operator[](int i) const noexcept {
      return _data[i+1];
   }
   Location* insert(const T& v) noexcept {
      if (_at >= _mxs) resize();
      _data[_at]->_val = v;
      _data[_at]->_p = _at;
      ++_at;
      return _data[_at-1];
   }
   Location* find(const T& v) {
      for(int i=1;i<_mxs;i++)
         if (_data[i]->_val == v)
            return _data[i];
      return nullptr;
   }
   void decrease(Location* at) {
      at->_val--;
      refloat(at->_p);
   }
   T remove(Location* at) {      
      sendToRoot(at->_p);
      return extractMax();
   }
   const Location* insertHeap(const T& v) noexcept {
      auto loc = insert(v);
      refloat(_at-1);
      return loc;
   }
   void buildHeap() noexcept {
      for(int i = _at / 2;i > 0;--i)
         heapify(i);
   }
   T extractMax() noexcept {
      assert(_at >= 1);
      Location* rv = _data[1];
      std::swap(_data[1],_data[_at-1]);
      _data[_at - 1]->_p = _at - 1;
      _data[1]->_p = 1;
      --_at;
      heapify(1);
      return rv->_val;
   }
   friend std::ostream& operator<<(std::ostream& os,const Heap<T,Ord>& h) {
      for(int i=1;i < h._at;i++) 
         os << "   " << i << ":" << h._data[i]->_val << "\n";      
      return os;
   }
   template <typename Printer>
   std::ostream& printHeap(std::ostream& os,const Printer& p) {
      for(int i=1;i < _at;i++) { 
         os << "   " << i << ":";
         p(os,_data[i]->_val);
         os << "\n";
      }
      return os;
   }
};

template <class T,typename Ord = bool(*)(const T&,const T&)> class ThreadSafeHeap {

public:
   ThreadSafeHeap(Pool::Ptr p,int sz,Ord ord): _heap(p, sz, ord), _ord(ord) {}

   typedef Heap<T,Ord>::LocType LocType;

   void insertHeap(T item) {
      //std::cout << "inserting!\n";
      std::lock_guard<std::mutex> lock(_mtx); 
      _heap.insertHeap(item); 
      _cv.notify_one();
      // unlock _mtx
   }
   template <typename U, typename TRANS = std::optional<T>(*)(const U&)>
   void insertAllIf(std::vector<U>& toAdd, TRANS&& transformer) {
      std::lock_guard<std::mutex> lock(_mtx); 
      for(const auto& u: toAdd) {
         // std::cout << "inserting!\n";
         std::optional<T> t = transformer(u);
         if(t.has_value()) _heap.insertHeap(t.value());
      }
      _cv.notify_one();
      // unlock _mtx
   }
   std::optional<T> extractMax() { 
      std::unique_lock<std::mutex> lock(_mtx);
      if(empty()) return std::nullopt; // unlock _mtx
      //std::cout << _heap.size() << "(" << empty() << ")" << "\n";
      auto tmp = _heap.extractMax();
      //std::cout << _heap.size() << "(" << empty() << ")" << "\n\n";
      return tmp;
      // unlock _mtx
   }

   // template <typename PRED = bool(*)(const T&)>
   // std::optional<T> extractFirstValid(PRED isValid) {
   //      auto indexOrd = [this](int a, int b) { return !_ord(**_heap[a], **_heap[b]); }; //std queue used opposite default order
   //      std::priority_queue<int, std::vector<int>, decltype(indexOrd)> frontier(indexOrd);

   //      std::unique_lock<std::mutex> lock(_mtx);
   //      _cv.wait(lock, [&]() { return !empty(); });

   //      frontier.push(0);
   //      while (!frontier.empty()) {
   //          //std::cout << frontier.__get_container() << "\n";
   //          int i = frontier.top();
   //          frontier.pop();

   //          LocType* currLoc = _heap[i];
   //          //std::cout << i << " " << *currLoc << "\n";
   //          if (isValid(currLoc->value())) {
   //              _heap.remove(currLoc);
   //              return currLoc->value();
   //              //unlock
   //          }

   //          unsigned int left  = 2*i + 1;
   //          unsigned int right = 2*i + 2;

   //          if (left  < size()) frontier.push(left );
   //          if (right < size()) frontier.push(right);
   //          //std::cout << frontier.__get_container() << "\n";
   //    }
   //    return std::nullopt;
   //    //unlock
   // }

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
   void vetHeap(PRED&& p, ThreadSafeHeap<T,Ord>& vetted) {
      std::cout << "vetting...\n";
      std::unique_lock<std::mutex> lock(_mtx);
      while(true) {
         while (empty() && !_done) {
            _cv.wait(lock);
         }
         if(_done) break;
         //std::cout << "vetting candidate (" << _heap.size() << "," << vetted.size() << ")\n";
         auto candidate = _heap.extractMax();
         if (vetted.size() < _heap.size() * 0.1) {
            vetted.insertHeap(candidate);
            continue;
         } else {            
         // std::cout << " -> (" << size() << ")\n";
            lock.unlock();
            if(p(candidate)) {
               //std::cout << "vetting -> moved:" << candidate.node->getBound() << "\n";
               vetted.insertHeap(candidate);
            } else {
               //std::cout << "rejecting --> " << candidate.node->getBound() << "\n";
               if(empty()) { // if the last candidate was rejected wake the main thread manually
                  // std::cout << "ran out, wake up main!\n";
                  vetted.notify_one(); 
               }
            }
            lock.lock();
         }
      }
      std::cout << "done vetting\n";
   }

   template <typename ACTION, typename PRED>
   void onArrival(PRED&& p, ACTION&& action) {
      std::unique_lock<std::mutex> lock(_mtx);

      while (true) {
         while (p() && !_done) {
            _cv.wait(lock); 
            // std::cout << "waking up...\n";
         }
         if (_done) break;

         lock.unlock();
         action();
         lock.lock(); 
      }
   }
   // template <typename PRED>
   // void filter(PRED&& p) {
   //    //std::cout << "filtering...\n";
   //    size_t i = 0;
   //    std::unique_lock<std::mutex> lock(_mtx, std::defer_lock); // create, but don't lock, the mutex
   //    while(true) {
   //       lock.lock();
   //       _cv.wait(lock, [&]() { return !empty() || _done; });
   //       if(_done) {
   //          //std::cout << "done culling\n";
   //          return;
   //       }
   //       LocType* at;
   //       do {
   //          if(i >= size()) i = 0;
   //          at = _heap[i];
   //       } while(at->value().getState() != T::State::OPEN);
   //       at->value().setState(T::State::STOLEN);
   //       lock.unlock();
   //       if(p(at->value()) && !empty()) {
   //          lock.lock();
   //          _heap.remove(at);
   //       } else {
   //          lock.lock();
   //          at->value().setState(T::State::VETTED);
   //       }
   //       lock.unlock();
   //    }
   // } 
   void notify_one() { _cv.notify_one(); }
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

#endif
