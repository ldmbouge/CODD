#ifndef __UTIL_H
#define __UTIL_H

#include <iostream>
#include <vector>
#include <tuple>
#include <set>
#include <limits>

namespace std {
   template <class T> T min(const set<T>& s) {
      if (s.size() > 0)
         return *s.begin();
      else 
         return numeric_limits<T>::max();
   }
   template <class T> T max(const set<T>& s) {
      if (s.size() > 0)
         return *(s.rbegin());
      else 
         return numeric_limits<T>::min();
   }
   template <class T,typename Pred>
   set<T> filter(set<T> inSet,Pred p) {
      std::set<T> outSet = {};
      for(const auto& v : inSet)
         if (p(v))
            outSet.insert(v);
      return outSet;
   }
   template <class Container,class Pred>
   bool member(const Container& cont,Pred c) {
      for(const auto& v : cont)
         if (c(v)) return true;
      return false;
   }
};

template <class T> std::ostream& operator<<(std::ostream& os,const std::set<T>& s) {
   os << "{";
   auto cnt = 0u;
   for(auto i=s.begin();i!= s.end();i++,cnt++)
      os << *i << ((cnt==s.size()-1) ? "" : ",");
   return os << "}";
}

template <class T> std::ostream& operator<<(std::ostream& os,const std::vector<T>& msg) {
   os << '[';
   for(const T& v : msg)
      os << v << " ";
   return os << ']';
}

template<class T> struct std::hash<std::vector<T>> {
   std::size_t operator()(const std::vector<T>& v) const noexcept {
      std::size_t ttl = 0;
      for(auto e : v)
         ttl = (ttl << 5) | std::hash<T>{}(e);
      return ttl;
   }   
};

template<class T> struct std::hash<std::set<T>> {
   std::size_t operator()(const std::set<T>& v) const noexcept {
      std::size_t ttl = 0;
      for(auto e : v)
         ttl = (ttl << 3) | std::hash<T>{}(e);
      return ttl;
   }
};

template <typename T>
std::set<T> operator|(const std::set<T>& s1,const std::set<T>& s2)
{
   std::set<T> r = s1;
   r.insert(s2.begin(),s2.end());
   return r;
}

template <typename T>
std::set<T> operator&(const std::set<T>& s1,const std::set<T>& s2)
{
   std::set<T> r {};
   for(const auto& e : s1)
      if (s2.contains(e))
         r.insert(e);
   return r;
}


template <typename T>
std::set<T> remove(const std::set<T>& s1,const T& v) {
   std::set<T> r;
   for(T e : s1) {
      if (e == v) continue;
      r.insert(e);
   }
   return r;
}


class NatSet {
   unsigned short _mxw;
   unsigned short _nbp;
   unsigned long long* _t;
public:
   NatSet() : _mxw(0),_nbp(0),_t(nullptr) {}
   NatSet(const NatSet& s) {
      _mxw = s._mxw;
      _nbp = s._nbp;
      _t   = new unsigned long long[_mxw];
      for(int i=0;i<_mxw;i++)
         _t[i] = s._t[i];
   }   
   NatSet(unsigned short nb) {
      _mxw = (nb >> 6) + (((nb & 63) != 0) ? 1 : 0);
      _nbp = nb;
      if (_mxw) {
         _t   = new unsigned long long[_mxw];
         for(int i=0;i<_mxw;i++) _t[i]=0;
      } else _t = nullptr;
   }
   NatSet(unsigned long long* buf,unsigned short nb) {
      _mxw = (nb >> 6) + (((nb & 63) != 0) ? 1 : 0);
      _nbp = nb;
      _t   = buf;
      for(int i=0;i<_mxw;i++) _t[i]=0;
   }
   short nbWords() const noexcept { return _mxw;}
   int largestPossible() const noexcept { return _nbp;}
   void clear() noexcept {
      for(short i=0;i < _mxw;i++)
         _t[i] = 0;
   }
   int size() const noexcept {
      int ttl = 0;
      for(short i=0;i < _mxw;++i)
         ttl += __builtin_popcountll(_t[i]);
      return ttl;
   }
   void insert(int p) noexcept         { _t[p >> 6] |= (1ull << (p & 63));}
   bool contains(int p) const noexcept { return (_t[p >> 6] &  (1ull << (p & 63))) != 0;}
   void unionWith(const NatSet& ps) noexcept {
      assert(_mxw == ps._mxw);
      switch (_mxw) {
         case 1: _t[0] |= ps._t[0];break;
#if defined(__x86_64__)
         case 2: {
            __m128i op0 = *(__m128i*)_t;
            __m128i op1 = *(__m128i*)ps._t;
            *(__m128i*)_t = _mm_or_si128(op0,op1);
         }break;
         case 3: {
            __m128i op0 = *(__m128i*)_t;
            __m128i op1 = *(__m128i*)ps._t;
            *(__m128i*)_t = _mm_or_si128(op0,op1);
            _t[2] |= ps._t[2];
         }break;
#endif
         default: {
            for(short i=0;i < _mxw;++i)
               _t[i] |= ps._t[i]; 
         }
      }
   }
   void interWith(const NatSet& ps) noexcept {
      for(short i=0;i < _mxw;i++)
         _t[i] &= ps._t[i];
   }
   class iterator { 
      unsigned long long*    _t;
      const unsigned short _nbw;
      unsigned short       _cwi;    // current word index
      unsigned long long    _cw; // current word
      iterator(unsigned long long* t,unsigned short nbw,unsigned short at)
         : _t(t),_nbw(nbw),_cwi(at),_cw((at < nbw) ? t[at] : 0) {
         while (_cw == 0 && ++_cwi < _nbw) 
            _cw = _t[_cwi];         
      }
      iterator(unsigned long long* t,unsigned short nbw) : _t(t),_nbw(nbw),_cwi(nbw),_cw(0) {} // end constructor
   public:
      using iterator_category = std::forward_iterator_tag;
      using value_type = short;
      using difference_type = short;
      using pointer = short*;
      using reference = short&;
      iterator& operator++()  noexcept {
         long long test = _cw & -_cw;  // only leaves LSB at 1
         _cw ^= test;                  // clear LSB
         while (_cw == 0 && ++_cwi < _nbw)  // all bits at zero-> done with this word.            
            _cw = _t[_cwi];        
         return *this;
      }
      iterator operator++(int) { iterator retval = *this; ++(*this); return retval;}
      bool operator==(iterator other) const {return _cwi == other._cwi && _cw == other._cw;}
      bool operator!=(iterator other) const {return !(*this == other);}
      short operator*() const   { return (_cwi<<6) + __builtin_ctzl(_cw);}
      friend class NatSet;
   };
   iterator begin() const { return iterator(_t,_mxw,0);}
   iterator end()   const { return iterator(_t,_mxw);}
   friend std::ostream& operator<<(std::ostream& os,const NatSet& ps) {
      os << '[' << ps.size() << ']' << '{';
      for(short i : ps) os << i << ' ';
      return os << '}';
   }
};

inline NatSet operator|(const NatSet& s1,const NatSet& s2)
{
   NatSet r(s1);
   r.unionWith(s2);
   return r;
}

inline NatSet operator&(const NatSet& s1,const NatSet& s2)
{
   NatSet r(s1);
   r.interWith(s2);
   return r;
}

#endif
