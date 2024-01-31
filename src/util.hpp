#ifndef __UTIL_H
#define __UTIL_H

#include <iostream>
#include <vector>
#include <tuple>
#include <set>
#include <limits>
#include <initializer_list>
#include <ranges>
#include <assert.h>
#if defined(__x86_64__)
#include <intrin.h>
#endif


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

template <unsigned short nbw=1> 
class NatSet {
   unsigned long long _t[nbw];
public:
   NatSet() {
      for(auto i=0u;i < nbw;i++)
         _t[i]=0;
   }
   NatSet(const NatSet& s) {
      for(int i=0;i<nbw;i++)
         _t[i] = s._t[i];
   }
   NatSet(std::initializer_list<int> l) {
      for(auto i=0u;i < nbw;i++)
         _t[i]=0;
      for(auto it = l.begin();it!=l.end();it++) {
         assert((*it >> 6) <= nbw-1);
         insert(*it);
      }
   }
   NatSet& operator=(const NatSet& s) noexcept {
      for(int i=0;i<nbw;i++)
         _t[i] = s._t[i];
      return *this;
   }
   constexpr unsigned short nbWords() const noexcept { return nbw;}
   constexpr unsigned short largestPossible() const noexcept { return nbw*64;}
   void clear() noexcept {
      for(short i=0;i < nbw;i++)
         _t[i] = 0;
   }
   int size() const noexcept {
      int ttl = 0;
      for(short i=0;i < nbw;++i)
         ttl += __builtin_popcountll(_t[i]);
      return ttl;
   }
   void insert(int p) noexcept         { _t[p >> 6] |= (1ull << (p & 63));}
   bool contains(int p) const noexcept { return (_t[p >> 6] &  (1ull << (p & 63))) != 0;}
   void unionWith(const NatSet& ps) noexcept {
      for(short i=0;i < nbw;++i)
         _t[i] |= ps._t[i]; 
   }
   void interWith(const NatSet& ps) noexcept {
      for(short i=0;i < nbw;i++)
         _t[i] &= ps._t[i];
   }
   class iterator { 
      const unsigned long long*    _t;
      unsigned short       _cwi;    // current word index
      unsigned long long    _cw; // current word
      iterator(const unsigned long long* t,unsigned short at)
         : _t(t),_cwi(at),_cw((at < nbw) ? t[at] : 0)
      {
         while (_cw == 0 && ++_cwi < nbw) 
            _cw = _t[_cwi];
         _cw = (_cwi < nbw) * _cw;
         assert(_cwi >= nbw || _cw <= _t[_cwi]);
      }
      iterator(const unsigned long long* t) : _t(t),_cwi(nbw),_cw(0) {} // end constructor
   public:
      using iterator_category = std::forward_iterator_tag;
      using value_type = short;
      using difference_type = short;
      using pointer = short*;
      using reference = short&;
      iterator& operator++()  noexcept {
         assert(_cwi >= nbw || _cw <= _t[_cwi]);
         long long test = _cw & -_cw;  // only leaves LSB at 1
         _cw ^= test;                  // clear LSB
         while (_cw == 0 && ++_cwi < nbw)  // all bits at zero-> done with this word.            
            _cw = _t[_cwi];
         _cw = (_cwi < nbw) * _cw;
         assert(_cwi >= nbw || _cw <= _t[_cwi]);
         return *this;
      }
      iterator operator++(int) { iterator retval = *this; ++(*this); return retval;}
      bool operator==(iterator other) const {return _cwi == other._cwi && _cw == other._cw;}
      bool operator!=(iterator other) const {return !(*this == other);}
      short operator*() const   { return (_cwi<<6) + __builtin_ctzl(_cw);}
      friend class NatSet;
   };
   class citerator { 
      const unsigned long long*    _t;
      unsigned short       _cwi;    // current word index
      unsigned long long    _cw; // current word
      citerator(const unsigned long long* t,unsigned short at)
         : _t(t),_cwi(at),_cw((at < nbw) ? t[at] : 0)
      {
         while (_cw == 0 && ++_cwi < nbw) 
            _cw = _t[_cwi];         
      }
      citerator(const unsigned long long* t) : _t(t),_cwi(nbw),_cw(0) {} // end constructor
   public:
      using iterator_category = std::forward_iterator_tag;
      using value_type = short;
      using difference_type = short;
      using pointer = short*;
      using reference = short&;
      citerator& operator++()  noexcept {
         long long test = _cw & -_cw;  // only leaves LSB at 1
         _cw ^= test;                  // clear LSB
         while (_cw == 0 && ++_cwi < nbw)  // all bits at zero-> done with this word.            
            _cw = _t[_cwi];        
         return *this;
      }
      citerator operator++(int) { citerator retval = *this; ++(*this); return retval;}
      bool operator==(citerator other) const {return _cwi == other._cwi && _cw == other._cw;}
      bool operator!=(citerator other) const {return !(*this == other);}
      short operator*() const   { return (_cwi<<6) + __builtin_ctzl(_cw);}
      friend class NatSet;
   };
   iterator begin() const { return iterator(_t,0);}
   iterator end()   const { return iterator(_t);}
   citerator cbegin() const { return citerator(_t,0);}
   citerator cend()   const { return citerator(_t);}
   friend std::ostream& operator<<(std::ostream& os,const NatSet& ps) {
      os << "{";
      auto cnt = 0;
      for(auto i=ps.cbegin();i!= ps.cend();i++,cnt++)
         os << *i << ((cnt==ps.size()-1) ? "" : ",");
      return os << "}";
   }
   friend bool operator==(const NatSet& s1,const NatSet& s2) {
      unsigned short nw = 0;
      for(auto i = 0;i < nbw;i++)
         nw += s1._t[i] == s2._t[i];
      return nw == nbw;
   }
   std::size_t hash() const noexcept {
      std::size_t hv = 0;
      for(auto i = 0;i < nbw;i++)
         hv = hv ^ _t[i];
      return hv;
   }
};



class GNSet {
   unsigned short _mxw;
   unsigned short _nbp;
   unsigned long long *_t;
public:
   GNSet(unsigned short nb=64) {
      _mxw = (nb >> 6) + (((nb & 63) != 0) ? 1 : 0);
      _nbp = nb;
      if (_mxw) {
         _t   = new unsigned long long[_mxw];
         for(int i=0;i<_mxw;i++) _t[i]=0;
      } else _t = nullptr;
   }
   GNSet(const GNSet& s) {
      _mxw = s._mxw;
      _nbp = s._nbp;
      _t   = new unsigned long long[_mxw];
      for(int i=0;i<_mxw;i++)
         _t[i] = s._t[i];
   }   
   GNSet(std::initializer_list<int> l) {
      _mxw = 1;
      auto nb = (l.end() - l.begin()) >> 6;
      _nbp  = nb << 6;
      while (nb >= _mxw) _mxw <<= 1;
      _t = new unsigned long long[_mxw];
      for(auto i=0u;i < _mxw;i++)
         _t[i]=0;
      for(auto it = l.begin();it!=l.end();it++) {
         insert(*it);
      }
   }
   ~GNSet() {
      delete[] _t;
   }
   GNSet& operator=(const GNSet& s) {
      if (s._mxw != _mxw) {
         delete[] _t;
         _mxw = s._mxw;
         _t = new unsigned long long[_mxw];
      }
      _nbp = s._nbp;      
      for(int i=0;i<_mxw;i++)
         _t[i] = s._t[i];
      return *this;
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
   void insert(int p) noexcept         {
      const int i = p >> 6;
      const auto old = _mxw;
      if (i >= _mxw) {
         while(i >= _mxw) _mxw <<= 1;
         auto np = new unsigned long long[_mxw];
         for(int i=0;i < old;i++)
            np[i] = _t[i];
         for(int i=old;i < _mxw;i++)
            np[i] = 0;
         delete []_t;
         _t = np;
      }      
      _t[i] |= (1ull << (p & 63));
   }
   void remove(int p) noexcept {
      const int i = p >> 6;
      if (i < _mxw) 
         _t[i] &= ~(1ull << (p & 63));      
   }
   bool contains(int p) const noexcept { return (_t[p >> 6] &  (1ull << (p & 63))) != 0;}
   void unionWith(const GNSet& ps) noexcept {
      assert(_mxw == ps._mxw);
      switch (_mxw) {
         case 1: _t[0] |= ps._t[0];break;
            /*
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
            */
         default: {
            for(short i=0;i < _mxw;++i)
               _t[i] |= ps._t[i]; 
         }
      }
   }
   void interWith(const GNSet& ps) noexcept {
      assert(_mxw == ps._mxw);
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
      iterator& operator=(const iterator i) {
         _t = i._t;
         assert(_nbw == i._nbw);
         _cwi = i._cwi;
         _cw  = i._cw;
         return *this;
      }
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
      friend class GNSet;
   };
   iterator begin() const { return iterator(_t,_mxw,0);}
   iterator end()   const { return iterator(_t,_mxw);}
   iterator rbegin() const {
      int lw = _mxw - 1;
      while (_t[lw]==0) lw--;
      auto c = iterator(_t,_mxw,lw);
      auto last = c;
      auto found = c;
      while (c._cwi == last._cwi) {
         found = c;
         c++;         
      }
      return found;
   }
   friend std::ostream& operator<<(std::ostream& os,const GNSet& ps) {
      os << '{';
      auto cnt = 0;
      for(auto i=ps.begin();i!= ps.end();i++,cnt++)
         os << *i << ((cnt==ps.size()-1) ? "" : ",");
      return os << '}';
   }
   friend bool operator==(const GNSet& s1,const GNSet& s2) {
      if (s1._mxw == s2._mxw) {
         unsigned short nw = 0;
         for(auto i = 0;i < s1._mxw;i++)
            nw += s1._t[i] == s2._t[i];
         return nw == s1._mxw;
      } else return false;
   }
};


inline int min(const GNSet& s) {
   if (s.size() > 0)
      return *s.begin();
   else 
      return std::numeric_limits<int>::max();
}
inline int max(const GNSet& s) {
   if (s.size() > 0)
      return *(s.rbegin());
   else 
      return std::numeric_limits<int>::min();
}
template <typename Pred>
GNSet filter(const GNSet& inSet,Pred p) {
   GNSet outSet = {};
   for(const auto& v : inSet)
      if (p(v))
         outSet.insert(v);
   return outSet;
}


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

template<unsigned short sz> struct std::hash<NatSet<sz>> {
   std::size_t operator()(const NatSet<2>& v) const noexcept {
      return v.hash();
   }
};

typedef struct std::hash<NatSet<1>> NS1;
typedef struct std::hash<NatSet<2>> NS2;

template <> struct std::hash<GNSet> {
   std::size_t operator()(const GNSet& v) const noexcept {
      std::size_t ttl = 0;
      for(auto e : v)
         ttl = (ttl << 3) | std::hash<int>{}(e);
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




inline GNSet operator|(const GNSet& s1,const GNSet& s2)
{
   GNSet r(s1);
   r.unionWith(s2);
   return r;
}

inline GNSet operator&(const GNSet& s1,const GNSet& s2)
{
   GNSet r(s1);
   r.interWith(s2);
   return r;
}

inline NatSet<1> operator|(const NatSet<1>& s1,const NatSet<1>& s2)
{
   NatSet<1> r(s1);
   r.unionWith(s2);
   return r;
}

inline NatSet<1> operator&(const NatSet<1>& s1,const NatSet<1>& s2)
{
   NatSet<1> r(s1);
   r.interWith(s2);
   return r;
}

inline NatSet<2> operator|(const NatSet<2>& s1,const NatSet<2>& s2)
{
   NatSet<2> r(s1);
   r.unionWith(s2);
   return r;
}

inline NatSet<2> operator&(const NatSet<2>& s1,const NatSet<2>& s2)
{
   NatSet<2> r(s1);
   r.interWith(s2);
   return r;
}


template<class T,class B>
GNSet setFrom(const std::ranges::iota_view<T,B>& from) {
   GNSet res {};
   for(auto v : from)
      res.insert(v);
   return res;
}

#endif
