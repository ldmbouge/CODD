#ifndef __UTIL_H
#define __UTIL_H

#include <iostream>
#include <vector>
#include <tuple>
#include <set>
#include <limits>
#include <initializer_list>
#include <ranges>
#include <bit>
#include <assert.h>
//#if defined(__x86_64__)
//#include <intrin.h>
//#endif

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
   template <class Container,class AccType,typename Fop>
   AccType foldl(const Container& c,Fop op,AccType acc) {
      for(const auto& v : c) 
         acc = op(acc,v);
      return acc;
   }
   inline unsigned char revBitsOfByte(unsigned char b) {
      b = ((b * 0x80200802ULL) & 0x0884422110ULL) * 0x0101010101ULL >> 32;
      return b;
   }
   inline unsigned long long revBitsOfWord(unsigned long long w) {
      unsigned long long out;
      unsigned char* sw = (unsigned char*)&w;
      unsigned char* dw = (unsigned char*)&out;
      for(int i=0;i < 7;i++)
         dw[i] = revBitsOfByte(sw[7-i]);
      return out;
   }
   inline constexpr unsigned long long revBitsOfLong(unsigned long long v) noexcept {
      // swap odd and even bits
      v = ((v >> 1) & 0x5555555555555555) | ((v & 0x5555555555555555) << 1);
      // swap consecutive pairs
      v = ((v >> 2) & 0x3333333333333333) | ((v & 0x3333333333333333) << 2);
      // swap nibbles ... 
      v = ((v >> 4) & 0x0F0F0F0F0F0F0F0F) | ((v & 0x0F0F0F0F0F0F0F0F) << 4);
      // swap bytes
      v = ((v >> 8) & 0x00FF00FF00FF00FF) | ((v & 0x00FF00FF00FF00FF) << 8);
      // swap 2-byte long pairs
      v = ((v >> 16) & 0x0000FFFF0000FFFF) | ((v & 0x0000FFFF0000FFFF) << 16);
      // swap 4-byte ints
      v = ( v >> 32             ) | ( v               << 32);
      return v;
   }
};

class Range {
   constexpr static int inc[2] = {+1,-1};
   int _from;
   int _to;
public:
   enum  Dir {Forward,Backward};
   class iterator {
    private:
      friend class Range;
      const enum Dir _d;
      int _i;
   protected:
      iterator(enum Dir d,int start) : _d(d),_i(start) {}
   public:
      int operator *() const  { return _i; }
      const iterator& operator++() { _i += inc[_d];return *this;} // pre-increment
      iterator operator ++(int)    { iterator copy(*this);_i += inc[_d];return copy;} // post-increment
      bool operator ==(const iterator& other) const { return _i == other._i; }
      bool operator !=(const iterator& other) const { return _i != other._i; }
   };
   Range(int f,int t) : _from(f),_to(t) {}
   Range(Range&& r) : _from(r._from),_to(r._to) {}
   Range(const Range& r) : _from(r._from),_to(r._to) {}
   bool contains(int p) const noexcept { return _from <= p && p <= _to;}
   int size() const noexcept { return (_to >= _from) ? _to - _from + 1 : _from - _to + 1;}
   auto flip() const noexcept  { return Range(_to -1,_from-1);}
   auto begin() const noexcept { return iterator(_from <= _to ? Forward : Backward,_from); }
   auto end() const  noexcept  { return iterator(_from <= _to ? Forward : Backward,_to); }
   static Range open(int f,int t) noexcept  { return Range(f,t);}
   static Range close(int f,int t) noexcept { return Range(f,t+inc[f > t]);}
   static Range openInc(int f,int t) noexcept  { return (f <= t) ? Range(f,t) : Range(0,0);}
   static Range closeInc(int f,int t) noexcept { return (f < t+1) ? Range(f,t+inc[f > t]) : Range(0,0);}
   friend std::ostream& operator<<(std::ostream& os,const Range& r) {
      return os << "[" << r._from << " -> " << r._to << ")";
   }
};

/**
 * Bounded Set for Naturals [0..64*nbw)
 * It does not resizes and all storage is self-contained.
 * It is a bit-vector implementation (64 values to a double-word).
 * insert / remove / contains are O(1)
 * size and operator== are O(1)   (provided a constant number of words, otherwise Theta(n/64)
 * unionWith / interWith are O(1) (provided a constant number of words, otherwise Theta(n/64)
 */
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
   NatSet(NatSet&& s) {
      memcpy(_t,s._t,sizeof(_t));
   }
   NatSet(int lb,int ub) {
      if (lb > ub) {
         for(auto i=0u;i < nbw;i++)
            _t[i]=0;
      } else {
         ++ub;
         assert(ub >= 0);
         assert(lb <= ub);
         assert((ub >> 6) < nbw);
         for(int i=0;i<nbw;i++) _t[i]=0;
         for(auto i = lb;i < ub;i++) {
            const int ix = i >> 6;
            assert(ix >= 0 && ix < nbw);
            _t[ix] |= (1ull << (i & 63));         
         }
      }
   }
   NatSet(int ofs,const NatSet& s) {  // returns { ofs - v | v in S }
      int lw = nbw-1;
      while(lw >= 0 && s._t[lw]==0) lw--;
      if (lw < 0) {
         for(auto i=0u;i < nbw;i++)
            _t[i]=0;
         return;
      }
      const auto lvinLW = 63 - __builtin_clzll(s._t[lw]);
      const auto lv = (lw * 64) + lvinLW;
      const auto dec = ofs - lv;
      switch(lw) {
         case 0: {
            _t[0] = std::revBitsOfLong(s._t[0]) >> (63 - lv);
            for(int i=lw+1;i < nbw;i++) _t[i] = 0; 
         }break;
         default: {
            for(int i=0;i <= lw;i++)
               _t[lw-i] = std::revBitsOfLong(s._t[i]);
            for(int i=lw+1;i < nbw;i++) _t[i] = 0;  
            const auto nbs = __builtin_ffsll(_t[0])-1;
            unsigned long long inb = 0;
            for(int i=lw;i >= 0;i--) {
               const auto ds = _t[i] & ((1ull << nbs)-1);
                _t[i] = (_t[i] >> nbs) | (inb << (64 - nbs));
                inb = ds;
            }
         }
      }
      unsigned long long inb = 0;
      auto mask = (dec < 64) ? (1ull << dec)-1 : 0xffffffffffffffff;
      for(int i=0;i< nbw;i++) {
         unsigned long long outb = std::rotl(_t[i],dec) & mask;
         _t[i] = inb | ((dec < 64)*(_t[i] << dec));
         inb = outb;
      }
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
      memcpy(_t,s._t,sizeof(_t));
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
   bool empty() const noexcept {
      bool az = true;
      for(short i=0;i < nbw;++i)
         az &= (_t[i] == 0);
      return az;
   }
   void insert(int p) noexcept         { _t[p >> 6] |= (1ull << (p & 63));}
   bool contains(int p) const noexcept { return (_t[p >> 6] &  (1ull << (p & 63))) != 0;}
   NatSet& complement() noexcept {
      for(short i=0;i < nbw;++i)
         _t[i] = ~_t[i];
      return *this;
   }
   NatSet& unionWith(const NatSet& ps) noexcept {
      for(short i=0;i < nbw;++i)
         _t[i] |= ps._t[i];
      return *this;
   }
   NatSet& interWith(const NatSet& ps) noexcept {
      for(short i=0;i < nbw;i++)
         _t[i] &= ps._t[i];
      return *this;
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
   friend NatSet operator-(int l,const NatSet& s2) noexcept            { return NatSet(l,s2);}
   friend NatSet operator|(const NatSet& s1,const NatSet& s2) noexcept { return NatSet(s1).unionWith(s2);}
   friend NatSet operator&(const NatSet& s1,const NatSet& s2) noexcept { return NatSet(s1).interWith(s2);}
   std::size_t hash() const noexcept {
      std::size_t hv = 0;
      for(auto i = 0;i < nbw;i++)
         hv = hv ^ _t[i];
      return hv;
   }
   friend struct std::hash<NatSet<nbw>>;
};


/**
 * General Set for Naturals [0..+MAX_INT)
 * It automatically resizes (on the C++ Heap) to adapt to insertions.
 * It is a bit-vector implementation (64 values to a double-word).
 * insert / remove / contains are O(1)
 * size and operator== are O(1)   (provided a constant number of words, otherwise Theta(n/64)
 * unionWith / interWith are O(1) (provided a constant number of words, otherwise Theta(n/64)
 */
class GNSet {
   unsigned short _mxw;
   unsigned short _nbp;
   unsigned long long *_t;
public:
   GNSet(unsigned short nb=64) { // 64 values ... so 0 .. 63 -> 1 word (65 values -> 0..64 -> 2 words)
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
   GNSet(GNSet&& s) {
      _mxw = s._mxw;
      _nbp = s._nbp;
      _t = s._t;
      s._t = nullptr;      
   }
   GNSet(int lb,int ub) {
      //auto nb = ub >>  6;
      //_mxw = nb + ((ub & 0x3F) ? 1 : 0);
      if (lb > ub) {
         _mxw = 0;
         _nbp = 0;
         _t   = nullptr;
      } else {
         ++ub;
         assert(ub >= 0);
         assert(lb <= ub);
         _mxw = (ub >> 6) + (((ub & 63) != 0) ? 1 : 0);
         //_mxw = (1 << (64 - __builtin_clzll(ub))) >> 6;
         assert(_mxw > 0);
         _nbp = _mxw << 6;
         _t = new unsigned long long[_mxw];
         for(int i=0;i<_mxw;i++) _t[i]=0;
         for(auto i = lb;i < ub;i++) {
            const int ix = i >> 6;
            assert(ix >= 0 && ix < _mxw);
            _t[ix] |= (1ull << (i & 63));         
         }
      }
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
      if (_t) delete[] _t;
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
   GNSet& operator=(GNSet&& s) {
      if (_t) delete[] _t;      
      _mxw = s._mxw;
      _nbp = s._nbp;
      _t = s._t;
      s._t = nullptr;
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
      assert(i>=0 && i < _mxw);
      if (i < _mxw) 
         _t[i] &= ~(1ull << (p & 63));      
   }
   bool contains(int p) const noexcept { return (_t[p >> 6] &  (1ull << (p & 63))) != 0;}
   GNSet& unionWith(const GNSet& ps) noexcept {
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
      return *this;
   }
   GNSet& interWith(const GNSet& ps) noexcept {
      assert(_mxw == ps._mxw);
      for(short i=0;i < _mxw;i++)
         _t[i] &= ps._t[i];
      return  *this;
   }
   void removeAbove(int p) noexcept {
      int w = ++p >> 6;
      assert(w >= 0 && w < _mxw);
      const int b = p & 63;
      _t[w] = _t[w] & ((1ull << b) - 1);
      w++;
      while(w < _mxw) 
        _t[w++] = 0;
   }
   class iterator { 
      unsigned long long*    _t;
      const unsigned short _nbw;
      unsigned short       _cwi;    // current word index
      unsigned long long    _cw; // current word
      iterator(unsigned long long* t,unsigned short nbw,unsigned short at)
         : _t(t),_nbw(nbw),_cwi(at),_cw((at < nbw) ? t[at] : 0) {
         while (_cw == 0 && _t && ++_cwi < _nbw) 
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
         int  c = s1._mxw;
         auto a = s1._t,b = s2._t;
         while(c-- && *a++ == *b++);
         return c<0;
      } else return false;
   }
   std::size_t hash() const noexcept {
      std::size_t hv = 0;
      for(auto i = 0;i < _mxw;i++)
         hv = std::rotl(hv,2) ^ _t[i];
      return hv;
   }   
   friend GNSet operator|(const GNSet& s1,const GNSet& s2) { return std::move(GNSet(s1).unionWith(s2));}
   friend GNSet operator&(const GNSet& s1,const GNSet& s2) { return std::move(GNSet(s1).interWith(s2));}
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
   os << "(" << msg.size() << ")[";
   //int i = 0;
   for(const T& v : msg) {
      //os << i++ << ':' << v << " ";
      os  << v << " ";
   }
   return os << ']';
}

template<class T> struct std::hash<std::vector<T>> {
   std::size_t operator()(const std::vector<T>& v) const noexcept {
      std::size_t ttl = 0;
      for(const auto& e : v)
         ttl = std::rotl(ttl,3) ^ std::hash<T>{}(e);
      return ttl;
   }   
};

template<class T> struct std::hash<std::set<T>> {
   std::size_t operator()(const std::set<T>& v) const noexcept {
      std::size_t ttl = 0;
      for(const auto& e : v)
         ttl = (ttl << 3) | std::hash<T>{}(e);
      return ttl;
   }
};

template<unsigned short sz> struct std::hash<NatSet<sz>> {
   std::size_t operator()(const NatSet<sz>& v) const noexcept {
      return v.hash();
   }
};

// Hash for General Nat Set
template <> struct std::hash<GNSet> {
   std::size_t operator()(const GNSet& v) const noexcept { return v.hash();}
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

template<class T,class B>
GNSet setFrom(const std::ranges::iota_view<T,B>& from) {
   GNSet res {};
   for(auto v : from)
      res.insert(v);
   return res;
}


template <class T> class FArray {
   T*            _tab;
   std::size_t    _mx;
public:
   FArray() : _tab(nullptr),_mx(0) {}
   FArray(std::size_t isz) : _mx(isz) {
      if (_mx > 0)
         _tab = new T[_mx];
      else _tab = nullptr;
   }
   FArray(std::size_t isz,const T& value) : _mx(isz) {
      _tab = new T[_mx];
      for(auto i=0u;i < _mx;i++)
         _tab[i] = value;
   }
   FArray(const FArray& t) : _mx(t._mx) {
      _tab = new T[_mx];
      for(auto i=0u;i < _mx;i++)
         _tab[i] = t._tab[i];
   }
   FArray(FArray&& t) : _tab(t._tab),_mx(t._mx) {
      t._tab = nullptr;
   }
   ~FArray() { if (_tab) delete[] _tab;}
   FArray& operator=(const FArray& t) {
      if (_mx != t._mx) {
         delete []_tab;
         _mx = t._mx;
         _tab = new T[_mx];
      }
      for(auto i=0u;i < _mx;i++)
         _tab[i] = t._tab[i];      
      return *this;
   }
   FArray& operator=(FArray&& a) {
      if (_tab) delete[] _tab;
      _mx = a._mx;
      _tab = a._tab;
      a._tab = nullptr;
      return *this;
   }
   std::size_t size() const noexcept { return _mx;}
   T& operator[](std::size_t i) noexcept { assert(i>=0 && i < _mx);return _tab[i];}
   const T& operator[](std::size_t i) const noexcept { assert(i>=0 && i < _mx);return _tab[i];}
   class iterator { 
      T* const      _data;
      std::size_t    _num;
      iterator(const FArray* d,std::size_t num) : _data(d->_tab),_num(num) {}
   public:
      using iterator_category = std::input_iterator_tag;
      using value_type = T;
      using difference_type = long;
      using pointer = T*;
      using reference = T&;
      iterator& operator++()   { ++_num;return *this;}
      iterator operator++(int) { iterator retval = *this; ++(*this); return retval;}
      bool operator==(iterator other) const noexcept { return _num == other._num;}
      bool operator!=(iterator other) const noexcept { return _num != other._num;;}
      T operator*() const noexcept {return _data[_num];}
      T& operator*() noexcept {return _data[_num];}
      friend class FArray;
   };
   iterator begin() const noexcept { return iterator(this,0);}
   iterator end()   const noexcept { return iterator(this,_mx);}
   friend bool operator==(const FArray& a1,const FArray& a2) {
      if (a1._mx == a2._mx) {
         int  c = a1._mx;
         auto a = a1._tab,b = a2._tab;
         while(c-- && *a++ == *b++);
         return c<0;
      } else return false;
   }
   friend struct std::hash<FArray<T>>;
   friend std::ostream& operator<<(std::ostream& os,const FArray<T>& msg) {
   os << '(' << msg.size() << ")[";
   for(auto i = 0u ;i < msg._mx;i++)
      //   for(const T& v : msg)
      os << i << ':' << msg._tab[i] << " ";
   return os << ']';
}

};

template<class T> struct std::hash<FArray<T>> {
   std::size_t operator()(const FArray<T>& v) const noexcept {
      std::size_t ttl = 0;
      for(auto i=0u;i < v._mx;i++) 
         ttl = std::rotl(ttl,3) ^ std::hash<T>{}(v._tab[i]);
      return ttl;
   }   
};


#endif
