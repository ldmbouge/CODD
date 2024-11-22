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
#include <algorithm>
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
   int size() const noexcept { return (_to >= _from) ? _to - _from : _from - _to;}
   auto largest() const noexcept { return (_to >= _from) ? _to : _from;}
   auto from() const noexcept  { return _from;}
   auto to() const noexcept    { return _to;}
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
      // for(auto i=0u;i < nbw;i++)
      //    _t[i]=0;
      // for(auto i : s)
      //    insert(ofs - i);
      // return;
      
      // unsigned long long bt[nbw];
      // for(auto i=0u;i < nbw;i++)
      //    bt[i] = _t[i];
      // for(auto i=0u;i < nbw;i++)
      //    _t[i]=0;

      // std::cout << "INPUT:\n";
      // for(int i=0; i < nbw;i++) {
      //    std::bitset<64> word(s._t[i]);
      //    std::cout << word << " ";
      // }
      // std::cout << "\n";      
      
      int lw = nbw-1;
      while(lw >= 0 && s._t[lw]==0) lw--;
      if (lw < 0) {
         for(auto i=0u;i < nbw;i++)
            _t[i]=0;
         return;
      }
      const auto lvinLW = 63 - __builtin_clzll(s._t[lw]); // index of MSB bit at 1 in word lw
      const auto lv = (lw * 64) + lvinLW; // largest value in S
      const auto dec = ofs - lv;          // Substraction  OFS - max(S)  : size of gap. Could be > 64


      for(int i=0;i <= lw;i++) 
         _t[lw-i] = std::revBitsOfLong(s._t[i]);            
      for(int i=lw+1;i < nbw;i++) _t[i] = 0;
      
      // std::cout << "FLIP:\n";
      // for(int i=0; i < nbw;i++) {
      //    std::bitset<64> word(_t[i]);
      //    std::cout << word << " ";
      // }
      // std::cout << "\n";      

            
      const auto nbs = __builtin_ffsll(_t[0])-1; // least significant bit set to 1 (nbs in 0..63)
      // std::cout << "DEC=" << dec << "\n"; 
      // std::cout << "NBS=" << nbs << "\n";
      const bool down = nbs > dec;
      if (down) {
         const auto x = nbs - dec;
         //std::cout << "MOVE Down=" << x << " lw:" << lw << "\n";
         unsigned long long nt[nbw] = {0}; 
         for(int i=0;i <= std::min(lw,nbw-1);i++) {
            auto next =  (i+1==nbw) ? 0 : _t[i+1] & ((1ull << (x+1))-1); // get the next block to inject here.
            nt[i] = (_t[i] >> x) | (next << (64 - x));
         }
         for(int i=0;i < nbw;i++) _t[i] = nt[i];         
      } else {
         auto delta = dec >> 6; // number of words to skip
         auto moveUp = dec & ((1ull << 6) -1); // remainder (number of bits to shift)
         auto useBits = 64 - nbs;
         auto by = 64 - (long)moveUp - useBits; // Could be > 0 or negative (different shift direction)
         // std::cout << "DELTA=" << delta << " MUP:" << moveUp << " USED=" << useBits << " BY:" << by << " LW=" << lw 
         //           << " NBW=" << nbw << "\n";
         if (by > 0) {
            // moving bits DOWN
            assert(lw < nbw); // by definition, it's at most the index of the last word (1 less than size)
            unsigned long long nt[nbw] = {0}; 
            for(int i=0;i <= lw;i++) {
               auto next = (i+1==nbw) ? 0 : _t[i+1] & ((1ull << (by+1))-1);
               nt[i+delta] = (_t[i] >> by) | (next << (64 - by));
               for(int j=i;j < i+delta;j++) _t[j] = 0;
            }
            for(int i=0;i < nbw;i++) _t[i] = nt[i];
         } else if (by < 0) {
            //moving bits UP
            unsigned long long next = 0;
            unsigned long long nt[nbw] = {0}; 
            for(int i=0;i <= std::min(lw+1,nbw-1);i++) {
               nt[i] = (_t[i] << (-by)) | next; 
               next = (_t[i] & ~((1ull << (64 + by))-1)) >> (64+by); // from MSB w(i) to LSB w(i+1)
               //std::cout << "NT " << i << " : " << std::bitset<64>(nt[i]) << "\n";
            }            
            for(int i=0;i < delta;i++)   _t[i] = 0;
            for(int i=delta;i < nbw;i++) _t[i] = nt[i-delta]; 
         } else if (delta) {
            assert(delta >= 0 && delta < nbw);
            unsigned long long nt[nbw] = {0}; 
            for(int i=delta;i < nbw;i++) 
               nt[i] = _t[i-delta];            
            for(int i=0;i < nbw;i++) _t[i] = nt[i];            
         }
      }
      // for(auto i=0u;i < nbw;i++)
      //    if (bt[i] != _t[i]) {
      //       std::cout << "Oopsies " << ofs << " - " << s << "\n";
      //       for(auto j=0u;j < nbw;j++)
      //          std::cout << std::bitset<64>(bt[j]) << " " << std::bitset<64>(_t[j]) << " " << (bt[j] == _t[j]) << "\n";
      //       abort();
      //    }      
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
   int largest() const noexcept { // returns -1 when empty
      int lw = nbw-1;
      while(lw >= 0 && _t[lw]==0) lw--;
      if (lw < 0) return -1;
      const auto lvinLW = 63 - __builtin_clzll(_t[lw]); // index of MSB bit at 1 in word lw
      const auto lv = (lw * 64) + lvinLW; // largest value in S
      return lv;
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
   friend class GNSet;
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
   template <unsigned short nbw> GNSet(const NatSet<nbw>& s) {     
      _mxw = nbw;
      _t   = new unsigned long long[_mxw];
      for(int i=0;i < _mxw;i++) _t[i] = s._t[i];
      _nbp = s.largest();
   }
   GNSet(const Range& s) {
      auto sz = s.size(), ub = s.largest();
      if (sz==0) {
         _mxw = 0;
         _nbp = 0;
         _t = nullptr;
      } else {
         _mxw = ((ub & 0x3f)? 1 : 0) + (ub >> 6); // if remainder 1 : 0 + division
         _nbp = _mxw << 6;
         assert(_mxw != 0);
         _t = new unsigned long long[_mxw];
         for(int i=0;i<_mxw;i++) _t[i]=0;
         for(auto i : s)
            insert(i);
      }
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
   bool empty() const noexcept {
      int ttl = 0;
      for(short i=0;(ttl==0) && i < _mxw;++i)
         ttl += (_t[i] != 0);
      return ttl==0;
   }
   int size() const noexcept {
      int ttl = 0;
      for(short i=0;i < _mxw;++i)
         ttl += __builtin_popcountll(_t[i]);
      return ttl;
   }
   GNSet& insert(int p) noexcept         {
      assert(p >=0);
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
      assert(i>=0 && i < _mxw);
      _t[i] |= (1ull << (p & 63));
      _nbp = std::max(_nbp,(unsigned short)p);
      return *this;
   }
   GNSet& remove(int p) noexcept {
      const int i = p >> 6;
      if (0 <= i && i < _mxw) 
         _t[i] &= ~(1ull << (p & 63));
      return *this;
   }
   GNSet& complement() noexcept {
      for(short i=0;i < _mxw;++i)
         _t[i] = ~_t[i];
      return *this;
   }
   bool contains(int p) const noexcept {
      return (_t[p >> 6] &  (1ull << (p & 63))) != 0;
   }
   GNSet& unionWith(const GNSet& ps) noexcept {
      short i;
      for(i=0;i < std::min(_mxw,ps._mxw);i++)
         _t[i] |= ps._t[i];
      if (_mxw < ps._mxw) {
         auto nt = new unsigned long long[ps._mxw];
         for(i=0;i < _mxw;i++)
            nt[i] = _t[i];
         for(i=_mxw;i < ps._mxw;i++)
            nt[i] = ps._t[i];
         delete[] _t;
         _t = nt;
         _mxw = ps._mxw;
         _nbp = ps._nbp;        
      }
      return *this;
   }
   GNSet& interWith(const GNSet& ps) noexcept {
      short i;
      for(i=0;i < std::min(_mxw,ps._mxw);i++)
         _t[i] &= ps._t[i];
      while(i < _mxw)
         _t[i++] = 0;
      return  *this;
   }
   GNSet& diffWith(const GNSet& ps) noexcept {
      for(short i=0;i < std::min(_mxw,ps._mxw);i++)
         _t[i] = _t[i] & ~ps._t[i];
      return *this;
   }
   void removeAbove(int p) noexcept {
      int w = ++p >> 6;
      if (p > _nbp) return;
      assert(w >= 0 && w < _mxw);
      const int b = p & 63;
      _t[w] = _t[w] & ((1ull << b) - 1);
      w++;
      while(w < _mxw) 
        _t[w++] = 0;
   }
   void removeBelow(int p) noexcept {
      for(int i = 0;i < p;i++)
         remove(i);
      return;
      if (p < 0) return;
      if (p > _nbp) {
         for(int i=0;i<_mxw;i++) _t[i]=0;
      } else {
         int w = p >> 6;
         if (w >= _mxw) abort();
         assert(w >= 0 && w < _mxw);
         const int b = p & 63;
         _t[w] = _t[w] & ~((1ull << (b)) - 1);
         w--;
         while (w >= 0)
            _t[w--] = 0;
      }
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
      iterator(unsigned long long* t,unsigned short nbw)
         : _t(t),_nbw(nbw),_cwi(nbw),_cw(0) {} // end constructor
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
      } else {
         const auto ub = std::min(s1._mxw,s2._mxw);
         for(auto i=0;i < ub;i++)
            if (s1._t[i] != s2._t[i])
               return false;
         if (s1._mxw < s2._mxw) {
            for(auto i = s1._mxw;i < s2._mxw;i++)
               if (s2._t[i]!=0) return false;
         } else {
            for(auto i = s2._mxw;i < s1._mxw;i++)
               if (s1._t[i]!=0) return false;
         }
         return true;
      }
   }
   friend bool operator!=(const GNSet& s1,const GNSet& s2) {
      if (s1._mxw == s2._mxw) {
         for(auto i=0;i < s1._mxw;i++)
            if (s1._t[i] != s2._t[i])
               return true;
         return false;
      } else {
         const auto ub = std::min(s1._mxw,s2._mxw);
         for(auto i=0;i < ub;i++)
            if (s1._t[i] != s2._t[i])
               return true;
         bool diff = false;
         if (s1._mxw < s2._mxw)
            for(auto i = s1._mxw;i < s2._mxw;i++)
               diff = diff || (s2._t[i]!=0);
         else
            for(auto i = s2._mxw;i < s1._mxw;i++)
               diff = diff || (s1._t[i]!=0);
         return diff;
      }
  }
   std::size_t hash() const noexcept {
      std::size_t hv = 0;
      for(auto i = 0;i < _mxw;i++)
         hv = std::rotl(hv,2) ^ _t[i];
      return hv;
   }   
   friend GNSet operator|(const GNSet& s1,const GNSet& s2) { return std::move(GNSet(s1).unionWith(s2));}
   friend GNSet operator&(const GNSet& s1,const GNSet& s2) { return std::move(GNSet(s1).interWith(s2));}
   friend GNSet operator-(const GNSet& s1,const GNSet& s2) { return std::move(GNSet(s1).diffWith(s2));}
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
template <typename Term=int(*)(int)>
int sum(const GNSet& inSet,const Term& t) {
   int ttl = 0;
   for(auto v : inSet)
      ttl += t(v);
   return ttl;
}

template <typename Filter=bool(*)(int),typename Term=int(*)(int)>
int min(const GNSet& inSet,const Filter& f,const Term& t) {
   int ttl = std::numeric_limits<int>::max();
   for(auto v : inSet)
      if (f(v)) {
         const auto tv = t(v);
         ttl = (ttl  < tv) ? ttl : tv;
      }
   return ttl;
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
   for(const T& v : msg) 
      os  << v << " ";   
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
   typedef T ValType;
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

// --------------------------------------------------------------------------------

template <class FAT,int arity> class  FMatrix;
template <class FAT,int arity> class  FMatrixProxyCst;

template <class FAT,int arity> class  FMatrixProxyCst {
   friend class FMatrix<FAT,arity+1>;
   friend class FMatrixProxyCst<FAT,arity+1>;
   FAT&           _flat;
   const int*      _sfx;
   int             _acc;
   FMatrixProxyCst(FAT& flat,const int* sfx,int acc) : _flat(flat),_sfx(sfx),_acc(acc) {}
public:
   FMatrixProxyCst<FAT,arity-1> operator[](const int idx);
   const FMatrixProxyCst<FAT,arity-1> operator[](const int idx) const;
};

template <class FAT> class  FMatrixProxyCst<FAT,1> {
   friend class FMatrix<FAT,2>;
   friend class FMatrixProxyCst<FAT,2>;
   FAT&            _flat;
   const int*       _sfx;
   int              _acc;
   FMatrixProxyCst(FAT& flat,const int* sfx,int acc) : _flat(flat),_sfx(sfx),_acc(acc) {}
public:
   typename FAT::ValType& operator[](const int idx)  { return _flat[_acc * *_sfx + idx ];}
   typename FAT::ValType  operator[](const int idx) const  { return _flat[_acc * *_sfx + idx];}
};

template <class FAT,int arity>
FMatrixProxyCst<FAT,arity-1> FMatrixProxyCst<FAT,arity>::operator[](const int idx)
{
   return FMatrixProxyCst<FAT,arity-1>(_flat,_sfx+1,_acc * *_sfx + idx);   
}
template <class FAT,int arity>
const FMatrixProxyCst<FAT,arity-1> FMatrixProxyCst<FAT,arity>::operator[](const int idx) const
{
   return FMatrixProxyCst<FAT,arity-1>(_flat,_sfx+1,_acc * *_sfx + idx);   
}

inline size_t prodOf(const int* t,size_t sz) {
   auto i=0u;
   size_t fsz = 1;
   while(i < sz) fsz = fsz * t[i++];
   return fsz;
}

template <class FAT,int arity> class  FMatrix {
protected:
   FAT _flat;
   int _dims[arity];
   void print(std::ostream& os,int* path,int dId,int ofs) const;
   void prepare(const int* dims);
public:
   FMatrix() {}
   FMatrix(const FMatrix<FAT,arity>& mtx);
   FMatrix(const int* dims);
   FMatrix<FAT,arity>& operator=(const FMatrix<FAT,arity>& mtx);
   FMatrixProxyCst<FAT,arity-1> operator[](const int idx);
   const FMatrixProxyCst<FAT,arity-1> operator[](const int idx) const;
   FAT getFlat()           { return _flat;}
   int getArity() const    { return arity;}
   int getDim(int d) const { return _dims[d];}
   void print(std::ostream& os) const { int* path = (int*)alloca(sizeof(int)*arity);print(os,path,0,0);}
   friend std::ostream& operator<<(std::ostream& os,const FMatrix<FAT,arity>& m) { m.print(os);return os;}
};


template <class FAT,int arity> void FMatrix<FAT,arity>::prepare(const int* dims)
{
   _flat  = FAT(prodOf(dims,arity));
   for(int k=0;k<arity;k++)
      _dims[k] = dims[k];
}

template <class FAT,int arity> FMatrix<FAT,arity>::FMatrix(const FMatrix<FAT,arity>& mtx)
{
   _flat = mtx._flat;
   for(int k=0;k<arity;k++)
      _dims[k] = mtx._dims[k];   
}

template <class FAT,int arity> FMatrix<FAT,arity>::FMatrix(const int* dims)
{   
   _flat  = FAT(prodOf(dims,arity));
   for(int k=0;k<arity;k++)
      _dims[k] = dims[k];
}

template <class FAT,int arity> 
FMatrix<FAT,arity>& FMatrix<FAT,arity>::operator=(const FMatrix<FAT,arity>& mtx)
{
   _flat = mtx._flat;
   for(int k=0;k<arity;k++)
      _dims[k] = mtx._dims[k];   
   return *this;
}

template <class FAT,int arity>
FMatrixProxyCst<FAT,arity-1> FMatrix<FAT,arity>::operator[](const int idx)
{
   return FMatrixProxyCst<FAT,arity-1>(_flat,_dims+1,idx);
}

template <class FAT,int arity>
const FMatrixProxyCst<FAT,arity-1> FMatrix<FAT,arity>::operator[](const int idx) const
{
   return FMatrixProxyCst<FAT,arity-1>(_flat,_dims+1,idx);
}

template <class FAT,int arity> 
void FMatrix<FAT,arity>::print(std::ostream& os,int* path,int dId,int ofs) const 
{
   if (dId < arity) {
      int mult = dId > 0 ? _dims[dId] : 0;
      for(int k=0;k < _dims[dId];++k) {
         path[dId] = k;
         print(os,path,dId+1,ofs * mult + k);
      }
   } else {
      os << "m";
      for(int k=0;k<arity;k++) 
         os << '[' << path[k] << ']';
      os << " = " << _flat[ofs] << std::endl;
   }
}

// ================================================================================
// Matrix definition
// ================================================================================

template <class T,int arity> class  Matrix :public FMatrix<FArray<T>,arity> {
public:
   Matrix() {}
};

template <class T> class  Matrix<T,2> :public FMatrix<FArray<T>,2> {
public:
   Matrix() {}
   Matrix(int nbr,int nbc) : FMatrix<FArray<T>,2>() {
      int rt[] = {nbr,nbc};
      this->prepare(rt);
   }
};

#endif
