#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <list>
#include <set>
#include <functional>
#include <assert.h>

class Range {
   int _from;
   int _to;
public:
   typedef int value_type;
   class iterator {
    private:
      friend class Range;
      int _i;
   protected:
      iterator(int start) : _i(start) {}
   public:
      iterator(const iterator& i2) : _i(i2._i) {}
      iterator operator=(const iterator& i2) { _i = i2._i;return *this;}
      int operator *() const  { return _i; }
      const auto& operator+=(int d) { _i += d;return *this;}
      const auto& operator++() { _i++;return *this;} // pre-increment
      auto operator ++(int)    { iterator copy(*this);_i++;return copy;} // post-increment
      const auto& operator--() { _i--;return *this;} // pre-increment
      auto operator--(int)     { iterator copy(*this);_i--;return copy;} // post-increment
      bool operator ==(const iterator& other) const { return _i == other._i; }
      bool operator !=(const iterator& other) const { return _i != other._i; }
   };
   class const_iterator {
    private:
      friend class Range;
      int _i;
   protected:
      const_iterator(int start) : _i(start) {}
   public:
      const_iterator(const const_iterator& i2) : _i(i2._i) {}
      const_iterator operator=(const const_iterator& i2) { _i = i2._i;return *this;}
      int operator *() const  { return _i; }
      const auto& operator+=(int d) { _i += d;return *this;}
      const auto& operator++() { _i++;return *this;} // pre-increment
      auto operator ++(int)    { const_iterator copy(*this);_i++;return copy;} // post-increment
      const auto& operator--() { _i--;return *this;} // pre-increment
      auto operator--(int)     { const_iterator copy(*this);_i--;return copy;} // post-increment
      auto operator-(int d)    { const_iterator copy(*this);--copy;return copy;}
      bool operator ==(const const_iterator& other) const { return _i == other._i; }
      bool operator !=(const const_iterator& other) const { return _i != other._i; }
   };
   Range(int f,int t) : _from(f),_to(t) {}
   Range(Range&& r) : _from(r._from),_to(r._to) {}
   Range(const Range& r) : _from(r._from),_to(r._to) {}
   int size() const noexcept { return (_to >= _from) ? _to - _from : _from - _to;}
   auto largest() const noexcept { return (_to >= _from) ? _to : _from;}
   auto from() const noexcept  { return _from;}
   auto to() const noexcept    { return _to;}
   auto begin() const noexcept { return const_iterator(_from); }
   auto end()   const noexcept  { return const_iterator(_to); }
   auto rbegin() const noexcept { return const_iterator(_to-1);}
   auto rend()  const noexcept { return const_iterator(_from-1);}
   static Range open(int f,int t) noexcept  { return Range(f,t);}
   static Range close(int f,int t) noexcept { return Range(f,t+1);}   
   friend std::ostream& operator<<(std::ostream& os,const Range& r) {
      return os << "[" << r._from << " -> " << r._to << ")";
   }
};

class GNSet {
   unsigned short _mxw;
   unsigned short _nbp;
   unsigned long long *_t;
public:
   typedef int value_type;
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
      const int at = p >> 6;
      if (at >= _mxw) return false;
      assert(at >= 0 && at < _mxw);
      return (_t[at] &  (1ull << (p & 63))) != 0;
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
      short                _cwi;    // current word index
      int                  _cnt;    // rank of current bit
      unsigned long long    _cw;    // current word
      iterator(unsigned long long* t,unsigned short nbw,unsigned short at)
         : _t(t),_nbw(nbw),_cwi(at),_cnt(0),_cw((at < nbw) ? t[at] : 0) {
         while (_cw == 0 && _t && ++_cwi < _nbw) _cw = _t[_cwi];         
      }
      iterator(unsigned long long* t,unsigned short nbw,const GNSet& gns)
         : _t(t),_nbw(nbw),_cwi(nbw),_cnt(gns.size()),_cw(0) {} // end constructor
      static constexpr auto msb(unsigned long long w) noexcept {return (0x8000000000000000u >> __builtin_clzl(w));}
      static constexpr auto lsb(unsigned long long w)  noexcept {return w & -w;}
      static constexpr auto clearMSB(unsigned long long w) noexcept { return w ^ msb(w);}
      static constexpr auto clearLSB(unsigned long long w) noexcept { return w ^ lsb(w);}
      static constexpr auto bitId(unsigned long long w) noexcept { return 63 - __builtin_clzl(w);}
   public:
      using iterator_category = std::forward_iterator_tag;
      using value_type = short;
      using difference_type = short;
      using pointer = short*;
      using reference = short&;
      iterator& operator=(const iterator i) {
         _t = i._t;
         assert(_nbw == i._nbw);
         _cwi = i._cwi;
         _cnt = i._cnt;
         _cw  = i._cw;
         return *this;
      }
      iterator& operator++()  noexcept {
         long long test = _cw & -_cw;  // only leaves LSB at 1
         _cw ^= test;                  // clear LSB
         while (_cw == 0 && ++_cwi < _nbw)  // all bits at zero-> done with this word.            
            _cw = _t[_cwi];
         ++_cnt;
         return *this;
      }
      iterator& operator--() noexcept {
         if (_cwi==_nbw) { // i'm at the end. try to read a full word backward.
            while(_cw==0 && --_cwi >=0) _cw = _t[_cwi];
            assert(_cw!=0 || _cwi < 0);
         } else if (_cnt==0) {
            _cwi = -1;
            _cw  =  0;
            return *this;
         } else 
            _cw = clearMSB(_cw);
         while(_cw==0 && --_cwi >= 0) _cw = _t[_cwi];
         --_cnt;
         return *this;
      }
      iterator operator++(int) { iterator retval = *this; this->operator++(); return retval;}
      iterator operator--(int) { iterator retval = *this; this->operator--(); return retval;}
      auto& operator+=(int d) {
         while(d-- > 0) this->operator++();
         while(d++ < 0) this->operator--();
         assert(d==0);
         return *this;
      }
      iterator operator+(unsigned d) const {iterator r(*this);while(d-- > 0) ++r;return r;}
      iterator operator-(unsigned d) const {iterator r(*this);while(d-- > 0) --r;return r;}
      bool operator==(iterator other) const {return _cwi == other._cwi && _cw == other._cw;}
      bool operator!=(iterator other) const {return !(*this == other);}
      //short operator*() const   { return (_cwi<<6) + __builtin_ctzl(_cw);}
      short operator*() const   { return (_cwi<<6) + bitId(lsb(_cw));}
      //short operator*() const   { return (_cwi<<6) + bitId(msb(_cw));}
      friend class GNSet;
   };
   typedef iterator const_iterator;
   iterator begin() const { return iterator(_t,_mxw,0);}
   iterator end()   const { return iterator(_t,_mxw,*this);}
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


template <class C> 
class Ordered {
public:
   enum Dir {Increase,Decrease};
private:
   C _c; // we are making a copy of the thing we loop on.
   typedef typename C::value_type VT;
   typedef std::function<VT(VT)> Fun;
   struct E {
      VT i,fi;
   };
   std::vector<E> _p;
   Fun            _f;
   enum Dir       _d;
public:
   Ordered(const C& c,enum Dir d) : _c(c),_f(nullptr),_d(d) {}
   Ordered(C&& c,enum Dir d) : _c(std::move(c)),_f(nullptr),_d(d) {}
   Ordered(const C& c,Fun f,enum Dir d) : _c(c),_f(f),_d(d) {
      for(const auto& i : _c)
         _p.push_back(E {i,f(i)});
      std::sort(_p.begin(),_p.end(),[](const auto& a,const auto& b) {
         return a.fi < b.fi;
      });
   }
   Ordered(C&& c,Fun f,enum Dir d) : _c(std::move(c)),_f(f),_d(d) {
      for(const auto& i : _c)
         _p.push_back(E {i,f(i)});
      std::sort(_p.begin(),_p.end(),[](const auto& a,const auto& b) {
         return a.fi < b.fi;
      });
   }
   class iterator {
      static constexpr const int inc[2] = {1,-1}; 
   private:
      friend class Ordered<C>;
      union {
         std::vector<E>::const_iterator _i;
         C::const_iterator             _ri;
      };
      bool _nat:1; // native or not
      bool _dir:1; // 0 - increase, 1 : decrease
      bool     :6; // 6 unused bits      
   public:
      iterator(const Ordered<C>& oc,C::const_iterator ri,enum Dir d)
         : _ri(ri),_nat(true),_dir(d)    {}      
      iterator(const Ordered<C>& oc,std::vector<E>::const_iterator start,enum Dir d) 
         : _i(start),_nat(false),_dir(d) {}
   public:
      iterator(const Ordered<C>::iterator& i2) : _nat(i2._nat),_dir(i2._dir) {
         if (_nat)
            _ri = i2._ri;
         else _i = i2._i;
      }
      const iterator& operator=(const iterator& i2) {
         _nat = i2._nat;
         _dir = i2._dir;
         if (_nat)
            _ri = i2._ri;
         else _i = i2._i;
         return *this;
      }
      VT operator *() const  {
         return _nat ? (*_ri) : (*_i).i;
      }
      const auto& operator++() { 
         if (_nat)
            _ri += inc[_dir];
         else _i += inc[_dir];
         return *this;
      } // pre-increment
      auto operator++(int)    { 
         iterator copy(*this);
         if (_nat)
            _ri += inc[_dir];
         else _i += inc[_dir];
         return copy;
      } // post-increment      
      bool operator ==(const iterator& other) const { return (_nat) ? (_ri == other._ri) : (_i == other._i); }
      bool operator !=(const iterator& other) const { return (_nat) ? (_ri != other._ri) : (_i != other._i); }
   };
   Ordered<C>::iterator begin() const  {
      if (_f)
         return iterator(*this,(_d == Increase) ? _p.begin() : _p.end()-1,_d);
      else
         return iterator(*this,(_d == Increase) ? _c.begin() : _c.end()-1,_d);
   }
   Ordered<C>::iterator end()   const  {
      if (_f)
         return iterator(*this,(_d == Increase) ? _p.end() : _p.begin()-1,_d);
      else
         return iterator(*this,(_d == Increase) ? _c.end() : _c.begin()-1,_d);
   }
};

template <class C,typename Fun> static auto increasing(C&& c,Fun f) {
   return Ordered<C>(std::move(c),f,Ordered<C>::Increase);
}
template <class C,typename Fun> static auto decreasing(C&& c,Fun f) {
   return Ordered<C>(std::move(c),f,Ordered<C>::Decrease);
}
template <class C> static auto increasing(C&& c) {
   return Ordered<C>(std::move(c),Ordered<C>::Increase);
}
template <class C> static auto decreasing(C&& c) {
   return Ordered<C>(std::move(c),Ordered<C>::Decrease);
}

class DDGen {
public:
   typedef std::unique_ptr<DDGen> Ptr;
   DDGen() {}
   virtual ~DDGen() {}
   virtual int getAndNext() noexcept = 0;
   virtual bool more() const noexcept  = 0;
};

template <class C> class DDGenOC :public DDGen {
   Ordered<C> _oc;
   Ordered<C>::iterator _it;
   const Ordered<C>::iterator _end;
public:
   DDGenOC(Ordered<C>&& oc) : _oc(std::move(oc)),
                              _it(_oc.begin()),
                              _end(_oc.end()) {}
   ~DDGenOC() {
      std::cout << "DDGenOC::~DDGenOC " << this << "\n";
   }
   int getAndNext() noexcept {
      return *_it++;
   }
   bool more() const noexcept {
      return _it != _end;
   }
};

class DDGenRange :public DDGen {
   int _it;
   const int _end;
public:
   DDGenRange(const Range& oc) : _it(oc.from()),_end(oc.to()) {}
   ~DDGenRange() {
      std::cout << "DDGenRange::~DDGenRange " << this << "\n";
   }
   int getAndNext() noexcept {
      return _it++;
   }
   bool more() const noexcept {
      return _it != _end;
   }
};

template <class C> auto genValue(Ordered<C>&& oc) {
   return std::make_unique<DDGenOC<C>>(std::move(oc));
}

auto genValue(const Range& oc) {
   return std::make_unique<DDGenRange>(oc);
}

int main()
{
    std::cout << "[-4, 4]  increasing  v->v*v" << std::endl;
    for(auto i : increasing(Range::close(-4,4),[](int v) {return v*v;})) 
       std::cout << i << " ";
    std::cout << "\n";

    std::cout << "[-4, 4]  decreasing  v -> v*v" << std::endl;
    for(auto i : decreasing(Range::close(-4,4),[](int v) { return v*v;}))
        std::cout << i << " ";
    std::cout << "\n";

    std::cout << "[-4, 4]" << std::endl;
    for(auto i : Range::close(-4,4))
        std::cout << i << " ";
    std::cout << "\n";

    std::cout << "[-4, 4] increasing" << std::endl;
    for(auto i : increasing(Range::close(-4,4)))
        std::cout << i << " ";
    std::cout << "\n";

    std::cout << "[-4, 4] decreasing" << std::endl;
    for(auto i : decreasing(Range::close(-4,4)))
        std::cout << i << " ";
    std::cout << "\n";

    std::cout  << "vectors = ----------------------------------------------------------------------\n";
    std::cout << "[1,2,3,4,5,10,9,8,7]" << std::endl;    
    for(auto i : std::vector<int> {1,2,3,4,5,10,9,8,7})
       std::cout << i << " ";
    std::cout << "\n";
    
    std::cout << "[1,2,3,4,5,10,9,8,7] increasing" << std::endl;    
    for(auto i : increasing(std::vector<int> {1,2,3,4,5,10,9,8,7}))
       std::cout << i << " ";
    std::cout << "\n";

    std::cout << "[1,2,3,4,5,10,9,8,7] decreasing" << std::endl;    
    for(auto i : decreasing(std::vector<int> {1,2,3,4,5,10,9,8,7}))
       std::cout << i << " ";
    std::cout << "\n";
   
    std::cout << "[1,2,3,4,5,10,9,8,7] increasing i -> i" << std::endl;    
    for(auto i : increasing(std::vector<int> {1,2,3,4,5,10,9,8,7},[](auto i) { return i;}))
       std::cout << i << " ";
    std::cout << "\n";

    std::cout << "[1,2,3,4,5,10,9,8,7] increasing i -> -i" << std::endl;    
    for(auto i : increasing(std::vector<int> {1,2,3,4,5,10,9,8,7},[](auto i) { return -i;}))
       std::cout << i << " ";
    std::cout << "\n";

    std::cout << "[1,2,3,4,5,10,9,8,7] decreasing i -> i" << std::endl;    
    for(auto i : decreasing(std::vector<int> {1,2,3,4,5,10,9,8,7},[](auto i) { return i;}))
       std::cout << i << " ";
    std::cout << "\n";

    std::cout << "[1,2,3,4,5,10,9,8,7] decreasing i -> -i" << std::endl;    
    for(auto i : decreasing(std::vector<int> {1,2,3,4,5,10,9,8,7},[](auto i) { return -i;}))
       std::cout << i << " ";
    std::cout << "\n";

    std::cout << "[1..10] increasing i -> |6-i|" << std::endl;    
    for(auto i : increasing(std::vector<int> {1,2,3,4,5,6,7,8,9,10},[](auto i) { return abs(6-i);}))
       std::cout << i << " ";
    std::cout << "\n";

    auto s0 = GNSet {1,7,4,12,253};
    std::cout << s0 << "\n";
    std::cout << "forward loop:\t";
    for(auto i = s0.begin(); i!=s0.end();i++)
       std::cout << *i << " ";
    std::cout << "\n";
    std::cout << "backward loop:\t";
    auto s0E = s0.end();
    auto s0B = s0.begin();
    s0E = s0E - 1;
    s0B = s0B - 1;
    for(auto i = s0E; i!=s0B;i--)
       std::cout << *i << " ";
    std::cout << "\n";
    

    DDGen::Ptr gOrd = genValue(decreasing(std::vector<int> {1,2,3,4,5,10,9,8,7},
                                    [](auto i) { return i;}));
    while (gOrd->more()) {
       int v = gOrd->getAndNext();
       std::cout << v << " ";
    }
    std::cout << "\n";

    DDGen::Ptr gOrd2 = genValue(Range(0,10));
    while (gOrd2->more()) {
       int v = gOrd2->getAndNext();
       std::cout << v << " ";
    }
    std::cout << "\n";

    return 0;
  }
