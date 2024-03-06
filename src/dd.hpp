#ifndef __DD_HPP__
#define __DD_HPP__

#include <algorithm>
#include "node.hpp"
#include <vector>
#include <list>
#include <optional>
#include <set>
#include <functional>
#include "lighthash.hpp"
#include "util.hpp"
#include "msort.hpp"

class Strategy;
class AbstractDD;

class Bounds {
   double _primal;
   std::vector<int> _inc;
   std::function<void(const std::vector<int>&)> _checker;
public:
   Bounds() {}
   Bounds(std::function<void(const std::vector<int>&)> checker) : _checker(checker) {}
   Bounds(std::shared_ptr<AbstractDD> dd);
   void attach(std::shared_ptr<AbstractDD> dd);
   void setPrimal(double p) { _primal = p;}
   double getPrimal() const { return _primal;}
   void setIncumbent(auto begin,auto end) {
      _inc.clear();
      for(auto it = begin;it != end;it++)
         _inc.push_back(*it);
      if (_checker)
         _checker(_inc);
   }
   friend std::ostream& operator<<(std::ostream& os,const Bounds& b) {
      return os << "<P:" << b._primal << /* "," << " D:" << b._dual << */ ", INC:" << b._inc << ">";
   }
};

class AbstractDD {
protected:
   Pool::Ptr _mem;
   ANode::Ptr _root;
   ANode::Ptr _trg;
   GNSet   _labels;
   ANList      _an;
   bool _exact;
   PoolMark _baseline;
   void addArc(Edge::Ptr e);
   friend class Strategy;
   friend class Exact;
   friend class Restricted;
   friend class Relaxed;
   Strategy* _strat;
   virtual bool eqSink(ANode::Ptr s) const = 0;
   virtual bool eq(ANode::Ptr f,ANode::Ptr s) const = 0;
   void computeBest(const std::string m);
   void computeBestBackward(const std::string m);
   void saveGraph(std::ostream& os,std::string gLabel);
public:
   typedef std::shared_ptr<AbstractDD> Ptr;
   AbstractDD(const GNSet& labels);
   virtual ~AbstractDD();
   virtual void reset() = 0;
   virtual ANode::Ptr init() = 0;
   virtual ANode::Ptr target() = 0;
   virtual ANode::Ptr transition(ANode::Ptr src,int label) = 0;
   virtual ANode::Ptr merge(const ANode::Ptr first,const ANode::Ptr snd) = 0;
   virtual double cost(ANode::Ptr src,int label) = 0;
   virtual ANode::Ptr duplicate(const ANode::Ptr src) = 0;
   virtual double initialBest() const = 0;
   virtual double initialWorst() const = 0;
   virtual bool   isBetter(double obj1,double obj2) const = 0;
   virtual double better(double obj1,double obj2) const = 0;
   virtual void update(Bounds& bnds) const = 0;
   virtual void printNode(ANode::Ptr n) const = 0;
   virtual GNSet getLabels(ANode::Ptr src) const = 0;
   double currentOpt() const { return _trg->getBound();}
   std::vector<int> incumbent();
   void compute();
   std::vector<ANode::Ptr> computeCutSet();
   void print(std::ostream& os,std::string gLabel);
   void setStrategy(Strategy* s);
   void display();
   bool isExact() const { return _exact;}
   virtual AbstractDD::Ptr duplicate() = 0;
   virtual void makeInitFrom(ANode::Ptr src) {}
};

class Strategy {
protected:
   AbstractDD* _dd;
   friend class AbstractDD;
   GNSet remainingLabels(ANode::Ptr p);
public:
   Strategy() : _dd(nullptr) {}
   virtual const std::string getName() const = 0;
   virtual void compute() {}
   virtual std::vector<ANode::Ptr> computeCutSet() { return std::vector<ANode::Ptr> {};}
   virtual bool primal() const { return false;}
   virtual bool dual() const { return false;}
};

class Exact:public Strategy {
public:
   Exact() : Strategy() {}
   const std::string getName() const { return "Exact";}
   void compute();
   bool primal() const { return true;}
   bool dual() const { return true;}
};

template <class T> class CQueue;

class NDArray {
   ANode::Ptr*   _tab;
   std::size_t    _mx;
   std::size_t    _sz;
   std::size_t    _st;
   std::size_t _holes;
public:
   NDArray(std::size_t isz=128) : _mx(isz),_sz(0),_st(0),_holes(0) {
      _tab = new ANode::Ptr[_mx];
   }
   ~NDArray() { delete[] _tab;}
   std::size_t size() const noexcept { return _sz - _st - _holes;}
   void push_back(ANode::Ptr n) noexcept {
      if (_sz == _mx) {
         auto p = new ANode::Ptr[_mx << 1];
         for(size_t i = 0;i < _sz;i++)
            p[i] = _tab[i];
         delete[] _tab;
         _tab = p;
         _mx <<= 1;
      }
      _tab[_sz++] = n;
   }
   void push_front(ANode::Ptr n) noexcept {
      assert(_st>0);
      if (_st == 0)
         push_back(n);
      else
         _tab[--_st] = n;
   }
   ANode::Ptr operator[](std::size_t i) const noexcept { return _tab[i];}
   void clear() noexcept { _sz = 0;_st = 0;_holes = 0;} 
   class iterator { 
      NDArray*            _nd;
      std::size_t        _num;
      std::size_t        _end;
      iterator(NDArray* d,std::size_t num) : _nd(d),_num(num),_end(d->_sz) {
         while(_num < _end && _nd->_tab[_num] == nullptr) ++_num;
      }
   public:
      using iterator_category = std::input_iterator_tag;
      using value_type = ANode::Ptr;
      using difference_type = long;
      using pointer = ANode::Ptr*;
      using reference = ANode::Ptr&;
      iterator& operator++()   {
         _num = _num + 1;
         while(_num < _end && _nd->_tab[_num] == nullptr) ++_num;
         return *this;
      }
      iterator operator++(int) { iterator retval = *this; ++(*this); return retval;}
      iterator& operator--()   {
         _num = _num - 1;
         while(_num>0 && _nd->_tab[_num] == nullptr) --_num;
         return *this;
      }
      iterator operator--(int) { iterator retval = *this; --(*this); return retval;}
      bool operator==(iterator other) const noexcept { return _num == other._num;}
      bool operator!=(iterator other) const noexcept { return _num != other._num;;}
      ANode::Ptr operator*() const noexcept {return _nd->_tab[_num];}
      iterator& erase() {
         _nd->_tab[_num] = nullptr;
         _nd->_holes++;
         _num++;
         while(_num < _end && _nd->_tab[_num] == nullptr) ++_num;
         return *this;
      }
      friend class NDArray;
      friend std::ostream& operator<<(std::ostream& os,const iterator& i) {
         return os << '[' << i._num  << ']';
      }
   };
   iterator at(std::size_t ofs) noexcept { return iterator(this,ofs);}
   iterator begin() noexcept {
      auto from = _st;
      while(from < _sz && _tab[from] == nullptr) {
         ++from;
         --_holes;
      }
      _st = from;
      return iterator(this,from);
   }
   iterator end()   noexcept { return iterator(this,_sz);}
   iterator erase(iterator at) noexcept {
      _tab[at._num] = nullptr;
      _holes++;
      at++; // advance the iterator to next legit position
      return at;      
   }
   void eraseSuffix(iterator from) noexcept {
      assert(_holes == 0);
      _sz = from._num;
   }
   ANode::Ptr front() noexcept {
      auto p = _st;
      while(p < _sz && _tab[p] == nullptr) {
         ++p;
         --_holes;
      }
      _st = p;
      assert(p < _sz);
      assert(_tab[p] != nullptr);
      return _tab[p];
   }
   void pop_front() noexcept   {
      auto p = _st;
      while(p < _sz && _tab[p] == nullptr) {
         ++p;
         --_holes;
      }
      _tab[p] = nullptr;
      _st = p+1;
   }
   void sort(auto cmp) {
      mergeSort(_tab,_sz,cmp);
   }
};

class WidthBounded :public Strategy {
protected:
   unsigned _mxw;
   NDArray  _nda;
   NDArray& pullLayer(CQueue<ANode::Ptr>& q);
   std::size_t estimate(CQueue<ANode::Ptr>& q);
public:
   WidthBounded(const unsigned mxw) : Strategy(),_mxw(mxw) {}
   void setWidth(unsigned  mxw) { _mxw = mxw;}
   unsigned getWidth() const  { return _mxw;}
};

class Restricted: public WidthBounded {
   void truncate(NDArray& layer);
public:
   Restricted(const unsigned mxw) : WidthBounded(mxw) {}
   const std::string getName() const { return "Restricted";}
   void compute();
   bool primal() const { return true;}
};


struct NDAction {
   enum Action { Delay,InFront,Noop};
   ANode::Ptr node;
   enum Action act;
};

class Relaxed :public WidthBounded {
   void transferArcs(ANode::Ptr donor,ANode::Ptr receiver);
public:
   Relaxed(const unsigned mxw) : WidthBounded(mxw) {}
   const std::string getName() const { return "Relaxed";}
   void compute();
   std::vector<ANode::Ptr> computeCutSet();
   bool dual() const { return true;}
   NDAction mergePair(ANode::Ptr mNode,ANode::Ptr toMerge[2]);
   ANode::Ptr mergeOne(auto& layer,auto& final);
   template <typename Fun> void mergeLayer(auto& layer,Fun f);
};


template <typename ST,
          class Compare = std::less<double>,
          typename IBL2 = ST(*)(),
          typename LGF  = GNSet(*)(const ST&),
          typename STF  = std::optional<ST>(*)(const ST&,int),
          typename STC  = double(*)(const ST&,int),
          typename SMF  = std::optional<ST>(*)(const ST&,const ST&),
          typename EQSink = bool(*)(const ST&),
          class Equal = std::equal_to<ST>
          >
requires Printable<ST> && Hashable<ST>
class DD :public AbstractDD {
private:
   std::function<ST()> _sti;
   IBL2 _stt;
   LGF _lgf;
   STF _stf;
   STC _stc;
   SMF _smf;
   EQSink _eqs;
   LHashtable<ST> _nmap;
   unsigned _ndId;
   std::function<ANode::Ptr()> _initClosure;
   bool eq(ANode::Ptr f,ANode::Ptr s) const {
      auto fp = static_cast<const Node<ST>*>(f.get());
      auto sp = static_cast<const Node<ST>*>(s.get());
      return Equal{}(fp->get(),sp->get());
   }
   bool eqSink(ANode::Ptr s) const {
      auto sp = static_cast<const Node<ST>*>(s.get());
      return _eqs(sp->get());
   }
   bool   isBetter(double obj1,double obj2) const {
      return Compare{}(obj1,obj2);
   }
   double better(double obj1,double obj2) const {
      return Compare{}(obj1,obj2) ? obj1 : obj2;
   }
   double initialBest() const {
      constexpr auto gr = std::is_same<Compare,std::greater<double>>::value;
      auto v = gr ? -std::numeric_limits<double>::max() : std::numeric_limits<double>::max();
      return v;
   }
   double initialWorst() const {
      constexpr auto gr = std::is_same<Compare,std::greater<double>>::value;
      auto v = !gr ? -std::numeric_limits<double>::max() : std::numeric_limits<double>::max();
      return v;
   }
   void update(Bounds& bnds) const {
      if (_strat->primal())  {
         bnds.setPrimal(DD::better(_trg->getBound(),bnds.getPrimal()));
         bnds.setIncumbent(_trg->beginOptLabels(),_trg->endOptLabels());
         std::cout << "P TIGHEN: " << bnds << "\n";
      }
      else if (_strat->dual() && _exact) {
         bnds.setPrimal(DD::better(_trg->getBound(),bnds.getPrimal()));
         bnds.setIncumbent(_trg->beginOptLabels(),_trg->endOptLabels());
         std::cout << "D TIGHEN: " << bnds << "\n";
      }
   }
   ANode::Ptr makeNode(ST&& state,bool pExact = true) {
      Node<ST>* at = nullptr;
      auto inMap = _nmap.getLoc(state,at);
      if (inMap) {
         at->setExact(at->isExact() & pExact);
         if (at->nbParents() == 0 && at != _root && at != _trg) 
            _an.push_back(at); // The node was created but later removed. Put it back!         
         return at;
      } else {
         auto value = new (_mem) Node<ST>(_mem,std::move(state),_ndId++,pExact);
         _nmap.safeInsertAt(inMap,value);
         _an.push_back(value);
         return value;
      }
   }
   void makeInitFrom(ANode::Ptr src) {
      _initClosure = [theRoot = duplicate(src)]() {
         return theRoot;
      };
   }
   ANode::Ptr init() {
      return _root = _initClosure();
   }
   ANode::Ptr target() {
      return _trg = makeNode(_stt());
   }
   GNSet getLabels(ANode::Ptr src) const {
      auto op = static_cast<const Node<ST>*>(src.get());
      return std::move(_lgf(op->get()));
   }
   ANode::Ptr transition(ANode::Ptr src,int label) {
      auto op = static_cast<const Node<ST>*>(src.get());
      auto vs = _stf(op->get(),label);
      if (vs.has_value()) {
         ANode::Ptr rv = makeNode(std::move(vs.value()),src->isExact());
         rv->setBound(initialBest());
         return rv;
      } else return nullptr;
   }
   double cost(ANode::Ptr src,int label) {
      auto op = static_cast<const Node<ST>*>(src.get());
      auto cVal = _stc(op->get(),label);
      return cVal;
   }
   ANode::Ptr merge(const ANode::Ptr f,const ANode::Ptr s) {
      auto fp = static_cast<const Node<ST>*>(f.get());
      auto sp = static_cast<const Node<ST>*>(s.get());
      auto vs = _smf(fp->get(),sp->get());
      if (vs.has_value()) {
         //std::cout << "B(M1):" << fp->getBound() << "\n";
         //std::cout << "B(M2):" << sp->getBound() << "\n";
         ANode::Ptr rv = makeNode(std::move(vs.value()));
         if (isBetter(fp->getBound(),sp->getBound()))
            rv->setBound(fp->getBound());
         else
            rv->setBound(sp->getBound());                
         //std::cout << "B(RV):" << rv->getBound() << "\n";
         return rv;
      }
      else return nullptr;
   }
public:
   DD(std::function<ST()> sti,IBL2 stt,LGF lgf,STF stf,STC stc,SMF smf,EQSink eqs,const GNSet& labels)
      : AbstractDD(labels),
        _sti(sti),
        _stt(stt),
        _lgf(lgf),
        _stf(stf),
        _stc(stc),
        _smf(smf),
        _eqs(eqs),
        _nmap(_mem,200000),
        //_nmap(_mem,1024),
        _ndId(0)
   {
      _baseline = _mem->mark();
      _initClosure = [this]() {
         return makeNode(_sti());
      };
   }
   ~DD() { DD::reset();}
   template <class... Args>
   static AbstractDD::Ptr makeDD(Args&&... args) {
      return AbstractDD::Ptr(new DD(std::forward<Args>(args)...));
   }
   void printNode(ANode::Ptr n) const {
      auto sp = static_cast<const Node<ST>*>(n.get());
      sp->print(std::cout);
      std::cout << std::endl;
   }
   AbstractDD::Ptr duplicate() {
      return AbstractDD::Ptr(new DD(_sti,_stt,_lgf,_stf,_stc,_smf,_eqs,_labels));
   }
   ANode::Ptr duplicate(const ANode::Ptr src) {
      Node<ST>* at = nullptr;
      auto sp = static_cast<const Node<ST>*>(src.get());
      auto inMap = _nmap.getLoc(sp->get(),at);
      if (inMap) {
         return at;
      } else {
         auto nn = new (_mem) Node<ST>(_mem,_ndId++,*sp);
         _nmap.safeInsertAt(inMap,nn);
         _an.push_back(nn);
         return nn;
      }
   }
   void reset() {
      _ndId = 0;
      _nmap.doOnAll([](Node<ST>* np) {
         np->~Node<ST>();
      });
      _nmap.clear();      
      _an.clear();
      _mem->clear(_baseline);
      _root = _trg = nullptr;
   }
};

#endif
