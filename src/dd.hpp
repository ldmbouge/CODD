#ifndef __DD_HPP__
#define __DD_HPP__

#include <algorithm>
#include "node.hpp"
#include <vector>
#include <list>
#include <optional>
#include <set>
#include <functional>
#include "hashtable.hpp"
#include "util.hpp"

class Strategy;
class AbstractDD;

class Bounds {
   double _primal;
   double _dual;
   std::vector<int> _inc;
public:
   Bounds(std::shared_ptr<AbstractDD> dd);
   void setPrimal(double p) { _primal = p;}
   void setDual(double p)   { _dual = p;}
   double getPrimal() const { return _primal;}
   double getDual() const   { return _dual;}
   void setIncumbent(auto begin,auto end) {
      _inc.clear();
      for(auto it = begin;it != end;it++)
         _inc.push_back(*it);
   }
   friend std::ostream& operator<<(std::ostream& os,const Bounds& b) {
      return os << "<P:" << b._primal << "," << " D:" << b._dual << ", INC:" << b._inc << ">";
   }
};


class AbstractDD {
protected:
   Pool::Ptr _mem;
   ANode::Ptr _root;
   ANode::Ptr _trg;
   std::set<int> _labels;
   std::vector<ANode::Ptr> _an;
   bool _exact;
   PoolMark _baseline;
   void addArc(Edge::Ptr e);
   friend class Strategy;
   friend class Exact;
   friend class Restricted;
   friend class Relaxed;
   Strategy* _strat;
   virtual bool eq(ANode::Ptr f,ANode::Ptr s) const = 0;
   virtual bool neq(ANode::Ptr f,ANode::Ptr s) const = 0;
   void computeBest();
   void saveGraph(std::ostream& os,std::string gLabel);
public:
   typedef std::shared_ptr<AbstractDD> Ptr;
   AbstractDD(const std::set<int>& labels);
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
   double currentOpt() const { return _trg->getBound();}
   void compute();
   std::vector<ANode::Ptr> computeCutSet();
   void print(std::ostream& os,std::string gLabel);
   void setStrategy(Strategy* s);
   void display(std::string gLabel);
   bool isExact() const { return _exact;}
   virtual AbstractDD::Ptr duplicate() = 0;
   virtual void makeInitFrom(ANode::Ptr src) {}
};

class Strategy {
protected:
   AbstractDD* _dd;
   friend class AbstractDD;
   std::set<int> remainingLabels(ANode::Ptr p);
public:
   Strategy() : _dd(nullptr) {}
   virtual void compute() {}
   virtual std::vector<ANode::Ptr> computeCutSet() { return std::vector<ANode::Ptr> {};}
   virtual bool primal() const { return false;}
   virtual bool dual() const { return false;}
};

class Exact:public Strategy {
public:
   Exact() : Strategy() {}
   void compute();
   bool primal() const { return true;}
   bool dual() const { return true;}
};

template <class T> class CQueue;

class WidthBounded :public Strategy {
protected:
   const unsigned _mxw;
   std::list<ANode::Ptr> pullLayer(CQueue<ANode::Ptr>& q);
public:
   WidthBounded(const unsigned mxw) : Strategy(),_mxw(mxw) {}
};

class Restricted: public WidthBounded {
   void truncate(std::list<ANode::Ptr>& layer);
public:
   Restricted(const unsigned mxw) : WidthBounded(mxw) {}
   void compute();
   bool primal() const { return true;}
};

class Relaxed :public WidthBounded {
   void transferArcs(ANode::Ptr donor,ANode::Ptr receiver);
   std::list<ANode::Ptr> mergeLayer(std::list<ANode::Ptr>& layer);
public:
   Relaxed(const unsigned mxw) : WidthBounded(mxw) {}
   void compute();
   std::vector<ANode::Ptr> computeCutSet();
   bool dual() const { return true;}
};


template <typename ST,
          class Compare = std::less<double>,
          //typename IBL1 = ST(*)(),
          typename IBL2 = ST(*)(),
          typename STF  = std::optional<ST>(*)(const ST&,int),
          typename STC  = double(*)(const ST&,int),
          typename SMF  = std::optional<ST>(*)(const ST&,const ST&),
          class Equal = std::equal_to<ST>,
          class NotEqual=std::not_equal_to<ST>>
requires Printable<ST> && Hashable<ST>
class DD :public AbstractDD {
private:
   std::function<ST()> _sti;
   //IBL1 _sti;
   IBL2 _stt;
   STF _stf;
   STC _stc;
   SMF _smf;
   Hashtable<ST,ANode::Ptr> _nmap;
   unsigned _ndId;
   std::function<ANode::Ptr()> _initClosure;
   bool eq(ANode::Ptr f,ANode::Ptr s) const {
      auto fp = dynamic_cast<const Node<ST>*>(f.get());
      auto sp = dynamic_cast<const Node<ST>*>(s.get());
      return Equal{}(fp->get(),sp->get());
   }
   bool neq(ANode::Ptr f,ANode::Ptr s) const {
      auto fp = dynamic_cast<const Node<ST>*>(f.get());
      auto sp = dynamic_cast<const Node<ST>*>(s.get());
      return NotEqual{}(fp->get(),sp->get());
   }
   bool   isBetter(double obj1,double obj2) const {
      return Compare{}(obj1,obj2);
   }
   double better(double obj1,double obj2) const {
      return Compare{}(obj1,obj2) ? obj1 : obj2;
   }
   double dualBetter(double obj1,double obj2) const {
      return Compare{}(obj1,obj2) ? obj2 : obj1;
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
      }
      else if (_strat->dual()) {
         bnds.setDual(DD::dualBetter(_trg->getBound(),bnds.getDual()));
         if (_exact) {
            bnds.setPrimal(DD::better(_trg->getBound(),bnds.getPrimal()));
            bnds.setIncumbent(_trg->beginOptLabels(),_trg->endOptLabels());
         }
      }
   }
   ANode::Ptr makeNode(ST&& state,bool pExact = true) {
      ANode::Ptr at = nullptr;
      auto inMap = _nmap.get(state,at);
      if (inMap) {
         at->setExact(at->isExact() & pExact);
         return at;
      } else {
         auto value = new (_mem) Node<ST>(_mem,std::move(state),_ndId++);
         value->setExact(pExact);
         _nmap.insert(value->get(),value);
         _an.push_back(value);
         return value;
      }
   }
   void makeInitFrom(ANode::Ptr src) {
      _initClosure = std::function<ANode::Ptr()>([theRoot = duplicate(src)]() {
         return theRoot;
      });
   }
   ANode::Ptr init() {
      return _root = _initClosure();
   }
   ANode::Ptr target() {
      return _trg = makeNode(_stt());
   }
   ANode::Ptr transition(ANode::Ptr src,int label) {
      auto op = dynamic_cast<const Node<ST>*>(src.get());
      auto vs = _stf(op->get(),label);
      if (vs.has_value())
         return makeNode(std::move(vs.value()),src->isExact());
      else return nullptr;
   }
   double cost(ANode::Ptr src,int label) {
      auto op = dynamic_cast<const Node<ST>*>(src.get());
      auto cVal = _stc(op->get(),label);
      return cVal;
   }
   ANode::Ptr merge(const ANode::Ptr f,const ANode::Ptr s) {
      auto fp = dynamic_cast<const Node<ST>*>(f.get());
      auto sp = dynamic_cast<const Node<ST>*>(s.get());
      auto vs = _smf(fp->get(),sp->get());
      if (vs.has_value())
         return makeNode(std::move(vs.value()));
      else return nullptr;
   }
public:
   DD(std::function<ST()> sti,IBL2 stt,STF stf,STC stc,SMF smf,const std::set<int>& labels)
      : AbstractDD(labels),
        _sti(sti),
        _stt(stt),
        _stf(stf),
        _stc(stc),
        _smf(smf),
        _nmap(_mem,128),
        _ndId(0)
   {
      _baseline = _mem->mark();
      _initClosure = std::function<ANode::Ptr()>([this]() {
         return makeNode(_sti());
      });
   }
   ~DD() { DD::reset();}
   template <class... Args>
   static AbstractDD::Ptr makeDD(Args&&... args) {
      return AbstractDD::Ptr(new DD(std::forward<Args>(args)...));
   }
   AbstractDD::Ptr duplicate() {
      return AbstractDD::Ptr(new DD(_sti,_stt,_stf,_stc,_smf,_labels));
   }
   ANode::Ptr duplicate(const ANode::Ptr src) {
      ANode::Ptr at = nullptr;
      auto sp = dynamic_cast<const Node<ST>*>(src.get());
      auto inMap = _nmap.get(sp->get(),at);
      if (inMap) {
         return at;
      } else {
         auto nn = new (_mem) Node<ST>(_mem,_ndId++,*sp);
         _nmap.insert(nn->get(),nn);
         _an.push_back(nn);
         return nn;
      }
   }
   void reset() {
      _ndId = 0;
      _nmap.clear();
      for(auto n : _an)
         n->~ANode();
      _an.clear();
      _mem->clear(_baseline);
      _root = _trg = nullptr;
   }
};

#endif
