#ifndef __DD_HPP__
#define __DD_HPP__

#include <algorithm>
#include "node.hpp"
#include <vector>
#include <list>
#include <optional>
#include <set>
#include "hashtable.hpp"

class Strategy;

class AbstractDD {
protected:
   Pool::Ptr _mem;
   ANode::Ptr _root;
   ANode::Ptr _trg;
   std::set<int> _labels;
   std::vector<ANode::Ptr> _an;
   void addArc(Edge::Ptr e);
   friend class Strategy;
   friend class Exact;
   friend class Restricted;
   friend class Relaxed;
private:
   Strategy* _strat;
protected:
   virtual bool eq(ANode::Ptr f,ANode::Ptr s) const = 0;
   virtual bool neq(ANode::Ptr f,ANode::Ptr s) const = 0;
   virtual double better(double obj1,double obj2) const = 0;
   virtual double initialBest() const = 0;
   void computeBest();
   void saveGraph(std::ostream& os,std::string gLabel);
public:
   typedef std::shared_ptr<AbstractDD> Ptr;
   AbstractDD(const std::set<int>& labels);
   virtual ~AbstractDD();
   virtual ANode::Ptr init() = 0;
   virtual ANode::Ptr target() = 0;
   virtual ANode::Ptr transition(ANode::Ptr src,int label) = 0;
   virtual ANode::Ptr merge(const ANode::Ptr first,const ANode::Ptr snd) = 0;
   virtual double cost(ANode::Ptr src,int label) = 0;
   void compute();
   void print(std::ostream& os,std::string gLabel);
   void setStrategy(Strategy* s);
   void display(std::string gLabel);
   virtual AbstractDD::Ptr duplicate() = 0;
};

class Strategy {
protected:
   AbstractDD* _dd;
   friend class AbstractDD;
   std::set<int> remainingLabels(ANode::Ptr p);
public:
   Strategy() : _dd(nullptr) {}
   virtual void compute() {}
};

class Exact:public Strategy {
public:
   Exact() : Strategy() {}
   void compute();
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
};

class Relaxed :public WidthBounded {
   void transferArcs(ANode::Ptr donor,ANode::Ptr receiver);
   std::list<ANode::Ptr> mergeLayer(std::list<ANode::Ptr>& layer);
public:
   Relaxed(const unsigned mxw) : WidthBounded(mxw) {}
   void compute();
};

   
template <typename ST,
          class Compare = std::less<double>,
          typename IBL1 = ST(*)(),
          typename IBL2 = ST(*)(),
          typename STF  = std::optional<ST>(*)(const ST&,int),
          typename STC  = double(*)(const ST&,int),
          typename SMF  = std::optional<ST>(*)(const ST&,const ST&),
          class Equal = std::equal_to<ST>,
          class NotEqual=std::not_equal_to<ST>>
requires Printable<ST> && Hashable<ST>
class DD :public AbstractDD {
private:
   IBL1 _sti;
   IBL2 _stt;
   STF _stf;
   STC _stc;
   SMF _smf;
   Hashtable<ST,ANode::Ptr> _nmap;
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
   double better(double obj1,double obj2) const {
      return Compare{}(obj1,obj2) ? obj1 : obj2;
   }
   double initialBest() const {
      constexpr auto gr = std::is_same<Compare,std::greater<double>>::value;
      auto v = gr ? std::numeric_limits<double>::min() : std::numeric_limits<double>::max();
      return v;
   }
   ANode::Ptr makeNode(ST&& state,bool pExact = true) {
      ANode::Ptr at = nullptr;
      auto inMap = _nmap.get(state,at);
      if (inMap) {
         at->setExact(at->isExact() & pExact);
         return at;
      } else {
         auto value = new (_mem) Node<ST>(_mem,std::move(state));
         value->setExact(pExact);
         _nmap.insert(value->get(),value);
         _an.push_back(value);
         return value;
      }
   }
   ANode::Ptr init() {
      return _root = makeNode(_sti());
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
   DD(IBL1 sti,IBL2 stt,STF stf,STC stc,SMF smf,const std::set<int>& labels)
      : AbstractDD(labels),
        _sti(sti),
        _stt(stt),
        _stf(stf),
        _stc(stc),
        _smf(smf),
        _nmap(_mem,128)
   {}
   ~DD() {
      _nmap.clear();
   }
   template <class... Args>
   static AbstractDD::Ptr makeDD(Args&&... args) {
      return std::make_shared<DD>(std::forward<Args>(args)...);
   }
   AbstractDD::Ptr duplicate() {
      return std::make_shared<DD>(_sti,_stt,_stf,_stc,_smf,_labels);
   }
};

#endif
