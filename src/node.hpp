#ifndef __NODE_H
#define __NODE_H

#include <functional>
#include <iterator>
#include "vec.hpp"

template<typename T>
concept Printable = requires(std::ostream& os, const T& msg)
{
   {os << msg};
};

template<typename T>
concept Hashable = requires(const T& msg)
{
   {std::hash<T>{}(msg)};
};

template <Printable T> void print(std::ostream& os,const T& msg)
{
   os << msg;
}

class ANode;

struct Edge {
   typedef handle_ptr<Edge> Ptr;
   handle_ptr<ANode>  _from,_to;
   Vec<Edge::Ptr>::IdxType _fix,_tix;
   double _obj;
   int    _lbl;
   Edge(handle_ptr<ANode> fp,handle_ptr<ANode> tp,int lbl)
      : _from(fp),_to(tp),_fix(-1),_tix(-1),_obj(0),_lbl(lbl) {}
   friend std::ostream& operator<<(std::ostream& os,const Edge& e);
};

class ANList;
class ANode {
public:
   typedef handle_ptr<ANode> Ptr;
protected:
   Vec<Edge::Ptr,unsigned> _parents;
   Vec<Edge::Ptr,unsigned> _children;
   Vec<int,unsigned>       _optLabels;
   double                  _bound;
   double                  _bbound;
   unsigned                _layer:31; // will be used in restricted / (relaxed?)
   unsigned                _exact:1;  // true if node is exact
   unsigned                _nid;
   ANode::Ptr              _next,_prev;
   void addArc(Edge::Ptr ep);
public:
   friend class AbstractDD;
   friend class Relaxed;
   friend class ANList;
   ANode(Pool::Ptr mem,unsigned nid,bool exact);
   ANode(Pool::Ptr mem,unsigned nid,const ANode& o,bool exact);
   //~ANode();
   void print(std::ostream& os) const {}
   void clearParents() { _parents.clear();}
   void clearKids() { _children.clear();}
   void setLayer(unsigned l) noexcept {  _layer = l;}
   auto getLayer() const noexcept     { return _layer;}
   void setExact(bool ex) noexcept    { _exact = ex;}
   auto isExact() const noexcept      { return _exact == 1;}
   auto nbParents() const noexcept    { return _parents.size();}
   auto nbChildren() const noexcept   { return _children.size();}
   auto beginPar()  { return _parents.begin();}
   auto endPar()    { return _parents.end();}
   auto beginKids() { return _children.begin();}
   auto endKids()   { return _children.end();}
   auto beginOptLabels() { return _optLabels.begin();}
   auto endOptLabels() { return _optLabels.end();}
   void setBound(double b) { _bound = b;}
   void setBackwardBound(double b) { _bbound = b;}
   const auto getBound() const { return _bound;}
   const auto getBackwardBound() const { return _bbound;}
   void setIncumbent(auto begin,auto end) {
      for(auto it = begin;it != end;it++)
         _optLabels.push_back(*it);
   }
   const auto getId() const noexcept { return _nid;}
   void disconnect();
   friend std::ostream& operator<<(std::ostream& os,const ANode& s);
   friend std::ostream& operator<<(std::ostream& os,const ANList& v);
};

class ANList {
   ANode::Ptr _head,_tail;
public:
   ANList() : _head(nullptr),_tail(nullptr) {}
   ANList(const ANList& l) : _head(l._head),_tail(l._tail) {}
   ~ANList() { _head = _tail = nullptr;}
   std::size_t size() const noexcept {
      ANode::Ptr cur = _head;
      std::size_t s = 0;
      while(cur) {
         cur = cur->_next;
         ++s;
      }
      return s;
   }
   void push_back(ANode::Ptr n) noexcept {
      assert(n->_next == nullptr && n->_prev == nullptr);
      if (_head==nullptr) {
         _head = _tail = n;
         n->_next = n->_prev = nullptr;
      } else {
         n->_prev = _tail;
         n->_next = nullptr;
         if (_tail) _tail->_next = n;
         _tail = n;
      }
   }
   void remove(ANode::Ptr n) noexcept {
      //assert(confirmMembership(n)==true);
      const ANode::Ptr pv = n->_prev, nx = n->_next;
      (pv ? pv->_next : _head) = nx;
      (nx ? nx->_prev : _tail) = pv;
      n->_prev = n->_next = nullptr;
   }
   bool confirmMembership(ANode::Ptr n) {
      ANode::Ptr cur = _head;
      while (cur && cur != n)
         cur = cur->_next;
      return cur == n;
   }
   void clear() noexcept {
      _head = _tail = nullptr;
   }
   class iterator { 
      ANode::Ptr _data;
      iterator(ANode::Ptr d) : _data(d) {}
   public:
      using iterator_category = std::input_iterator_tag;
      using value_type = ANode::Ptr;
      using difference_type = long;
      using pointer = ANode::Ptr*;
      using reference = ANode::Ptr&;
      iterator& operator++()   { _data = _data->_next; return *this;}
      iterator operator++(int) { iterator retval = *this; ++(*this); return retval;}
      iterator& operator--()   { _data = _data->_prev; return *this;}
      iterator operator--(int) { iterator retval = *this; --(*this); return retval;}
      bool operator==(iterator other) const noexcept {return _data == other._data;}
      bool operator!=(iterator other) const noexcept {return !(*this == other);}
      ANode::Ptr operator*() const noexcept { return _data;}
      friend class ANList;
   };
   iterator begin() const noexcept { return iterator(_head);}
   iterator end()   const noexcept { return iterator(nullptr);}
   friend std::ostream& operator<<(std::ostream& os,const ANList& v) {
      os << "[";
      ANode::Ptr cur = v._head;
      while (cur) {
         os << *cur;
         if (cur->_next)
            os << ',';
         cur = cur->_next;
      }
      return os << "]";
   }
};

void print(const ANList& l);

template <typename T> requires Printable<T> && Hashable<T>
class Node :public ANode {
   T _val;  
public:
   typedef handle_ptr<Node<T>> Ptr;
   Node(Pool::Ptr mem,T&& v,unsigned nid,bool exact) : ANode(mem,nid,exact),_val(std::move(v)) {}
   Node(Pool::Ptr mem,unsigned nid,const Node<T>& o) : ANode(mem,nid,o,true),_val(o._val) {}
   const T& get() const {return _val;}
   void print(std::ostream& os) const {
      os << _nid << ',' << _val << ",B=" << _bound << ",BB=" << _bbound << ",LBLS:[";
      for(auto i=0u;i < _optLabels.size();i++)
         os << _optLabels[i] << " ";
      os << "]";
   }
};


#endif
