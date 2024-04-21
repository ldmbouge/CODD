#ifndef __UF_H
#define __UF_H

#include <iostream>
#include <memory>
#include <list>

template <class T>
class UnionFind {
public:
   class Node :public std::enable_shared_from_this<Node> {
      friend class UnionFind<T>;
      std::weak_ptr<Node>  _parent;  // use weak pointers since the list actually owns all nodes.
      T                     _value;  // actual data.
      int                      _sz;  // how many nodes in this set. (Only correct for root!)
   public:
      typedef std::shared_ptr<Node> Ptr;
      Node(const T& v) : _value(v),_sz(1)       {}
      Node(T&& v) : _value(std::move(v)),_sz(1) {}
   };
private:  // This is a list of shared pointers to Node instances defined within the UnionFind<T> class.
   std::list<typename Node::Ptr> _all;   
public:
   UnionFind() : _all() {}
   Node::Ptr makeSet(const T& v) {
      auto p = std::make_shared<Node>(v);
      _all.push_back(p);
      return p;
   }
   Node::Ptr setFor(std::shared_ptr<Node> v) {
      // Standard iterative version without path-compression.
      // std::shared_ptr<Node> c = v;
      // std::shared_ptr<Node> p = nullptr;
      // while ((p = c->_parent.lock()) != nullptr)
      //    c = p;
      // return c;
      // ------------------------------------------------------------
      // Recursive version with path compression (easier to do recursively)
      if (v->_parent.use_count() == 0) {  // if parent is null, there is no use.
         return v;
      } else {
         std::shared_ptr<Node> root = setFor(v->_parent.lock());
         v->_parent = root;
         return root;
      }
   }
   Node::Ptr merge(const Node::Ptr& s0,const Node::Ptr& s1) {
      auto r0 = setFor(s0);
      auto r1 = setFor(s1);
      if (r0 == r1) return r0;
      if (r0->_sz > r1->_sz) {
         r1->_parent = r0;
         r0->_sz += r1->_sz;
      } else {
         r0->_parent = r1;
         r1->_sz += r0->_sz;
      }
      return r0;
   }   
   friend std::ostream& operator<<(std::ostream& os,const Node::Ptr& v) {
      std::shared_ptr<Node> c = v;
      std::shared_ptr<Node> p = nullptr;
      // acquire the shared_ptr to the parent (lock) and assign to p. Then test for null. 
      while ((p = c->_parent.lock()) != nullptr) { 
         os << "node[" << c.use_count() << "](" << c.get() << ',' << c->_value << ") --> ";
         c = p;         
      }
      // at root of star, print the size. 
      os << "node[" << c.use_count() << "](" << c.get() << ',' << c->_value << ") :: " << c->_sz;
      return os;
   }
};

#endif
