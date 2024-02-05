#include "node.hpp"
#include <iostream>

void print(const ANList& l) {
   std::cout << l << "\n";
}

std::ostream& operator<<(std::ostream& os,const ANode& s)
{
   s.print(os);
   return os;
}

std::ostream& operator<<(std::ostream& os,const Edge& e)
{
   return os << "E<" << *e._from << " --- " << e._obj << "(" << e._lbl << ") ---> "  << *e._to;
}

ANode::ANode(Pool::Ptr mem,unsigned nid,bool exact)
   : _parents(mem,2),
     _children(mem,2),
     _optLabels(mem),
     _bound(0),
     _bbound(0),
     _layer(0),
     _exact(exact),
     _nid(nid),
     _next(nullptr),
     _prev(nullptr)
     
{}

ANode::ANode(Pool::Ptr mem,unsigned nid,const ANode& o,bool exact)
   : _parents(mem,2),
     _children(mem,2),
     _optLabels(mem,o._optLabels),
     _bound(o._bound),
     _bbound(o._bbound),
     _layer(0),
     _exact(exact),
     _nid(nid),
     _next(nullptr),
     _prev(nullptr)
    
{}

/*
ANode::~ANode()
{}
*/

void ANode::addArc(Edge::Ptr ep)
{
   if (ep->_from == this)
      ep->_fix = _children.push_back(ep);
   else if (ep->_to == this)
      ep->_tix = _parents.push_back(ep);
}

void ANode::disconnect()
{
   for(auto e : _parents) {
      assert(e->_from->_children[e->_fix] == e);
      e->_from->_children.remove(e->_fix,[e](Edge::Ptr o) { o->_fix = e->_fix;});
   }
   _parents.clear();
   for(auto e : _children) {
      assert(e->_to->_parents[e->_tix] == e);
      e->_to->_parents.remove(e->_tix,[e](Edge::Ptr o) { o->_tix = e->_tix;});
   }
   _children.clear();
}



