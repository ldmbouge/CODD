#include "node.hpp"
#include <iostream>

std::ostream& operator<<(std::ostream& os,const ANode& s)
{
   s.print(os);
   return os;
}

std::ostream& operator<<(std::ostream& os,const Edge& e)
{
   return os << "E<" << *e._from << " --- " << e._obj << " ---> "  << *e._to;
}

void ANode::addArc(Edge::Ptr ep)
{
   if (ep->_from == this)
      ep->_fix = _children.push_back(ep);
   else if (ep->_to == this)
      ep->_tix = _parents.push_back(ep);
}
