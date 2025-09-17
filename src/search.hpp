#ifndef __SEARCH_HPP__
#define __SEARCH_HPP__

#include <functional>
#include "dd.hpp"

struct QNode {
    ANode::Ptr node;
    double    bound;
    friend std::ostream& operator<<(std::ostream& os,const QNode& q) {
        return os << "QNode[(" << q.node->getId() << ','
                  << q.node->getBound() << ','
                  << q.node->getBackwardBound() << ")," << q.bound << "]";
    }
};

inline
std::vector<ANode::Ptr> filterLocal(Bounds& bnds, AbstractDD::Ptr dd, std::vector<ANode::Ptr> nodes)
{
    std::vector<ANode::Ptr> survived;
    for(auto n: nodes) {
        double localDual = dd->local(n, LocalContext::BBCtx);
        if(!dd->isBetterEQ(bnds.getPrimal(), n->getBound() + localDual))
            survived.push_back(n);
    }
    return survived;
}

class BAndB
{
protected:
    AbstractDD::Ptr _theDD;
    const unsigned    _mxw;
    std::function<bool(double)> _timeLimit;
public:
   BAndB(AbstractDD::Ptr dd,const unsigned width)
      : _theDD(dd),_mxw(width),_timeLimit(nullptr) {}

    virtual ~BAndB() {}
    virtual void search(Bounds & bnds) = 0;
    void setTimeLimit(std::function<bool(double)> lim) {_timeLimit = lim;}
};  

#endif
